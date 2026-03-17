from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import Settings
from vectorstore import (
    HybridVectorStore,
    PgVectorStore,
    collection_from_plan_name,
    load_pdf_chunks,
)

from agents.schemas import CompanyPlan, WorkflowState
from pathlib import Path

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover - optional dependency
    TavilyClient = None  # type: ignore[assignment]


def safe_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _clean_snippet(text: str, max_chars: int = 260) -> str:
    """문장 단위로 잘라 깔끔한 핵심 문장만 남긴다."""
    if not text:
        return ""
    # 공백 정리
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        snippet = cleaned
    else:
        snippet = cleaned[:max_chars]
        # 문장 경계 찾기 (한글 '다.' / '. ' / '!' / '?')
        candidates = [
            snippet.rfind("다."),
            snippet.rfind(". "),
            snippet.rfind("! "),
            snippet.rfind("? "),
        ]
        end_idx = max(candidates)
        if end_idx != -1:
            snippet = snippet[: end_idx + 2]
    return snippet.strip()


def _fetch_news_tavily(query: str, max_results: int) -> list[dict[str, str]]:
    """Tavily 기반 뉴스/웹 검색. 실패 시 빈 리스트 반환."""
    if TavilyClient is None:
        return []
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        client = TavilyClient(api_key=api_key)
        response: dict[str, Any] = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            include_images=False,
            include_raw_content=False,
        )
        results: list[dict[str, str]] = []
        for item in response.get("results", [])[:max_results]:
            results.append(
                {
                    "title": safe_content(item.get("title")),
                    "link": safe_content(item.get("url")),
                    "pub_date": safe_content(
                        item.get("published_date") or item.get("date")
                    ),
                    "source": safe_content(item.get("source") or item.get("url")),
                }
            )
        return results
    except Exception:
        return []


def _fetch_news_rss(query: str, max_results: int) -> list[dict[str, str]]:
    """Google News RSS 기반 뉴스 검색 (Tavily 실패 시 백업)."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw = response.read()
        root = ET.fromstring(raw)
        items = root.findall(".//item")
        results: list[dict[str, str]] = []
        for item in items[:max_results]:
            results.append(
                {
                    "title": safe_content(item.findtext("title")),
                    "link": safe_content(item.findtext("link")),
                    "pub_date": safe_content(item.findtext("pubDate")),
                    "source": safe_content(item.findtext("source")),
                }
            )
        return results
    except Exception:
        return []


def fetch_news(query: str, max_results: int) -> list[dict[str, str]]:
    """Tavily 우선, 실패 시 RSS로 뉴스/웹 검색."""
    results = _fetch_news_tavily(query=query, max_results=max_results)
    if results:
        return results
    return _fetch_news_rss(query=query, max_results=max_results)


def collect_news(
    plan: CompanyPlan,
    max_results_per_query: int,
) -> tuple[list[dict[str, str]], dict[str, bool]]:
    news_items: list[dict[str, str]] = []
    coverage = {"positive": False, "negative": False}
    for query in plan.positive_queries:
        for item in fetch_news(query=query, max_results=max_results_per_query):
            item["query_type"] = "positive"
            item["query"] = query
            news_items.append(item)
            coverage["positive"] = True
    for query in plan.negative_queries:
        for item in fetch_news(query=query, max_results=max_results_per_query):
            item["query_type"] = "negative"
            item["query"] = query
            news_items.append(item)
            coverage["negative"] = True
    return news_items, coverage


def format_sources(chunks: list[dict[str, str]]) -> str:
    lines = []
    for idx, source in enumerate(chunks, start=1):
        lines.append(
            f"{idx}. {source['filename']} p.{source['page']} - {source['snippet']}"
        )
    return "\n".join(lines)


def run_rag(
    plan: CompanyPlan, settings: Settings, per_company_page_limit: int
) -> tuple[str, list[dict[str, str]], str]:
    pdf_paths = sorted(plan.pdf_dir.glob("*.pdf"))
    docs, metadatas, page_count = load_pdf_chunks(
        pdf_paths=pdf_paths,
        page_limit=per_company_page_limit,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not docs:
        return "RAG 문서가 비어 있습니다.", [], "tfidf"
    collection = collection_from_plan_name(plan.name)
    pg_store = PgVectorStore(
        settings=settings, model_name=settings.embedding_model_name
    )
    use_pgvector = pg_store.ensure_tables() and pg_store.add_documents(
        collection=collection,
        docs=docs,
        metadatas=metadatas,
    )
    backend = pg_store.backend if use_pgvector else "tfidf"
    fallback_store: HybridVectorStore | None = None
    if not use_pgvector:
        fallback_store = HybridVectorStore(model_name=settings.embedding_model_name)
        fallback_store.add_documents(docs=docs, metadatas=metadatas)
        fallback_store.build()
        backend = fallback_store.backend
    selected_chunks: list[dict[str, str]] = []
    seen = set()
    for query in plan.rag_queries:
        if use_pgvector:
            retrieved = pg_store.similarity_search(
                collection=collection,
                query=query,
                top_k=settings.rag_top_k,
            )
        else:
            retrieved = fallback_store.similarity_search(
                query=query, top_k=settings.rag_top_k
            )
        for chunk in retrieved:
            key = (chunk.metadata["source"], chunk.metadata["page"])
            if key in seen:
                continue
            seen.add(key)
            src_path = Path(str(chunk.metadata["source"]))
            selected_chunks.append(
                {
                    "source": str(chunk.metadata["source"]),
                    "filename": src_path.name,
                    "page": str(chunk.metadata["page"]),
                    "snippet": _clean_snippet(chunk.content),
                }
            )
            if len(selected_chunks) >= 8:
                break
        if len(selected_chunks) >= 8:
            break
    summary_lines = [
        f"기업: {plan.name}",
        f"사용 PDF 수: {len(pdf_paths)}",
        f"사용 페이지 수: {page_count}",
        f"임베딩 백엔드: {backend}",
        f"저장 컬렉션: {collection}",
        "핵심 근거:",
        format_sources(selected_chunks) if selected_chunks else "근거 없음",
    ]
    return "\n".join(summary_lines), selected_chunks, backend


def invoke_llm(settings: Settings, prompt: str) -> str:
    if settings.openai_api_key:
        model = ChatOpenAI(model=settings.llm_model, temperature=0)
        response = model.invoke([HumanMessage(content=prompt)])
        return safe_content(response.content)
    return prompt[:1400]


def summarize_tech(
    settings: Settings,
    company_name: str,
    rag_summary: str,
    news_items: list[dict[str, str]],
) -> str:
    news_text = "\n".join(
        [
            f"- ({item['query_type']}) {item['title']} | {item['source']} | {item['pub_date']} | {item['link']}"
            for item in news_items[:12]
        ]
    )
    prompt = (
        f"{company_name}의 기술 경쟁력 핵심 요약을 작성하세요.\n"
        "요구사항: 핵심기술, 강점, 한계, 차별화 포인트를 분리하세요.\n"
        f"RAG 근거:\n{rag_summary}\n"
        f"최신 동향:\n{news_text}\n"
    )
    return invoke_llm(settings=settings, prompt=prompt)


def build_swot(
    settings: Settings, lg_text: str, catl_text: str, market: str
) -> tuple[dict[str, dict[str, list[str]]], str]:
    if not settings.openai_api_key:
        swot = {
            "LG에너지솔루션": {
                "S": ["북미 생산거점"],
                "W": ["원가경쟁"],
                "O": ["ESS 확대"],
                "T": ["수요 둔화"],
            },
            "CATL": {
                "S": ["LFP 규모"],
                "W": ["지역 규제"],
                "O": ["나트륨이온"],
                "T": ["무역 장벽"],
            },
        }
        comparison = "LG에너지솔루션은 북미 중심 공급망, CATL은 LFP/나트륨이온 중심 비용경쟁력이 핵심 차별점이다."
        return swot, comparison
    prompt = (
        "LG에너지솔루션과 CATL 비교 SWOT을 JSON으로 생성하세요.\n"
        '반드시 {"LG에너지솔루션":{"S":[],"W":[],"O":[],"T":[]},"CATL":{"S":[],"W":[],"O":[],"T":[]}} 형식만 출력하세요.\n'
        f"LG 정보:\n{lg_text}\n\nCATL 정보:\n{catl_text}\n\n시장정보:\n{market}\n"
    )
    raw = invoke_llm(settings=settings, prompt=prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        payload = raw[start : end + 1]
        swot = json.loads(payload)
    except Exception:
        swot = {
            "LG에너지솔루션": {
                "S": ["북미 생산거점"],
                "W": ["원가경쟁"],
                "O": ["ESS 확대"],
                "T": ["수요 둔화"],
            },
            "CATL": {
                "S": ["LFP 규모"],
                "W": ["지역 규제"],
                "O": ["나트륨이온"],
                "T": ["무역 장벽"],
            },
        }
    comparison_prompt = (
        "양사의 전략·기술 비교를 표 형식 문장으로 작성하세요.\n"
        f"SWOT 데이터:\n{json.dumps(swot, ensure_ascii=False)}"
    )
    comparison = invoke_llm(settings=settings, prompt=comparison_prompt)
    return swot, comparison


def build_report(state: WorkflowState) -> str:
    lg_sources = state.get("lg_sources", [])
    catl_sources = state.get("catl_sources", [])
    market_sources = state.get("market_sources", [])
    all_news = state.get("lg_news", [])[:4] + state.get("catl_news", [])[:4]
    reference_lines: list[str] = []
    for src in lg_sources:
        filename = Path(src["source"]).name
        reference_lines.append(
            f"- [기관보고서] LG PDF ({filename}, p.{src['page']})"
        )
    for src in catl_sources:
        filename = Path(src["source"]).name
        reference_lines.append(
            f"- [기관보고서] CATL PDF ({filename}, p.{src['page']})"
        )
    for src in market_sources:
        filename = Path(src["source"]).name
        reference_lines.append(
            f"- [기관보고서] Market PDF ({filename}, p.{src['page']})"
        )
    for item in all_news:
        reference_lines.append(
            f"- [웹페이지] {item['title']} ({item['link']}, {item['pub_date']})"
        )
    if not reference_lines:
        reference_lines = ["- [웹페이지] 출처 없음 (N/A, N/A)"]
    swot = state.get("swot", {})
    lg_swot = swot.get("LG에너지솔루션", {"S": [], "W": [], "O": [], "T": []})
    catl_swot = swot.get("CATL", {"S": [], "W": [], "O": [], "T": []})
    lg_swot_lines = [
        f"- S: {', '.join(lg_swot.get('S', []))}",
        f"- W: {', '.join(lg_swot.get('W', []))}",
        f"- O: {', '.join(lg_swot.get('O', []))}",
        f"- T: {', '.join(lg_swot.get('T', []))}",
    ]
    catl_swot_lines = [
        f"- S: {', '.join(catl_swot.get('S', []))}",
        f"- W: {', '.join(catl_swot.get('W', []))}",
        f"- O: {', '.join(catl_swot.get('O', []))}",
        f"- T: {', '.join(catl_swot.get('T', []))}",
    ]
    # SUMMARY는 market_assessment 전체를 사용하되, 비어 있으면 기본 문구 사용
    summary_src = state.get("market_assessment", "") or ""
    summary_text = summary_src.strip() or "- 전체 보고서의 1/2 요약 내용"
    return "\n".join(
        [
            "# 배터리 시장 전략 분석 보고서",
            "## LG에너지솔루션 vs CATL 포트폴리오 다각화 전략 비교",
            "",
            "---",
            "",
            "## SUMMARY",
            summary_text,
            "",
            "---",
            "",
            "## 1. 시장 배경",
            "### 1.1 글로벌 전기차 시장 캐즘 현황",
            state.get("market_rag", ""),
            "### 1.2 배터리 산업 패러다임 변화",
            state.get("market_assessment", ""),
            "### 1.3 HEV 피벗과 완성차 전략 변화",
            state.get("strategy_comparison", ""),
            "",
            "---",
            "",
            "## 2. LG에너지솔루션 전략 분석",
            "### 2.1 현재 사업 포트폴리오",
            state.get("lg_rag", ""),
            "### 2.2 다각화 전략",
            state.get("lg_tech_summary", ""),
            "### 2.3 기술 경쟁력 및 투자 현황",
            state.get("lg_tech_summary", ""),
            "### 2.4 최신 동향",
            "\n".join(
                [
                    f"- {item['title']} ({item['pub_date']})"
                    for item in state.get("lg_news", [])[:6]
                ]
            ),
            "",
            "---",
            "",
            "## 3. CATL 전략 분석",
            "### 3.1 현재 사업 포트폴리오",
            state.get("catl_rag", ""),
            "### 3.2 다각화 전략",
            state.get("catl_tech_summary", ""),
            "### 3.3 기술 경쟁력 및 글로벌 확장",
            state.get("catl_tech_summary", ""),
            "### 3.4 최신 동향",
            "\n".join(
                [
                    f"- {item['title']} ({item['pub_date']})"
                    for item in state.get("catl_news", [])[:6]
                ]
            ),
            "",
            "---",
            "",
            "## 4. 시장성 평가",
            "### 4.1 글로벌 배터리 시장 규모 및 전망",
            state.get("market_rag", ""),
            "### 4.2 LG에너지솔루션 시장성 평가",
            state.get("lg_tech_summary", ""),
            "### 4.3 CATL 시장성 평가",
            state.get("market_assessment", ""),
            "",
            "---",
            "",
            "## 5. 전략 비교 및 SWOT 분석",
            "### 5.1 핵심 전략 비교표",
            state.get("strategy_comparison", ""),
            "### 5.2 LG에너지솔루션 SWOT",
            *lg_swot_lines,
            "### 5.3 CATL SWOT",
            *catl_swot_lines,
            "### 5.4 SWOT 교차 비교 분석",
            state.get("strategy_comparison", ""),
            "",
            "---",
            "",
            "## 6. 종합 시사점",
            "### 6.1 각 기업의 전략적 포지셔닝",
            "LG에너지솔루션은 북미 중심 생산·공급망 안정화 전략이 강점이며, CATL은 LFP 기반 원가 경쟁력과 제품 다변화가 강점이다.",
            "### 6.2 향후 시장 전망 및 리스크",
            "EV 캐즘 장기화, 정책/무역 규제, 원재료 가격 변동성이 공통 리스크이며 ESS·HEV·신규 화학계열 확장이 완충 전략으로 작동한다.",
            "### 6.3 시사점",
            "양사는 EV 단일 수요 의존도를 낮추고, 지역·제품·기술 축의 포트폴리오 다각화 속도에서 경쟁 우위가 결정될 가능성이 높다.",
            "",
            "---",
            "",
            "## REFERENCE",
            *reference_lines,
        ]
    )


def validate_swot(swot: dict[str, dict[str, list[str]]]) -> bool:
    for company in ("LG에너지솔루션", "CATL"):
        section = swot.get(company, {})
        for key in ("S", "W", "O", "T"):
            items = section.get(key, [])
            if not isinstance(items, list) or len(items) == 0:
                return False
    return True


def validate_reference_format(report: str) -> bool:
    if "REFERENCE" not in report:
        return False
    refs = report.split("REFERENCE", maxsplit=1)[1].strip().splitlines()
    if not refs:
        return False
    pattern = re.compile(r"^- \[(기관보고서|논문|웹페이지)\] .+ \(.+, .+\)$")
    return all(pattern.match(line.strip()) for line in refs if line.strip())


def validate_summary_reference(report: str) -> bool:
    return (
        report.startswith("# 배터리 시장 전략 분석 보고서")
        and "## SUMMARY" in report
        and "## REFERENCE" in report
        and report.rstrip().splitlines()[-1].startswith("- [")
    )
