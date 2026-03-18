from __future__ import annotations

from config import Settings

from agents.core import invoke_llm, run_rag
from agents.schemas import CompanyPlan, WorkflowState


class MarketAssessmentAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.plan = CompanyPlan(
            name="market_trend",
            pdf_dir=self.settings.market_pdf_dir,
            rag_queries=[
                "글로벌 배터리 시장 규모 전망",
                "EV 캐즘과 HEV 피벗 영향",
                "ESS 및 로봇 배터리 수요 성장",
            ],
            positive_queries=[],
            negative_queries=[],
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        market_rag, market_sources, market_backend = run_rag(
            plan=self.plan,
            settings=self.settings,
            per_company_page_limit=max(1, state.get("rag_page_limit", 100) // 2),
        )
        chasm_prompt = (
            "보고서 섹션 '1.1 글로벌 전기차 시장 캐즘 현황'을 작성하세요.\n"
            "요구사항: 성장률 둔화/정의/원가 압력/중국 vs 비중국 관점 포함. 8~12줄.\n"
            f"시장 RAG 근거:\n{market_rag}\n"
        )
        paradigm_prompt = (
            "보고서 섹션 '1.2 배터리 산업 패러다임 변화'를 작성하세요.\n"
            "요구사항: EV→HEV/ESS 등 수요 축 이동, 원가/규제/공급망, 기술 트렌드(예: LFP, 나트륨이온) 언급. 8~12줄.\n"
            f"시장 RAG 근거:\n{market_rag}\n"
        )
        outlook_prompt = (
            "보고서 섹션 '4.1 글로벌 배터리 시장 규모 및 전망'을 작성하세요.\n"
            "요구사항: 단기(1~2년)/중기(3~5년) 전망을 구분하고, EV/HEV/ESS 축으로 성장 동인을 정리. 8~12줄.\n"
            f"시장 RAG 근거:\n{market_rag}\n"
        )
        market_prompt = (
            "글로벌 배터리 시장성(요약)을 작성하세요.\n"
            "포함 항목: EV 캐즘, HEV 피벗, ESS/로봇 배터리 성장, 양사 시장성 시사점.\n"
            f"LG 요약:\n{state.get('lg_tech_summary', '')}\n\n"
            f"CATL 요약:\n{state.get('catl_tech_summary', '')}\n\n"
            f"시장 RAG 근거:\n{market_rag}\n"
        )
        chasm_text = invoke_llm(settings=self.settings, prompt=chasm_prompt)
        paradigm_text = invoke_llm(settings=self.settings, prompt=paradigm_prompt)
        outlook_text = invoke_llm(settings=self.settings, prompt=outlook_prompt)
        market_text = invoke_llm(settings=self.settings, prompt=market_prompt)
        return {
            "market_rag": market_rag,
            "market_sources": market_sources,
            "market_embedding_backend": market_backend,
            "market_assessment": market_text,
            "market_chasm": chasm_text,
            "market_paradigm": paradigm_text,
            "market_outlook": outlook_text,
        }
