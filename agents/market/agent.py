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
        prompt = (
            "글로벌 배터리 시장성과를 평가하세요.\n"
            "포함 항목: EV 캐즘, HEV 피벗, ESS/로봇 배터리 성장, 양사 시장성.\n"
            f"LG 요약:\n{state.get('lg_tech_summary', '')}\n\n"
            f"CATL 요약:\n{state.get('catl_tech_summary', '')}\n\n"
            f"시장 RAG 근거:\n{market_rag}\n"
        )
        text = invoke_llm(settings=self.settings, prompt=prompt)
        return {
            "market_rag": market_rag,
            "market_sources": market_sources,
            "market_embedding_backend": market_backend,
            "market_assessment": text,
        }
