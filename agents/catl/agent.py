from __future__ import annotations

from config import Settings

from agents.core import collect_news, run_rag, summarize_tech
from agents.schemas import CompanyPlan, WorkflowState


class CATLTeamAgents:
    def __init__(self, settings: Settings, plan: CompanyPlan) -> None:
        self.settings = settings
        self.plan = plan

    def rag_node(self, state: WorkflowState) -> WorkflowState:
        per_company = max(1, state["rag_page_limit"] // 2)
        summary, sources, backend = run_rag(
            plan=self.plan,
            settings=self.settings,
            per_company_page_limit=per_company,
        )
        return {
            "catl_rag": summary,
            "catl_sources": sources,
            "catl_embedding_backend": backend,
        }

    def web_node(self, _: WorkflowState) -> WorkflowState:
        news, coverage = collect_news(
            plan=self.plan,
            max_results_per_query=self.settings.news_max_results_per_query,
        )
        return {"catl_news": news, "catl_query_coverage": coverage}

    def tech_node(self, state: WorkflowState) -> WorkflowState:
        summary = summarize_tech(
            settings=self.settings,
            company_name=self.plan.name,
            rag_summary=state.get("catl_rag", ""),
            news_items=state.get("catl_news", []),
        )
        return {"catl_tech_summary": summary}
