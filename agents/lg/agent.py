from __future__ import annotations

from config import Settings

from agents.core import (
    collect_news,
    run_rag,
    summarize_diversification,
    summarize_investment_and_capability,
    summarize_portfolio,
    summarize_tech,
)
from agents.schemas import CompanyPlan, WorkflowState


class LGTeamAgents:
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
            "lg_rag": summary,
            "lg_sources": sources,
            "lg_embedding_backend": backend,
        }

    def web_node(self, _: WorkflowState) -> WorkflowState:
        news, coverage = collect_news(
            plan=self.plan,
            max_results_per_query=self.settings.news_max_results_per_query,
        )
        return {"lg_news": news, "lg_query_coverage": coverage}

    def tech_node(self, state: WorkflowState) -> WorkflowState:
        tech_summary = summarize_tech(
            settings=self.settings,
            company_name=self.plan.name,
            rag_summary=state.get("lg_rag", ""),
            news_items=state.get("lg_news", []),
        )
        sources = state.get("lg_sources", [])
        news = state.get("lg_news", [])
        return {
            "lg_tech_summary": tech_summary,
            "lg_portfolio": summarize_portfolio(
                settings=self.settings, company_name=self.plan.name, sources=sources
            ),
            "lg_diversification": summarize_diversification(
                settings=self.settings,
                company_name=self.plan.name,
                sources=sources,
                news_items=news,
            ),
            "lg_investment": summarize_investment_and_capability(
                settings=self.settings,
                company_name=self.plan.name,
                sources=sources,
                news_items=news,
            ),
        }
