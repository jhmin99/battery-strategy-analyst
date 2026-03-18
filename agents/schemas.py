from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    query: str
    retry_count: int
    max_retry: int
    rag_page_limit: int
    embedding_model: str
    lg_embedding_backend: str
    catl_embedding_backend: str
    lg_rag: str
    catl_rag: str
    lg_sources: list[dict[str, str]]
    catl_sources: list[dict[str, str]]
    lg_news: list[dict[str, str]]
    catl_news: list[dict[str, str]]
    lg_query_coverage: dict[str, bool]
    catl_query_coverage: dict[str, bool]
    lg_tech_summary: str
    catl_tech_summary: str

    # --- 새로 추가해야 할 State 변수들 ---
    lg_portfolio: str
    lg_diversification: str
    lg_investment: str
    catl_portfolio: str
    catl_diversification: str
    catl_investment: str
    market_chasm: str
    market_paradigm: str
    market_hev_pivot: str  # 1.3 목차용
    market_outlook: str
    # ------------------------------------

    market_rag: str
    market_sources: list[dict[str, str]]
    market_embedding_backend: str
    market_assessment: str
    swot: dict[str, dict[str, list[str]]]
    strategy_comparison: str
    final_report: str
    criteria: dict[str, bool]
    errors: list[str]


@dataclass
class CompanyPlan:
    name: str
    pdf_dir: Path
    rag_queries: list[str]
    positive_queries: list[str]
    negative_queries: list[str]
