from __future__ import annotations

from typing import Literal

from config import Settings

from agents.core import (
    validate_reference_format,
    validate_summary_reference,
    validate_swot,
)
from agents.schemas import WorkflowState


class SupervisorAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def init_node(self, state: WorkflowState) -> WorkflowState:
        return {
            "retry_count": state.get("retry_count", 0),
            "max_retry": self.settings.max_retry,
            "rag_page_limit": self.settings.page_limit,
            "embedding_model": self.settings.embedding_model_name,
            "errors": state.get("errors", []),
        }

    def source_gate_node(self, state: WorkflowState) -> WorkflowState:
        lg_source_count = len(state.get("lg_sources", [])) + len(state.get("lg_news", []))
        catl_source_count = len(state.get("catl_sources", [])) + len(
            state.get("catl_news", [])
        )
        c1 = lg_source_count >= 3 and catl_source_count >= 3
        lg_cov = state.get("lg_query_coverage", {})
        catl_cov = state.get("catl_query_coverage", {})
        c2 = (
            lg_cov.get("positive", False)
            and lg_cov.get("negative", False)
            and catl_cov.get("positive", False)
            and catl_cov.get("negative", False)
        )
        criteria = dict(state.get("criteria", {}))
        criteria["C1"] = c1
        criteria["C2"] = c2
        updates: WorkflowState = {"criteria": criteria}
        if not c1 or not c2:
            updates["retry_count"] = state.get("retry_count", 0) + 1
        return updates

    def source_route(
        self, state: WorkflowState
    ) -> Literal["retry_dispatch", "market_assessment"]:
        c1 = state.get("criteria", {}).get("C1", False)
        c2 = state.get("criteria", {}).get("C2", False)
        if c1 and c2:
            return "market_assessment"
        if state.get("retry_count", 0) <= state.get("max_retry", 2):
            return "retry_dispatch"
        return "market_assessment"

    def retry_dispatch_node(self, _: WorkflowState) -> WorkflowState:
        return {}

    def final_gate_node(self, state: WorkflowState) -> WorkflowState:
        swot_ok = validate_swot(state.get("swot", {}))
        report = state.get("final_report", "")
        ref_ok = validate_reference_format(report)
        struct_ok = validate_summary_reference(report)
        criteria = dict(state.get("criteria", {}))
        criteria["C3"] = swot_ok
        criteria["C4"] = ref_ok
        criteria["C5"] = struct_ok
        updates: WorkflowState = {"criteria": criteria}
        if not (swot_ok and ref_ok and struct_ok):
            updates["retry_count"] = state.get("retry_count", 0) + 1
        return updates

    def final_route(self, state: WorkflowState) -> Literal["strategy_swot", "__end__"]:
        c3 = state.get("criteria", {}).get("C3", False)
        c4 = state.get("criteria", {}).get("C4", False)
        c5 = state.get("criteria", {}).get("C5", False)
        if c3 and c4 and c5:
            return "__end__"
        if state.get("retry_count", 0) <= state.get("max_retry", 2):
            return "strategy_swot"
        return "__end__"
