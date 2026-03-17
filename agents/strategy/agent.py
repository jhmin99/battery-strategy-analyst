from __future__ import annotations

from config import Settings

from agents.core import build_swot
from agents.schemas import WorkflowState


class StrategySwotAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: WorkflowState) -> WorkflowState:
        swot, comparison = build_swot(
            settings=self.settings,
            lg_text=state.get("lg_tech_summary", ""),
            catl_text=state.get("catl_tech_summary", ""),
            market=state.get("market_assessment", ""),
        )
        return {"swot": swot, "strategy_comparison": comparison}
