from __future__ import annotations

from config import Settings

from agents.core import build_report
from agents.schemas import WorkflowState


class ReportWriterAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: WorkflowState) -> WorkflowState:
        base_report = build_report(state)
        return {"final_report": base_report}
