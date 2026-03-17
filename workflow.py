from __future__ import annotations

from datetime import datetime
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.catl import CATLTeamAgents
from agents.lg import LGTeamAgents
from agents.market import MarketAssessmentAgent
from agents.report import ReportWriterAgent
from agents.schemas import CompanyPlan, WorkflowState
from agents.strategy import StrategySwotAgent
from agents.supervisor import SupervisorAgent
from config import Settings, get_settings


class BatteryStrategyService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.supervisor = SupervisorAgent(self.settings)
        self.lg_agents = LGTeamAgents(self.settings, self._build_lg_plan())
        self.catl_agents = CATLTeamAgents(self.settings, self._build_catl_plan())
        self.market_agent = MarketAssessmentAgent(self.settings)
        self.strategy_agent = StrategySwotAgent(self.settings)
        self.report_agent = ReportWriterAgent(self.settings)
        self.graph = self._build_graph()

    def _build_lg_plan(self) -> CompanyPlan:
        return CompanyPlan(
            name="LG에너지솔루션",
            pdf_dir=self.settings.lg_pdf_dir,
            rag_queries=[
                "사업 포트폴리오 다각화",
                "북미 투자와 생산전략",
                "ESS와 로봇 배터리 전략",
            ],
            positive_queries=[
                "LG에너지솔루션 수주 확대 ESS 성장",
                "LG에너지솔루션 북미 JV 투자 성과",
            ],
            negative_queries=[
                "LG에너지솔루션 실적 둔화 리스크",
                "LG에너지솔루션 EV 수요 감소",
            ],
        )

    def _build_catl_plan(self) -> CompanyPlan:
        return CompanyPlan(
            name="CATL",
            pdf_dir=self.settings.catl_pdf_dir,
            rag_queries=[
                "LFP 전략과 원가경쟁력",
                "나트륨이온 배터리 상업화",
                "글로벌 ESS 확장 전략",
            ],
            positive_queries=[
                "CATL ESS 수주 확대",
                "CATL 나트륨이온 양산",
            ],
            negative_queries=[
                "CATL 지정학 리스크",
                "CATL 가격 경쟁 압박",
            ],
        )

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("supervisor_init", self.supervisor.init_node)
        graph.add_node("lg_rag", self.lg_agents.rag_node)
        graph.add_node("catl_rag", self.catl_agents.rag_node)
        graph.add_node("lg_web", self.lg_agents.web_node)
        graph.add_node("catl_web", self.catl_agents.web_node)
        graph.add_node("lg_tech", self.lg_agents.tech_node)
        graph.add_node("catl_tech", self.catl_agents.tech_node)
        graph.add_node("source_gate", self.supervisor.source_gate_node)
        graph.add_node("retry_dispatch", self.supervisor.retry_dispatch_node)
        graph.add_node("market_assessment", self.market_agent.run)
        graph.add_node("strategy_swot", self.strategy_agent.run)
        graph.add_node("report_writer", self.report_agent.run)
        graph.add_node("final_gate", self.supervisor.final_gate_node)

        graph.add_edge(START, "supervisor_init")
        graph.add_edge("supervisor_init", "lg_rag")
        graph.add_edge("supervisor_init", "catl_rag")
        graph.add_edge("lg_rag", "lg_web")
        graph.add_edge("catl_rag", "catl_web")
        graph.add_edge("lg_web", "lg_tech")
        graph.add_edge("catl_web", "catl_tech")
        graph.add_edge("lg_tech", "source_gate")
        graph.add_edge("catl_tech", "source_gate")
        graph.add_conditional_edges(
            "source_gate",
            self.supervisor.source_route,
            {
                "retry_dispatch": "retry_dispatch",
                "market_assessment": "market_assessment",
            },
        )
        graph.add_edge("retry_dispatch", "lg_web")
        graph.add_edge("retry_dispatch", "catl_web")
        graph.add_edge("market_assessment", "strategy_swot")
        graph.add_edge("strategy_swot", "report_writer")
        graph.add_edge("report_writer", "final_gate")
        graph.add_conditional_edges(
            "final_gate",
            self.supervisor.final_route,
            {
                "strategy_swot": "strategy_swot",
                "__end__": END,
            },
        )
        return graph.compile(checkpointer=MemorySaver())

    def run(self, query: str) -> WorkflowState:
        initial_state: WorkflowState = {
            "query": query,
            "retry_count": 0,
            "errors": [],
        }
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": f"battery-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        }
        result = self.graph.invoke(initial_state, config=config)
        return result

    def stream(self, query: str):
        initial_state: WorkflowState = {
            "query": query,
            "retry_count": 0,
            "errors": [],
        }
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": f"battery-stream-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        }
        for event in self.graph.stream(
            initial_state, config=config, stream_mode="updates"
        ):
            yield event
