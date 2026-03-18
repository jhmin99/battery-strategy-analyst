## AI Battery Strategy Analyst Agent

본 프로젝트는 **LG에너지솔루션·CATL의 포트폴리오 다각화 전략을 자동으로 비교·분석**하는 멀티 에이전트 기반 배터리 전략 분석 시스템입니다.

---

## Overview

- **Objective**:  
  - LG에너지솔루션과 CATL의  
    - 사업 포트폴리오  
    - 기술 경쟁력  
    - 다각화 전략  
    - 글로벌 시장성 및 리스크  
    를 비교·분석하여 **전략적 인사이트와 시사점이 담긴 보고서를 자동 생성**합니다.

- **Method**:  
  - LangGraph 기반 **멀티 에이전트 워크플로우 + Agentic RAG**
  - 기업별 PDF(RIR/사업보고서 등) + 최신 뉴스(Web Search) + 시장 리포트(RAG)를 결합한 분석

- **Output**:  
  - 미리 정의된 목차를 따르는 **Markdown 전략 분석 보고서**  
  - 선택 시 `make_pdf_report.py`를 통한 **PDF 보고서**로 변환

---

## Features

- **PDF RAG 기반 기업 분석**
  - `data/lg`, `data/catl`, `data/market` 디렉터리의 PDF(사업보고서, ESG 리포트, 산업 리포트 등)를 읽어  
    - 텍스트 청크 → 벡터 DB(pgvector or TF-IDF) → **유사도 기반 핵심 근거 추출**

- **Tavily + RSS Web Search**
  - LG·CATL 각각에 대해 **긍정/부정 쿼리 세트**로 최신 뉴스 수집  
  - EV 수요 둔화, ESS 성장, 지정학 리스크 등 **실시간 이슈 반영**

- **섹션별 LLM 요약**
  - 시장 배경(캐즘/패러다임/전망),  
    기업별 포트폴리오/다각화 전략/투자·기술 경쟁력,  
    전략 비교 및 SWOT, 종합 시사점  
  를 **섹션 목적에 맞는 전용 프롬프트**로 생성합니다.

- **품질 기준 기반 Supervisor**
  - C1~C5 기준(기업당 출처 ≥3, 긍·부정 쿼리 커버, SWOT 4요소 채움, Reference 형식, SUMMARY~REFERENCE 구조)을 자동 검증하고,  
  - 부족 시 **재시도 루프**로 Web Search·SWOT 재생성을 수행합니다.

- **레포트 자동 저장**
  - 기본 실행 시 `reports/battery_strategy_report.md`로 자동 저장  
  - 출처(기관보고서·웹페이지)를 포맷을 맞춰 `REFERENCE` 섹션에 정리합니다.

---

## Tech Stack 

| Category      | Details                                      |
|---------------|----------------------------------------------|
| Framework     | LangGraph, LangChain, Python                 |
| LLM           | OpenAI GPT-4o-mini (via `langchain-openai`) |
| Retrieval     | pgvector(PostgreSQL) + TF-IDF Hybrid         |
| Embedding     | `BAAI/bge-m3` (Sentence Transformers)        |
| Vector Layer  | `pgvector` + HashingVectorizer fallback      |
| Web Search    | Tavily API (+ Google News RSS fallback)      |

---

## Agents

- **Supervisor Agent**  
  전체 그래프 흐름 제어, 품질 기준(C1~C5) 체크, 재시도 여부 결정.

- **LG 탐색 팀 에이전트 (`LGTeamAgents`)**  
  LG PDF RAG 수행, LG 관련 Web Search, LG 기술 경쟁력·포트폴리오·다각화·투자 요약 생성.

- **CATL 탐색 팀 에이전트 (`CATLTeamAgents`)**  
  CATL PDF RAG 수행, CATL 관련 Web Search, CATL 기술 경쟁력·포트폴리오·다각화·투자 요약 생성.

- **시장성 평가 에이전트 (`MarketAssessmentAgent`)**  
  시장 리포트 PDF(`market_trend`) RAG + 1.1/1.2/4.1에 대응하는 시장 캐즘/패러다임/전망 텍스트 생성.

- **전략 비교 및 SWOT 분석 에이전트 (`StrategySwotAgent`)**  
  LG/CATL 요약 + 시장 평가 기반 양사의 SWOT 및 전략 비교/교차 분석 생성.

- **보고서 생성 에이전트 (`ReportWriterAgent`)**  
  `WorkflowState`를 바탕으로 고정된 보고서 목차에 맞춰 Markdown 레포트를 조립하고 SUMMARY/REFERENCE까지 출력.

---

## Architecture (Graph 흐름)   

<img width="1063" height="703" alt="스크린샷 2026-03-17 오후 3 44 11" src="https://github.com/user-attachments/assets/e88adf8e-986a-4c12-96df-80b8937d2e8d" />



- LG 라인: `Start → LG 탐색 Agent(RAG) → Web Search Agent → 기술 요약 Agent`
- CATL 라인: `Start → CATL 탐색 Agent(RAG) → Web Search Agent → 기술 요약 Agent`
- 공통 라인: `시장성 평가 Agent → 전략 비교 및 SWOT Agent → 보고서 생성 Agent → Supervisor 최종 검증`

---

## Directory Structure

```text
├── data/
│   ├── lg/                 # LG에너지솔루션 관련 PDF
│   ├── catl/               # CATL 관련 PDF
│   └── market/             # 배터리 시장/산업 리포트 PDF
├── agents/
│   ├── lg/                 # LG 탐색 에이전트
│   ├── catl/               # CATL 탐색 에이전트
│   ├── market/             # 시장 평가 에이전트
│   ├── strategy/           # 전략 비교·SWOT 에이전트
│   ├── report/             # 보고서 생성 에이전트
│   ├── supervisor/         # Supervisor 에이전트
│   ├── core.py             # RAG, Web Search, LLM 유틸
│   └── schemas.py          # Workflow 상태/플랜 스키마
├── reports/                # 생성된 Markdown/PDF 보고서
├── vectorstore.py          # pgvector/Hybrid vector store 구현
├── workflow.py             # LangGraph 그래프 정의(BatteryStrategyService)
├── main.py                 # CLI 실행 스크립트 (보고서 생성)
├── make_pdf_report.py      # Markdown → PDF 변환 스크립트
├── config.py               # Settings 및 .env 로딩
├── requirements.txt        # Python 의존성
└── README.md
```

---

## How to Run

```bash
# 1) 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) .env 설정 (예시)
# OPENAI_API_KEY=...
# TAVILY_API_KEY=...
# (선택) PGVECTOR_ENABLED, POSTGRES_* 등

# 4) 보고서 생성 (기본 쿼리)
python main.py

# 5) 지정 쿼리 + 저장 경로
python main.py --query "LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교해줘" \
               --output reports/battery_strategy_report.md
```

---

## Contributors

- **민지홍**:  
- **백강민**: 
- **임진영**: 
