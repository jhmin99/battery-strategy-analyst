from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Settings:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    lg_pdf_dir: Path = field(init=False)
    catl_pdf_dir: Path = field(init=False)
    market_pdf_dir: Path = field(init=False)
    page_limit: int = 100
    max_retry: int = 2
    embedding_model_name: str = "BAAI/bge-m3"
    llm_model: str = "gpt-4o-mini"
    news_max_results_per_query: int = 5
    rag_top_k: int = 6
    chunk_size: int = 900
    chunk_overlap: int = 120
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    postgres_host: str = field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost")
    )
    postgres_port: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432"))
    )
    postgres_user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "postgres")
    )
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    postgres_db: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "postgres")
    )
    vectorstore_schema: str = field(
        default_factory=lambda: os.getenv("VECTORSTORE_SCHEMA", "public")
    )
    pgvector_enabled: bool = field(
        default_factory=lambda: os.getenv("PGVECTOR_ENABLED", "true").lower() == "true"
    )
    pgvector_dim: int = 1024

    def __post_init__(self) -> None:
        self.data_dir = self.base_dir / "data"
        self.reports_dir = self.base_dir / "reports"
        self.lg_pdf_dir = self.data_dir / "lg"
        self.catl_pdf_dir = self.data_dir / "catl"
        self.market_pdf_dir = self.data_dir / "market"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.lg_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.catl_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.market_pdf_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    env_path = Path(__file__).resolve().parent / ".env"
    load_env_file(env_path)
    return Settings()
