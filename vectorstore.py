from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from config import Settings


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    score: float


class HybridVectorStore:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self._backend = "tfidf"
        self._tfidf = TfidfVectorizer(max_features=20000)
        self._tfidf_matrix: Any = None
        self._st_model: Any = None
        self._dense_embeddings: np.ndarray | None = None
        self._try_init_sentence_transformer()

    def _try_init_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name)
            self._backend = "bge-m3"
        except Exception:
            self._st_model = None
            self._backend = "tfidf"

    @property
    def backend(self) -> str:
        return self._backend

    def add_documents(self, docs: list[str], metadatas: list[dict[str, Any]]) -> None:
        self.documents.extend(docs)
        self.metadatas.extend(metadatas)

    def build(self) -> None:
        if not self.documents:
            self._tfidf_matrix = None
            self._dense_embeddings = None
            return
        if self._backend == "bge-m3" and self._st_model is not None:
            matrix = self._st_model.encode(self.documents, normalize_embeddings=True)
            self._dense_embeddings = np.array(matrix, dtype=np.float32)
            self._tfidf_matrix = None
            return
        self._tfidf_matrix = self._tfidf.fit_transform(self.documents)
        self._dense_embeddings = None

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self.documents:
            return []
        scores: np.ndarray
        if (
            self._backend == "bge-m3"
            and self._dense_embeddings is not None
            and self._st_model is not None
        ):
            query_vec = self._st_model.encode([query], normalize_embeddings=True)[0]
            scores = self._dense_embeddings @ np.asarray(query_vec, dtype=np.float32)
        else:
            if self._tfidf_matrix is None:
                return []
            query_vec = self._tfidf.transform([query])
            scores = (self._tfidf_matrix @ query_vec.T).toarray().ravel()
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(
                content=self.documents[idx],
                metadata=self.metadatas[idx],
                score=float(scores[idx]),
            )
            for idx in top_indices
            if float(scores[idx]) > 0
        ]


CollectionName = Literal["lg", "catl", "market_trend"]


class PgVectorStore:
    def __init__(self, settings: Settings, model_name: str) -> None:
        self.settings = settings
        self.model_name = model_name
        self._backend = "hashing"
        self._hashing = HashingVectorizer(
            n_features=self.settings.pgvector_dim,
            alternate_sign=False,
            norm="l2",
        )
        self._st_model: Any = None
        self._enabled = False
        self._psycopg: Any = None
        self._try_init_backends()

    def _try_init_backends(self) -> None:
        try:
            import psycopg

            self._psycopg = psycopg
            self._enabled = self.settings.pgvector_enabled
        except Exception:
            self._enabled = False
            self._psycopg = None
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name)
            self._backend = "bge-m3"
        except Exception:
            self._st_model = None
            self._backend = "hashing"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str:
        return f"pgvector-{self._backend}" if self._enabled else "pgvector-disabled"

    def _dsn(self) -> str:
        return (
            f"host={self.settings.postgres_host} "
            f"port={self.settings.postgres_port} "
            f"user={self.settings.postgres_user} "
            f"password={self.settings.postgres_password} "
            f"dbname={self.settings.postgres_db}"
        )

    def _table_name(self, collection: CollectionName) -> str:
        return f"{self.settings.vectorstore_schema}.{collection}"

    def _vector_literal(self, vector: np.ndarray) -> str:
        return "[" + ",".join(f"{float(x):.8f}" for x in vector.tolist()) + "]"

    def _doc_hash(self, content: str, metadata: dict[str, Any]) -> str:
        key = content + "|" + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _embed_one(self, text: str) -> np.ndarray:
        if self._st_model is not None:
            vec = np.asarray(
                self._st_model.encode([text], normalize_embeddings=True)[0],
                dtype=np.float32,
            )
            if vec.shape[0] == self.settings.pgvector_dim:
                return vec
            if vec.shape[0] > self.settings.pgvector_dim:
                return vec[: self.settings.pgvector_dim]
            padded = np.zeros(self.settings.pgvector_dim, dtype=np.float32)
            padded[: vec.shape[0]] = vec
            return padded
        dense = self._hashing.transform([text]).toarray()[0]
        return np.asarray(dense, dtype=np.float32)

    def ensure_tables(self) -> bool:
        if not self._enabled or self._psycopg is None:
            return False
        try:
            with self._psycopg.connect(self._dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"CREATE SCHEMA IF NOT EXISTS {self.settings.vectorstore_schema}"
                    )
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    for collection in ("lg", "catl", "market_trend"):
                        table = self._table_name(collection)  # type: ignore[arg-type]
                        cur.execute(
                            f"""
                            CREATE TABLE IF NOT EXISTS {table} (
                                id BIGSERIAL PRIMARY KEY,
                                doc_hash TEXT UNIQUE NOT NULL,
                                content TEXT NOT NULL,
                                metadata JSONB NOT NULL,
                                embedding VECTOR({self.settings.pgvector_dim}) NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                            )
                            """
                        )
                        cur.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS {collection}_embedding_idx
                            ON {table}
                            USING ivfflat (embedding vector_cosine_ops)
                            """
                        )
                conn.commit()
            return True
        except Exception:
            return False

    def add_documents(
        self,
        collection: CollectionName,
        docs: list[str],
        metadatas: list[dict[str, Any]],
    ) -> bool:
        if not self._enabled or self._psycopg is None:
            return False
        if not docs:
            return True
        table = self._table_name(collection)
        rows: list[tuple[str, str, str, str]] = []
        for doc, metadata in zip(docs, metadatas, strict=False):
            vec = self._embed_one(doc)
            rows.append(
                (
                    self._doc_hash(doc, metadata),
                    doc,
                    json.dumps(metadata, ensure_ascii=False),
                    self._vector_literal(vec),
                )
            )
        try:
            with self._psycopg.connect(self._dsn()) as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        INSERT INTO {table}(doc_hash, content, metadata, embedding)
                        VALUES (%s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (doc_hash) DO NOTHING
                        """,
                        rows,
                    )
                conn.commit()
            return True
        except Exception:
            return False

    def similarity_search(
        self,
        collection: CollectionName,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not self._enabled or self._psycopg is None:
            return []
        table = self._table_name(collection)
        query_vec = self._vector_literal(self._embed_one(query))
        try:
            with self._psycopg.connect(self._dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score
                        FROM {table}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_vec, query_vec, top_k),
                    )
                    rows = cur.fetchall()
            return [
                RetrievedChunk(
                    content=str(row[0]),
                    metadata=dict(row[1]) if isinstance(row[1], dict) else {},
                    score=float(row[2]),
                )
                for row in rows
            ]
        except Exception:
            return []


def collection_from_plan_name(name: str) -> CollectionName:
    lowered = name.lower()
    if "market" in lowered:
        return "market_trend"
    if "catl" in lowered:
        return "catl"
    return "lg"


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def load_pdf_chunks(
    pdf_paths: list[Path],
    page_limit: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    docs: list[str] = []
    metadatas: list[dict[str, Any]] = []
    total_pages = 0
    for path in pdf_paths:
        if not path.exists():
            continue
        reader = PdfReader(str(path))
        for page_index, page in enumerate(reader.pages, start=1):
            if total_pages >= page_limit:
                return docs, metadatas, total_pages
            text = page.extract_text() or ""
            chunks = split_text(
                text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            for chunk_index, chunk in enumerate(chunks, start=1):
                docs.append(chunk)
                metadatas.append(
                    {
                        "source": str(path),
                        "page": page_index,
                        "chunk": chunk_index,
                    }
                )
            total_pages += 1
    return docs, metadatas, total_pages
