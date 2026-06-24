from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from psycopg import sql

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
load_dotenv(ENV_FILE)

DEFAULT_FOLDER = Path("/home/asila/Documents/Vector store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
TABLE_NAME = os.getenv("PGVECTOR_TABLE", "law_chunks")
LAW_DOCUMENTS_TABLE = os.getenv("LAW_DOCUMENTS_TABLE", "law_documents")

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=api_key)
    return _client


def get_async_openai_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        _async_client = AsyncOpenAI(api_key=api_key)
    return _async_client


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set")
    return database_url


def get_connection() -> psycopg.Connection:
    return psycopg.connect(get_database_url())


def embed(text: str) -> list[float]:
    client = get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def ensure_schema(conn: psycopg.Connection) -> None:
    index_name = f"{TABLE_NAME}_embedding_hnsw_idx"
    doc_index_name = f"{TABLE_NAME}_document_name_idx"

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS {{table_name}} (
                    id bigserial PRIMARY KEY,
                    document_name text NOT NULL,
                    chunk_text text NOT NULL,
                    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
                )
                """
            ).format(table_name=sql.Identifier(TABLE_NAME))
        )
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (document_name)"
            ).format(
                index_name=sql.Identifier(doc_index_name),
                table_name=sql.Identifier(TABLE_NAME),
            )
        )
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING hnsw (embedding vector_cosine_ops)"
            ).format(
                index_name=sql.Identifier(index_name),
                table_name=sql.Identifier(TABLE_NAME),
            )
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id bigserial PRIMARY KEY,
                    document_name text UNIQUE NOT NULL,
                    object_key text,
                    source_url text
                )
                """
            ).format(table_name=sql.Identifier(LAW_DOCUMENTS_TABLE))
        )
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (document_name)"
            ).format(
                index_name=sql.Identifier(f"{LAW_DOCUMENTS_TABLE}_document_name_idx"),
                table_name=sql.Identifier(LAW_DOCUMENTS_TABLE),
            )
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS law_answers (
                answer_id text PRIMARY KEY,
                created_at timestamptz NOT NULL DEFAULT now(),
                chat_id bigint NOT NULL,
                user_id bigint,
                username text,
                question text NOT NULL,
                answer text NOT NULL,
                source_documents text NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS law_feedback (
                answer_id text PRIMARY KEY,
                feedback text NOT NULL CHECK (feedback IN ('up', 'down')),
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()
