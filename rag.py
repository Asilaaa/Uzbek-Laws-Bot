from __future__ import annotations

import os
from dataclasses import dataclass

from psycopg import sql

from common import (
    CHAT_MODEL,
    LAW_DOCUMENTS_TABLE,
    TABLE_NAME,
    embed,
    ensure_schema,
    get_async_openai_client,
    get_connection,
    get_openai_client,
)

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
SYSTEM_PROMPT = os.getenv(
    "LAW_BOT_SYSTEM_PROMPT",
    (
        "You are a legal document assistant for a vector database of laws and regulations. "
        "Answer the user's question using the provided context first. "
        "If the context is insufficient, say that clearly. "
        "Do not invent statutes, articles, or requirements. "
        "Be concise, structured, and practical. "
        "This is informational assistance, not a substitute for professional legal advice."
        "Use the same language as the user."
    ),
)


@dataclass(slots=True)
class SearchResult:
    document_name: str
    chunk_text: str
    similarity: float
    page_start: int | None = None
    page_end: int | None = None

    @property
    def page_label(self) -> str | None:
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"


@dataclass(slots=True)
class SourceLink:
    document_name: str
    source_url: str | None
    page_start: int | None = None
    page_end: int | None = None

    @property
    def page_label(self) -> str | None:
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"

    @property
    def resolved_url(self) -> str | None:
        if not self.source_url:
            return None
        if self.page_start is None:
            return self.source_url
        return f"{self.source_url}#page={self.page_start}"

    @property
    def display_name(self) -> str:
        if not self.page_label:
            return self.document_name
        return f"{self.document_name} ({self.page_label})"


def search_chunks(query: str, top_k: int = RAG_TOP_K, document_name: str | None = None) -> list[SearchResult]:
    vector = embed(query)

    with get_connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            if document_name:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT document_name, chunk_text, 1 - (embedding <=> %s::vector) AS similarity, page_start, page_end
                        FROM {table_name}
                        WHERE document_name ILIKE %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """
                    ).format(table_name=sql.Identifier(TABLE_NAME)),
                    (vector, f"%{document_name}%", vector, top_k),
                )
            else:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT document_name, chunk_text, 1 - (embedding <=> %s::vector) AS similarity, page_start, page_end
                        FROM {table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """
                    ).format(table_name=sql.Identifier(TABLE_NAME)),
                    (vector, vector, top_k),
                )
            rows = cur.fetchall()

    return [SearchResult(*row) for row in rows]


def build_context(chunks: list[SearchResult], max_context_chars: int = MAX_CONTEXT_CHARS) -> str:
    sections: list[str] = []
    current_size = 0

    for index, chunk in enumerate(chunks, start=1):
        location_line = f"Location: {chunk.page_label}\n" if chunk.page_label else ""
        section = (
            f"Source {index}\n"
            f"Document: {chunk.document_name}\n"
            f"{location_line}"
            f"Similarity: {chunk.similarity:.4f}\n"
            f"Text:\n{chunk.chunk_text}\n"
        )
        if sections and current_size + len(section) > max_context_chars:
            break
        sections.append(section)
        current_size += len(section)

    return "\n---\n".join(sections)


def build_messages(question: str, chunks: list[SearchResult]) -> list[dict[str, str]]:
    context = build_context(chunks)
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved legal context:\n{context if context else '[No relevant context found in the vector database.]'}\n\n"
        "Instructions:\n"
        "1. Answer in the same language as the user if possible.\n"
        "2. Prefer the retrieved context over general knowledge.\n"
        "3. If the context is incomplete, say what is missing.\n"
        "4. Use the provided document names and page locations internally to stay grounded.\n"
        "5. Do not add a separate sources section at the end; it will be added by the application."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(messages: list[dict[str, str]]) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


async def stream_completion(messages: list[dict[str, str]]):
    client = get_async_openai_client()
    stream = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content or ""
        if delta:
            yield delta


def get_source_links(chunks: list[SearchResult]) -> list[SourceLink]:
    seen_citations: set[tuple[str, int | None, int | None]] = set()
    citation_order: list[tuple[str, int | None, int | None]] = []

    for chunk in chunks:
        citation_key = (chunk.document_name, chunk.page_start, chunk.page_end)
        if citation_key in seen_citations:
            continue
        seen_citations.add(citation_key)
        citation_order.append(citation_key)

    document_names = list({document_name for document_name, _, _ in citation_order})
    if not document_names:
        return []

    with get_connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT document_name, source_url
                    FROM {table_name}
                    WHERE document_name = ANY(%s)
                    """
                ).format(table_name=sql.Identifier(LAW_DOCUMENTS_TABLE)),
                (document_names,),
            )
            rows = cur.fetchall()

    url_by_name = {document_name: source_url for document_name, source_url in rows}
    return [
        SourceLink(
            document_name=document_name,
            source_url=url_by_name.get(document_name),
            page_start=page_start,
            page_end=page_end,
        )
        for document_name, page_start, page_end in citation_order
    ]


def summarize_sources(chunks: list[SearchResult]) -> str:
    links = get_source_links(chunks)
    if not links:
        return "No sources retrieved"
    return ", ".join(link.display_name for link in links)


def save_answer_record(
    answer_id: str,
    chat_id: int,
    user_id: int | None,
    username: str | None,
    question: str,
    answer: str,
    source_documents: str,
) -> None:
    with get_connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO law_answers (
                    answer_id,
                    chat_id,
                    user_id,
                    username,
                    question,
                    answer,
                    source_documents
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (answer_id) DO UPDATE
                SET answer = EXCLUDED.answer,
                    source_documents = EXCLUDED.source_documents
                """,
                (answer_id, chat_id, user_id, username, question, answer, source_documents),
            )
        conn.commit()


def save_feedback(answer_id: str, feedback: str) -> None:
    if feedback not in {"up", "down"}:
        raise ValueError("feedback must be 'up' or 'down'")

    with get_connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO law_feedback (answer_id, feedback)
                VALUES (%s, %s)
                ON CONFLICT (answer_id) DO UPDATE
                SET feedback = EXCLUDED.feedback,
                    created_at = now()
                """,
                (answer_id, feedback),
            )
        conn.commit()
