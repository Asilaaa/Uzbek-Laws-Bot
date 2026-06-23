CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS law_chunks (
    id bigserial PRIMARY KEY,
    document_name text NOT NULL,
    chunk_text text NOT NULL,
    embedding vector(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS law_chunks_document_name_idx
    ON law_chunks (document_name);

CREATE INDEX IF NOT EXISTS law_chunks_embedding_hnsw_idx
    ON law_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS law_answers (
    answer_id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    chat_id bigint NOT NULL,
    user_id bigint,
    username text,
    question text NOT NULL,
    answer text NOT NULL,
    source_documents text NOT NULL
);

CREATE TABLE IF NOT EXISTS law_feedback (
    answer_id text PRIMARY KEY,
    feedback text NOT NULL CHECK (feedback IN ('up', 'down')),
    created_at timestamptz NOT NULL DEFAULT now()
);
