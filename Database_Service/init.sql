-- Enable vector search extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Conversation History Table
CREATE TABLE IF NOT EXISTS conversation_history (
    id SERIAL PRIMARY KEY,
    conv_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conv_id ON conversation_history(conv_id);

-- 2. Document Chunks (RAG Vector Store)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    doc_title VARCHAR(255) NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Safely add column if the table existed previously without it
ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS embedding vector(1024);

-- Fast HNSW cosine similarity vector index
CREATE INDEX IF NOT EXISTS idx_document_embeddings 
ON document_chunks USING hnsw (embedding vector_cosine_ops);