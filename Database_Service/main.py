import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Database Microservice")

DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "chathistory")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )

@app.get("/health")
def health():
    return {"status": "healthy"}

# --- Models ---
class MessageCreate(BaseModel):
    conv_id: str
    role: str
    message: str

class ChunkInsert(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = {}

class SimilarityQuery(BaseModel):
    query_embedding: List[float]
    top_k: int = 5

# --- History Endpoints ---
@app.post("/history/add")
def add_message(payload: MessageCreate):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversation_history (conv_id, role, message) VALUES (%s, %s, %s)",
            (payload.conv_id, payload.role, payload.message)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "message": "Message saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/all", response_model=List[Dict[str, Any]])
def get_all_history():
    """Reads all rows from the conversation_history table."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, conv_id, role, message, created_at FROM conversation_history ORDER BY created_at ASC;")
        history = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{conv_id}")
def get_history(conv_id: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT conv_id, role, message, created_at FROM conversation_history WHERE conv_id = %s ORDER BY created_at ASC",
            (conv_id,)
        )
        history = cur.fetchall()
        cur.close()
        conn.close()
        return {"conv_id": conv_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Vector & RAG Endpoints ---
@app.post("/rag/chunks/add")
def add_chunk(payload: ChunkInsert):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO document_chunks (document_id, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            (payload.document_id, payload.chunk_index, payload.content, Json(payload.metadata), payload.embedding)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "message": "Chunk added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/chunks/all", response_model=List[Dict[str, Any]])
def get_all_chunks():
    """Reads all rows safely from document_chunks matching your exact DB schema."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM document_chunks ORDER BY id ASC;")
        chunks = cur.fetchall()
        cur.close()
        conn.close()
        
        return [dict(row) for row in chunks]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/search", response_model=List[Dict[str, Any]])
def search_similar_chunks(payload: SimilarityQuery):
    """Performs similarity search and returns matching rows as a list of dicts."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT id, document_id, chunk_index, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity_score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
            """,
            (payload.query_embedding, payload.query_embedding, payload.top_k)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))