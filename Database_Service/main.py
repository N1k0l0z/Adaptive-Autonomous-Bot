import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
from Schemas import MessageCreate, ChunkInsert, SimilarityQuery


app = FastAPI(title="Database Microservice")

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")

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

from typing import Optional

@app.get("/history/{conv_id}")
def get_history(conv_id: str, limit: Optional[int] = None):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        if limit and limit > 0:
            query = """
                SELECT conv_id, role, message, created_at 
                FROM (
                    SELECT conv_id, role, message, created_at 
                    FROM conversation_history 
                    WHERE conv_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                ) sub
                ORDER BY created_at ASC;
            """
            cur.execute(query, (conv_id, limit))
        else:
            query = """
                SELECT conv_id, role, message, created_at 
                FROM conversation_history 
                WHERE conv_id = %s 
                ORDER BY created_at ASC;
            """
            cur.execute(query, (conv_id,))
            
        history = cur.fetchall()
        cur.close()
        conn.close()
        return {"conv_id": conv_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
def get_all_chunks(limit: int = 100):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT *
            FROM document_chunks
            ORDER BY id ASC
            LIMIT %s;
        """
        cur.execute(query, (limit,))
        chunks = cur.fetchall()
        cur.close()
        return [dict(row) for row in chunks]

    except Exception as e:
        print(f"Error in /rag/chunks/all: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        if conn:
            conn.close()

@app.post("/rag/search", response_model=List[Dict[str, Any]])
def search_similar_chunks(payload: SimilarityQuery):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        vector_str = f"[{','.join(map(str, payload.query_embedding))}]"

        cur.execute(
            """
            SELECT id, document_id, chunk_index, content,
                   1 - (embedding <=> %s::vector) AS similarity_score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
            """,
            (vector_str, vector_str, payload.top_k)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in results]
    except Exception as e:
        print(f"DB SEARCH ERROR: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))