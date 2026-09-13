import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from Schemas import MessageCreate, ChunkInsert, SimilarityQuery
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Database Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "postgres")
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

@app.delete("/history/{conv_id}")
def delete_history(conv_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversation_history WHERE conv_id = %s;",
            (conv_id,)
        )
        deleted_rows = cur.rowcount
        conn.commit()
        cur.close()

        if deleted_rows == 0:
            return {
                "status": "warning",
                "message": f"No conversation found with ID {conv_id}.",
                "deleted_count": 0,
            }

        return {
            "status": "success",
            "conv_id": conv_id,
            "deleted_count": deleted_rows,
            "message": f"Successfully deleted history for conversation {conv_id}.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/history/{conv_id}")
def get_history(conv_id: str, limit: Optional[int] = None):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        if limit and limit > 0:
            query = """
                SELECT conv_id, role, message, created_at 
                FROM (
                    SELECT id, conv_id, role, message, created_at 
                    FROM conversation_history 
                    WHERE conv_id = %s 
                    ORDER BY id DESC 
                    LIMIT %s
                ) sub
                ORDER BY id ASC;
            """
            cur.execute(query, (conv_id, limit))
        else:
            query = """
                SELECT conv_id, role, message, created_at 
                FROM conversation_history 
                WHERE conv_id = %s 
                ORDER BY id ASC;
            """
            cur.execute(query, (conv_id,))
            
        raw_history = cur.fetchall()
        cur.close()

        formatted_messages = []
        for row in raw_history:
            msg = dict(row)
            if isinstance(msg.get("created_at"), (datetime, date)):
                msg["created_at"] = msg["created_at"].isoformat()
            formatted_messages.append(msg)

        return {"status": "success", "conv_id": conv_id, "messages": formatted_messages}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.post("/rag/chunks/add")
def add_chunk(payload: ChunkInsert):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        filename = getattr(payload, "filename", getattr(payload, "doc_title", "unknown"))

        cur.execute(
            """
            INSERT INTO document_chunks (document_id, filename, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            """,
            (payload.document_id, filename, payload.chunk_index, payload.content, Json(payload.metadata), payload.embedding)
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
            SELECT id, document_id, filename, chunk_index, content, metadata, "_insertionTimestamp"
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
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            vector_str = f"[{','.join(map(str, payload.query_embedding))}]"

            cur.execute(
                """
                SELECT id, document_id, filename, chunk_index, content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity_score
                FROM document_chunks
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s;
                """,
                (vector_str, vector_str, payload.min_sim, vector_str, payload.top_k)
            )
            results = cur.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"DB SEARCH ERROR: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()