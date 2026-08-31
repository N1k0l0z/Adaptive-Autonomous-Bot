import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm
from dotenv import load_dotenv

# 1. Explicitly load .env from the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

PARQUET_PATH = BASE_DIR / "Document_Upload_Service" / "Chunks" / "master_chunks.parquet"

# 2. Database configuration with fallbacks for local script execution
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "postgres"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "host": os.getenv("DB_HOST") or "localhost",
    "port": os.getenv("DB_PORT") or "5432"
}

def clean_nul_bytes(val):
    if isinstance(val, str):
        return val.replace('\x00', '')
    return val

def merge_parquet_to_postgres():
    print(f"Loading Parquet data from: {PARQUET_PATH}...")
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found at {PARQUET_PATH}")

    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas(types_mapper=None)
    
    print(f"Loaded {len(df):,} rows from Parquet.")
    
    print("Sanitizing text columns...")
    for col in ["content", "doc_title", "document_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('\x00', '', regex=False)

    conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

    print(f"Connecting to PostgreSQL (`{DB_CONFIG['dbname']}` at {DB_CONFIG['host']}:{DB_CONFIG['port']})...")
    
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            print("Ensuring pgvector extension is enabled...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            
            register_vector(conn)

            print("Ensuring `document_chunks` table exists...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id TEXT,
                    doc_title TEXT,
                    chunk_index INT,
                    content TEXT,
                    embedding vector(1024)
                );
            """)
            conn.commit()

            print("Bulk stream inserting into `document_chunks` via psycopg write_row...")
            
            with cur.copy(
                "COPY document_chunks (document_id, doc_title, chunk_index, content, embedding) FROM STDIN"
            ) as copy:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Streaming Chunks"):
                    doc_id = clean_nul_bytes(str(row["document_id"]))
                    doc_title = clean_nul_bytes(str(row["doc_title"]))
                    chunk_idx = int(row["chunk_index"])
                    content = clean_nul_bytes(str(row["content"]))
                    
                    emb_val = row["embedding"].tolist() if hasattr(row["embedding"], "tolist") else row["embedding"]
                    embedding_str = f"[{','.join(map(str, emb_val))}]"

                    copy.write_row((doc_id, doc_title, chunk_idx, content, embedding_str))
            
            conn.commit()

    print(f"Successfully merged {len(df):,} rows into `{DB_CONFIG['dbname']}.document_chunks`!")

if __name__ == "__main__":
    merge_parquet_to_postgres()