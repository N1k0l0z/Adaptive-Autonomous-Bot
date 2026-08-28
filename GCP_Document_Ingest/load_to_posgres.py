from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm
from dotenv import load_dotenv

PARQUET_PATH = Path("/Users/nikoloz/Desktop/Adaptive-Autonomous-Bot/Document_Upload_Service/Chunks/master_chunks.parquet")

import os
from dotenv import load_dotenv


load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def clean_nul_bytes(val):
    if isinstance(val, str):
        return val.replace('\x00', '')
    return val

def merge_parquet_to_postgres():
    print(f"Loading Parquet data from: {PARQUET_PATH}...")
    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas(types_mapper=None)
    
    print(f"Loaded {len(df):,} rows from Parquet.")
    
    print("Sanitizing text columns (removing NUL bytes)...")
    for col in ["content", "doc_title", "document_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('\x00', '', regex=False)

    conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

    print(f"Connecting to PostgreSQL (`{DB_CONFIG['dbname']}`)...")
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            print("Ensuring pgvector extension is enabled...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            
            register_vector(conn)

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

            batch_size = 5000
            total_rows = len(df)
            
            print(f"Starting bulk insert into `document_chunks` table...")
            
            for start_idx in tqdm(range(0, total_rows, batch_size), desc="Inserting Batches"):
                batch_df = df.iloc[start_idx : start_idx + batch_size]
                
                records = [
                    (
                        clean_nul_bytes(str(row["document_id"])),
                        clean_nul_bytes(str(row["doc_title"])),
                        int(row["chunk_index"]),
                        clean_nul_bytes(str(row["content"])),
                        row["embedding"].tolist() if hasattr(row["embedding"], "tolist") else row["embedding"]
                    )
                    for _, row in batch_df.iterrows()
                ]

                cur.executemany(
                    """
                    INSERT INTO document_chunks (document_id, doc_title, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector);
                    """,
                    records
                )
                conn.commit()

    print(f"Successfully merged master_chunks.parquet into `{DB_CONFIG['dbname']}.document_chunks` table!")

if __name__ == "__main__":
    merge_parquet_to_postgres()