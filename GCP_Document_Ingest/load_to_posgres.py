import json
import os
import re
import numpy as np
import pandas as pd
import psycopg
from tqdm import tqdm

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("POSTGRES_DB", "my_rag_db")
DB_USER = os.getenv("POSTGRES_USER", "my_db_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "super_secret_password_123")
DB_PORT = os.getenv("DB_PORT", "5432")

CONN_STR = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(
    BASE_DIR, "Document_Upload_Service", "Chunks", "master_chunks.parquet"
)
METADATA_PATH = os.path.join(
    BASE_DIR, "Document_Upload_Service", "Chunks", "full_metadata.parquet"
)


def safe_str(val, default="") -> str:
    """Converts values to string safely, preventing literal 'nan' entries."""
    if pd.isna(val) or val is None:
        return default
    return str(val).strip()


def safe_int(val, default=0) -> int:
    """Safely converts floats/NaNs to standard integers."""
    if pd.isna(val) or val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def format_embedding(embedding):
    """Parses embedding into valid pgvector string '[x, y, ...]' or None for SQL NULL."""
    if pd.isna(embedding) or embedding is None:
        return None

    if isinstance(embedding, str):
        embedding = embedding.strip()
        if not embedding or embedding in ["[]", "null", "None"]:
            return None
        try:
            embedding = json.loads(embedding)
        except Exception:
            numbers = re.findall(
                r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", embedding
            )
            if not numbers:
                return None
            return f"[{','.join(numbers)}]"

    if isinstance(embedding, np.ndarray):
        embedding = embedding.flatten().tolist()

    while (
        isinstance(embedding, list)
        and len(embedding) > 0
        and isinstance(embedding[0], list)
    ):
        embedding = embedding[0]

    if not isinstance(embedding, list) or len(embedding) == 0:
        return None

    return str(embedding)


def format_metadata(row_metadata, doc_metadata) -> str:
    combined = {}

    if isinstance(doc_metadata, dict):
        combined.update(doc_metadata)
    if isinstance(row_metadata, dict):
        combined.update(row_metadata)

    # Remove NaNs and Nones before stringification
    cleaned = {
        str(k): v
        for k, v in combined.items()
        if pd.notna(v) and v is not None
    }

    return json.dumps(cleaned)


def ingest_data():
    if not os.path.exists(CHUNKS_PATH):
        print(f"Error: Master chunks file not found at {CHUNKS_PATH}")
        return

    print("Loading Parquet datasets...")
    chunks_df = pd.read_parquet(CHUNKS_PATH)

    # Build document metadata lookup map
    meta_lookup = {}
    if os.path.exists(METADATA_PATH):
        meta_df = pd.read_parquet(METADATA_PATH)
        if "filename" in meta_df.columns:
            meta_lookup = meta_df.set_index("filename").to_dict(
                orient="index"
            )
    else:
        print(
            "Warning: full_metadata.parquet not found. Ingesting chunks without enriched metadata."
        )

    print(f"Loaded {len(chunks_df)} chunks from {os.path.basename(CHUNKS_PATH)}.")

    with psycopg.connect(CONN_STR) as conn:
        with conn.cursor() as cur:
            # 1. Truncate existing data
            print("Truncating existing records in `document_chunks` table...")
            cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")

            # 2. Stream enriched records via COPY
            print("Streaming cleaned and enriched records into PostgreSQL...")
            copy_query = """
                COPY document_chunks (
                    document_id, 
                    filename, 
                    chunk_index, 
                    content, 
                    embedding, 
                    metadata
                ) FROM STDIN
            """

            with cur.copy(copy_query) as copy:
                for row in tqdm(
                    chunks_df.to_dict(orient="records"),
                    desc="Ingesting Chunks",
                ):
                    doc_id = safe_str(row.get("document_id"))
                    filename = safe_str(
                        row.get("filename") or row.get("doc_title"),
                        default="unknown",
                    )
                    chunk_index = safe_int(row.get("chunk_index"))
                    content = safe_str(row.get("content"))

                    embedding_str = format_embedding(row.get("embedding"))

                    doc_meta = meta_lookup.get(filename, {})
                    row_meta = row.get("metadata")
                    metadata_str = format_metadata(row_meta, doc_meta)

                    copy.write_row((
                        doc_id,
                        filename,
                        chunk_index,
                        content,
                        embedding_str,
                        metadata_str,
                    ))

        conn.commit()
    print("\nDatabase cleanup, metadata enrichment, and ingestion completed successfully!")


if __name__ == "__main__":
    ingest_data()