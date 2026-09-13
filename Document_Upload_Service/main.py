import json
import os
import re
import glob
import requests
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CONTENT_DIR = os.path.join(BASE_DIR, "Raw_Content")
CHUNKS_DIR = os.path.join(BASE_DIR, "Chunks")
METADATA_PARQUET = os.path.join(CHUNKS_DIR, "full_metadata.parquet")
CHUNKS_PARQUET = os.path.join(CHUNKS_DIR, "master_chunks.parquet")

BATCH_SIZE = 128

app = FastAPI(title="Document Ingestion & Metadata Pipeline Service")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\u0000", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_str(val, default="") -> str:
    if pd.isna(val) or val is None:
        return default
    return str(val).strip()

def safe_int(val, default=0) -> int:
    if pd.isna(val) or val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def format_embedding(embedding) -> str | None:
    """Formats array into pgvector compatible string syntax '[x,y,z]'."""
    if pd.isna(embedding) or embedding is None:
        return None
    if isinstance(embedding, np.ndarray):
        embedding = embedding.flatten().tolist()
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except Exception:
            numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", embedding)
            return f"[{','.join(numbers)}]" if numbers else None
    if isinstance(embedding, list) and len(embedding) > 0:
        return str(embedding)
    return None

def format_metadata(row_meta: dict, doc_meta: dict) -> str:
    combined = {}
    if isinstance(doc_meta, dict):
        combined.update(doc_meta)
    if isinstance(row_meta, dict):
        combined.update(row_meta)
    cleaned = {str(k): v for k, v in combined.items() if pd.notna(v) and v is not None}
    return json.dumps(cleaned)

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10
    )

def fetch_embeddings_batch(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = requests.post(EMBEDDING_SERVICE_URL, json={"texts": batch})
        response.raise_for_status()
        embeddings.extend(response.json()["embeddings"])
    return embeddings

def process_raw_content_and_ingest():
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    os.makedirs(RAW_CONTENT_DIR, exist_ok=True)

    # 1. Process & Aggregate all CSV Metadata files
    csv_files = glob.glob(os.path.join(RAW_CONTENT_DIR, "*.csv"))
    if csv_files:
        meta_dfs = [pd.read_csv(f) for f in csv_files]
        full_meta_df = pd.concat(meta_dfs, ignore_index=True)
        expected_meta_cols = ['id', 'filename', 'title', 'author', 'description', 'pages', 'file_size', 'format', 'category', 'subcategory_url', 'download_url']
        for col in expected_meta_cols:
            if col not in full_meta_df.columns:
                full_meta_df[col] = None
        full_meta_df = full_meta_df[expected_meta_cols]
        full_meta_df.to_parquet(METADATA_PARQUET, index=False)
    else:
        full_meta_df = pd.DataFrame()

    meta_lookup = {}
    if not full_meta_df.empty and "filename" in full_meta_df.columns:
        meta_lookup = full_meta_df.set_index("filename").to_dict(orient="index")

    pdf_files = glob.glob(os.path.join(RAW_CONTENT_DIR, "*.pdf"))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=150,
        separators=["\n\n", ". ", "\n", " ", ""]
    )

    chunks_records = []
    chunk_global_id = 1

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        cleaned = clean_text(full_text)
        if not cleaned:
            continue

        doc_chunks = [clean_text(c) for c in text_splitter.split_text(cleaned) if clean_text(c)]
        if not doc_chunks:
            continue

        embeddings = fetch_embeddings_batch(doc_chunks)

        for idx, (chunk_text, vector) in enumerate(zip(doc_chunks, embeddings)):
            chunks_records.append({
                "id": chunk_global_id,
                "filename": filename,
                "chunk_index": idx,
                "content": chunk_text,
                "embedding": vector
            })
            chunk_global_id += 1

    master_chunks_df = pd.DataFrame(chunks_records)
    if not master_chunks_df.empty:
        master_chunks_df.to_parquet(CHUNKS_PARQUET, index=False)
    else:
        return {"status": "skipped", "message": "No chunks extracted."}

    db_records = []
    for row in master_chunks_df.to_dict(orient="records"):
        doc_id = safe_str(row.get("filename"))
        filename = safe_str(row.get("filename"))
        chunk_idx = safe_int(row.get("chunk_index"))
        content = safe_str(row.get("content"))
        embedding_str = format_embedding(row.get("embedding"))
        
        doc_meta = meta_lookup.get(filename, {})
        metadata_json = format_metadata({}, doc_meta)

        db_records.append((doc_id, filename, chunk_idx, content, embedding_str, metadata_json))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            insert_query = """
                INSERT INTO document_chunks (
                    document_id, 
                    filename, 
                    chunk_index, 
                    content, 
                    embedding, 
                    metadata
                ) VALUES %s;
            """
            execute_values(cur, insert_query, db_records)
            conn.commit()

    return {
        "status": "success",
        "metadata_files_processed": len(csv_files),
        "pdfs_processed": len(pdf_files),
        "total_chunks_ingested": len(db_records)
    }

@app.post("/upload")
async def trigger_full_ingestion(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_raw_content_and_ingest)
    return {"message": "Full document and metadata ingestion process started in background."}

@app.post("/upload")
async def upload_pdf_and_ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    destination_path = os.path.join(RAW_CONTENT_DIR, file.filename)
    with open(destination_path, "wb") as buffer:
        buffer.write(await file.read())

    result = process_raw_content_and_ingest()
    return {"filename": file.filename, **result}