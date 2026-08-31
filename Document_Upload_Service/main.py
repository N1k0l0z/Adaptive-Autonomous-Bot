import os
import re
import requests
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import date, datetime
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
PDF_DIR = os.getenv("PDF_DIR")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")

BATCH_SIZE = 128

app = FastAPI(title="Document Ingestion Service")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\u0000", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_db_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5
        )
    except Exception:
        raise SystemExit("No database found.")


def fetch_all_embeddings(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = requests.post(EMBEDDING_SERVICE_URL, json={"texts": batch})
        response.raise_for_status()
        all_embeddings.extend(response.json()["embeddings"])
    return all_embeddings


def process_pdf(file_path: str, filename: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM document_chunks WHERE document_id = %s LIMIT 1;", (filename,))
            if cur.fetchone():
                return {"status": "skipped", "message": "Document already ingested."}

        reader = PdfReader(file_path)
        doc_title = clean_text(reader.metadata.title) if reader.metadata and reader.metadata.title else filename
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        cleaned_text = clean_text(full_text)

        if not cleaned_text:
            return {"status": "skipped", "message": "No text extracted from PDF."}

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=150,
            separators=["\n\n", ". ", "\n", " ", ""]
        )        
        chunks = [clean_text(c) for c in text_splitter.split_text(cleaned_text) if clean_text(c)]

        if not chunks:
            return {"status": "skipped", "message": "No chunks generated."}

        embeddings = fetch_all_embeddings(chunks)

        records = [
            (filename, doc_title, idx, chunk, vector)
            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings))
        ]

        insert_query = """
            INSERT INTO document_chunks (document_id, doc_title, chunk_index, content, embedding)
            VALUES %s;
        """
        with conn.cursor() as cur:
            execute_values(cur, insert_query, records)
            conn.commit()

    return {"status": "success", "chunks_ingested": len(records)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(PDF_DIR, exist_ok=True)
    destination_path = os.path.join(PDF_DIR, file.filename)

    with open(destination_path, "wb") as buffer:
        buffer.write(await file.read())

    result = process_pdf(destination_path, file.filename)
    return {"filename": file.filename, **result}