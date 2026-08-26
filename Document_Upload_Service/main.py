import os
import re
import time
import shutil
import queue
import threading
import requests
import psycopg2
from psycopg2.extras import execute_values
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DB_HOST = os.getenv("DB_HOST", "database_pgvector")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "chathistory")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
PDF_DIR = os.getenv("PDF_DIR", "/app/Raw_Content")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding_service:8001/embed")
EMBEDDING_HEALTH_URL = EMBEDDING_SERVICE_URL.replace("/embed", "/docs")

BATCH_SIZE = 128
HTTP_TIMEOUT = 600

ingestion_queue = queue.Queue()

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\u0000", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def init_db():
    print("🛠️  Initializing database schema...", flush=True)
    for attempt in range(1, 31):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        doc_title TEXT,
                        chunk_index INT NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(1024),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
            conn.close()
            print("✅ Database schema verified and table 'document_chunks' ready.", flush=True)
            return True
        except Exception as e:
            print(f"⏳ [DB INIT RETRY {attempt}/30] Database setup waiting: {e}", flush=True)
            time.sleep(2)
    return False

def check_embedding_health() -> bool:
    try:
        res = requests.get(EMBEDDING_HEALTH_URL, timeout=3)
        return res.status_code in (200, 404, 405)
    except Exception:
        return False

def wait_for_embedding_service():
    print("⏳ Waiting for embedding service model to initialize...", flush=True)
    for attempt in range(1, 120):
        if check_embedding_health():
            print("✅ Embedding service is reachable and ready.", flush=True)
            return True
        time.sleep(3)
    print("⚠️ Warning: Embedding service check timed out. Proceeding anyway...", flush=True)
    return False

def get_db_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    except Exception as e:
        print(f"❌ DB connection error: {e}", flush=True)
        return None

def fetch_embeddings_in_batches(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        success = False
        
        for attempt in range(1, 11):
            try:
                response = requests.post(
                    EMBEDDING_SERVICE_URL,
                    json={"texts": batch},
                    timeout=HTTP_TIMEOUT
                )
                response.raise_for_status()
                all_embeddings.extend(response.json()["embeddings"])
                success = True
                break
            except Exception as e:
                print(f"⏳ [EMBEDDING RETRY {attempt}/10] Embedding service error: {e}. Retrying in 5s...", flush=True)
                time.sleep(5)
                
        if not success:
            raise RuntimeError(f"Failed to fetch embeddings after 10 retries.")
            
    return all_embeddings

def process_single_pdf(file_path: str) -> bool:
    pdf_file = os.path.basename(file_path)
    file_prefix = f"⚡ [AUTO-INGEST]"

    conn = get_db_connection()
    if not conn:
        print(f"❌ {file_prefix} Failed to connect to DB for '{pdf_file}'.", flush=True)
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM document_chunks WHERE document_id = %s LIMIT 1;", (pdf_file,))
            if cur.fetchone():
                print(f"⏭️  {file_prefix} Skipping (Already Ingested): '{pdf_file}'", flush=True)
                conn.close()
                return True

        start_time = time.time()
        print(f"📄 {file_prefix} Processing '{pdf_file}'...", flush=True)

        reader = PdfReader(file_path)
        doc_title = clean_text(reader.metadata.title) if reader.metadata and reader.metadata.title else pdf_file

        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        cleaned_text = clean_text(full_text)

        if not cleaned_text:
            print(f"⚠️  {file_prefix} Skipping '{pdf_file}': No text found.", flush=True)
            conn.close()
            return True

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = [clean_text(c) for c in text_splitter.split_text(cleaned_text) if clean_text(c)]

        if not chunks:
            conn.close()
            return True

        print(f"⚡ {file_prefix} Requesting vector embeddings for {len(chunks)} chunks...", flush=True)
        embeddings_list = fetch_embeddings_in_batches(chunks)

        records = [
            (pdf_file, doc_title, chunk_idx, chunk_text, vector)
            for chunk_idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings_list))
        ]

        print(f"💾 {file_prefix} Writing {len(records)} chunks to PostgreSQL...", flush=True)
        insert_query = """
            INSERT INTO document_chunks (document_id, doc_title, chunk_index, content, embedding)
            VALUES %s;
        """
        with conn.cursor() as cur:
            execute_values(cur, insert_query, records)
            conn.commit()

        elapsed = round(time.time() - start_time, 2)
        print(f"✅ {file_prefix} Successfully ingested '{pdf_file}' in {elapsed}s", flush=True)
        return True

    except Exception as e:
        conn.rollback()
        print(f"❌ {file_prefix} Error processing '{pdf_file}': {e}", flush=True)
        return False
    finally:
        conn.close()

def queue_worker():
    print("👷 Background ingestion queue worker thread active...", flush=True)
    while True:
        file_path = ingestion_queue.get()
        if file_path is None:
            break
        try:
            time.sleep(0.5)
            success = process_single_pdf(file_path)
            if not success:
                print(f"🔄 Re-queuing '{os.path.basename(file_path)}' due to processing failure.", flush=True)
                time.sleep(10)
                ingestion_queue.put(file_path)
        except Exception as e:
            print(f"❌ Worker thread error: {e}", flush=True)
        finally:
            ingestion_queue.task_done()

class PDFFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            print(f"🔍 [WATCHDOG] Detected new PDF file drop: '{event.src_path}'", flush=True)
            ingestion_queue.put(event.src_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(PDF_DIR, exist_ok=True)
    init_db()
    wait_for_embedding_service()

    worker_thread = threading.Thread(target=queue_worker, daemon=True)
    worker_thread.start()

    print(f"📂 Scanning '{PDF_DIR}' for existing documents...", flush=True)
    if os.path.exists(PDF_DIR):
        existing_files = sorted([os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
        print(f"📊 Queuing {len(existing_files)} files for automatic background ingestion.", flush=True)
        for file_path in existing_files:
            ingestion_queue.put(file_path)

    event_handler = PDFFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=PDF_DIR, recursive=False)
    observer.start()
    print("🚀 Auto-Ingestion system fully initialized.", flush=True)

    yield

    observer.stop()
    observer.join()

app = FastAPI(
    title="Auto-Ingestion Document Service",
    lifespan=lifespan
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    destination_path = os.path.join(PDF_DIR, file.filename)
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")
    finally:
        file.file.close()

    ingestion_queue.put(destination_path)
    return {"status": "uploaded", "filename": file.filename, "queue_position": ingestion_queue.qsize()}

@app.get("/status")
def get_queue_status():
    return {"pending_queue_count": ingestion_queue.qsize()}