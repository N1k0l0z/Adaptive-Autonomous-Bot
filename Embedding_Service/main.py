import os
import re
import time
import threading
import psycopg2
import torch
from psycopg2.extras import execute_values
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from contextlib import asynccontextmanager

# Enable CPU fallback for unsupported PyTorch MPS operators on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Environment configurations
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "chathistory")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
PDF_DIR = os.getenv("PDF_DIR", "/app/Raw_Content")

# --- Hardware & Engine Acceleration Setup ---
if torch.cuda.is_available():
    device = "cuda"
    # Enable TensorFloat-32 (TF32) on Ampere+ GPUs for faster math without accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    use_fp16 = True
elif torch.backends.mps.is_available():
    device = "mps"
    use_fp16 = True
else:
    device = "cpu"
    use_fp16 = False

MODEL_NAME = "BAAI/bge-m3"
print(f"🚀 Loading model '{MODEL_NAME}' on device: {device.upper()} (FP16: {use_fp16})...")

model = SentenceTransformer(MODEL_NAME, device=device)

if use_fp16:
    model.half()  # Convert model weights to FP16 for hardware acceleration

# Warmup pass to allocate buffers and compile engine kernels
print("🔥 Running model warmup pass...")
with torch.inference_mode():
    _ = model.encode(["warmup query text"], normalize_embeddings=True)

print(f"✅ BGE-M3 Model loaded and optimized on {device.upper()}!")


# --- Helper Functions & Ingestion Logic ---

def clean_text(text: str) -> str:
    """Removes null bytes and cleans up excess whitespace."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\u0000", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_pdf_ingestion():
    """Reads PDFs, checks existing document_ids in Postgres, and ingests missing files."""
    print("🚀 Starting background PDF ingestion workflow...")

    conn = None
    for attempt in range(1, 15):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print("✅ Ingestion thread connected to database.")
            break
        except Exception:
            print(f"⏳ Waiting for database connection... (Attempt {attempt}/15)")
            time.sleep(3)

    if not conn:
        print("❌ Ingestion stopped: Could not establish DB connection after retries.")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT document_id FROM document_chunks WHERE document_id IS NOT NULL;")
            existing_files = {row[0] for row in cur.fetchall()}
        print(f"📊 Found {len(existing_files)} document(s) already in DB.")
    except Exception as e:
        print(f"❌ Failed to fetch existing files: {e}")
        conn.close()
        return

    if not os.path.exists(PDF_DIR):
        print(f"⚠️ Directory '{PDF_DIR}' does not exist. Skipping ingestion.")
        conn.close()
        return

    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print(f"⚠️ No PDF files found in '{PDF_DIR}'.")
        conn.close()
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    total_files = len(pdf_files)

    for idx, pdf_file in enumerate(pdf_files, start=1):
        file_prefix = f"[{idx}/{total_files}]"
        file_path = os.path.join(PDF_DIR, pdf_file)

        if pdf_file in existing_files:
            print(f"⏭️  {file_prefix} Skipping (Already Ingested): '{pdf_file}'")
            continue

        start_time = time.time()
        try:
            reader = PdfReader(file_path)
            
            doc_title = None
            if reader.metadata and reader.metadata.title:
                doc_title = clean_text(reader.metadata.title)
            if not doc_title:
                doc_title = pdf_file

            print(f"📄 {file_prefix} Processing '{pdf_file}' (Title: '{doc_title}')...")

            full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            cleaned_full_text = clean_text(full_text)

            if not cleaned_full_text:
                print(f"⚠️  {file_prefix} Warning: No extractable text found. Skipping.")
                continue

            chunks = [clean_text(c) for c in splitter.split_text(cleaned_full_text) if clean_text(c)]
            if not chunks:
                continue

            print(f"⚡ {file_prefix} Encoding {len(chunks)} chunks on {device.upper()} (batch_size=128)...")
            
            # Disable gradient calculation completely for pure inference speed
            with torch.inference_mode():
                embeddings_array = model.encode(
                    chunks, 
                    batch_size=128, 
                    show_progress_bar=False, 
                    normalize_embeddings=True
                )
            
            embeddings_list = embeddings_array.tolist()

            records = [
                (pdf_file, doc_title, chunk_idx, chunk_text, vector)
                for chunk_idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings_list))
            ]

            print(f"💾 {file_prefix} Writing records to PostgreSQL...")
            insert_query = """
                INSERT INTO document_chunks (document_id, doc_title, chunk_index, content, embedding)
                VALUES %s;
            """
            with conn.cursor() as cur:
                execute_values(cur, insert_query, records)
                conn.commit()

            existing_files.add(pdf_file)
            elapsed = round(time.time() - start_time, 2)
            print(f"✅ {file_prefix} Ingested '{pdf_file}' ({len(records)} chunks) in {elapsed}s")

            # Periodically release cache to prevent memory bloat on large batches
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        except Exception as e:
            conn.rollback()
            print(f"❌ {file_prefix} Failed to process '{pdf_file}': {e}")

    conn.close()
    print("🎉 Ingestion task finished.")


# --- FastAPI Lifespan & Application ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=run_pdf_ingestion, daemon=True)
    thread.start()
    yield


app = FastAPI(
    title="BGE-M3 Embedding Microservice & Ingestor",
    description="Standalone microservice for generating 1024-dimension dense embeddings and automated PDF ingestion.",
    version="1.0.0",
    lifespan=lifespan
)


class EmbeddingRequest(BaseModel):
    text: Optional[Union[str, List[str]]] = Field(
        default=None, 
        description="Single text string or list of text strings."
    )
    texts: Optional[List[str]] = Field(
        default=None, 
        description="List of text strings to embed."
    )


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME, "device": device, "fp16": use_fp16}


@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    try:
        input_data = request.texts if request.texts is not None else request.text

        if input_data is None:
            raise HTTPException(
                status_code=422, 
                detail="Request payload must contain either 'texts' or 'text'."
            )

        if isinstance(input_data, str):
            input_texts = [input_data]
        else:
            input_texts = input_data

        cleaned_texts = [t.strip() for t in input_texts if isinstance(t, str) and t.strip()]

        if not cleaned_texts:
            raise HTTPException(
                status_code=400, 
                detail="Text input cannot be empty."
            )

        # High-throughput vectorization inside inference_mode context
        with torch.inference_mode():
            embeddings_array = model.encode(
                cleaned_texts, 
                batch_size=128, 
                normalize_embeddings=True
            )
        
        embeddings_list = embeddings_array.tolist()

        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0])
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))