import os
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence-transformers import SentenceTransformer
import torch

DATA_DIR = "./documents"
OUTPUT_FILE = "new_chunks.parquet"
MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 150

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

model = SentenceTransformer(MODEL_NAME, device=device)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

raw_records = []
pdf_files = list(Path(DATA_DIR).glob("*.pdf"))

for pdf_path in pdf_files:
    doc_name = pdf_path.name
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
            
    chunks = text_splitter.split_text(full_text)
    
    for idx, chunk_text in enumerate(chunks):
        raw_records.append({
            "document_name": doc_name,
            "chunk_index": idx,
            "chunk_text": chunk_text
        })

if raw_records:
    chunk_texts = [r["chunk_text"] for r in raw_records]
    embeddings = model.encode(
        chunk_texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    
    for i, emb in enumerate(embeddings):
        raw_records[i]["embedding"] = emb.tolist()

df = pd.DataFrame(raw_records)
df.to_parquet(OUTPUT_FILE, index=False)