import os
import re
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

DATA_DIR = "/home/nikoloz/Raw_Content"
OUTPUT_DIR = Path("/home/nikoloz/GCP_Document_Ingest/Final_Data_Chunks")
MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

model = SentenceTransformer(MODEL_NAME, device=device)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files in {DATA_DIR}")

def clean_filename(filename: str) -> str:
    """Sanitize document name for safe file output naming."""
    name = Path(filename).stem
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return clean_name

for pdf_path in tqdm(pdf_files, desc="Processing PDF Documents"):
    doc_name = pdf_path.name
    doc_id = pdf_path.stem  
    
    safe_name = clean_filename(doc_name)
    output_file_path = OUTPUT_DIR / f"{safe_name}_chunks.parquet"
    
    if output_file_path.exists():
        tqdm.write(f"Skipping (already exists): {output_file_path.name}")
        continue

    try:
        reader = PdfReader(pdf_path)
        
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                tqdm.write(f"Skipping encrypted file: {doc_name}")
                continue

        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        if not full_text.strip():
            tqdm.write(f"No text extracted from: {doc_name}")
            continue
            
        chunks = text_splitter.split_text(full_text)
        
        doc_records = []
        for idx, chunk_text in enumerate(chunks):
            if isinstance(chunk_text, str) and chunk_text.strip():
                doc_records.append({
                    "document_id": str(doc_id),
                    "doc_title": str(doc_name),
                    "chunk_index": int(idx),
                    "content": str(chunk_text).strip()
                })

        if not doc_records:
            continue

        chunk_texts = [r["content"] for r in doc_records]
        embeddings = model.encode(
            chunk_texts,
            batch_size=64,
            show_progress_bar=False,  
            normalize_embeddings=True
        )
        
        for i, emb in enumerate(embeddings):
            doc_records[i]["embedding"] = emb.tolist()

        df = pd.DataFrame(doc_records)
        df.to_parquet(output_file_path, index=False)
        tqdm.write(f"Saved {len(df)} chunks -> {output_file_path.name}")

    except Exception as e:
        tqdm.write(f"Error reading {doc_name}: {e}")
        continue

print(f"Processing complete! All Parquet files saved to: {OUTPUT_DIR}")