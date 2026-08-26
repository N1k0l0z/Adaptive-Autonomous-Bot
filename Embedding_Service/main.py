import os
import torch
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from Schemas import EmbeddingRequest, EmbeddingResponse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = "cuda"
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
print(f"Loading model on {device.upper()}...")

model = SentenceTransformer(MODEL_NAME, device=device)

if use_fp16:
    model.half()

with torch.inference_mode():
    _ = model.encode(["warmup query text"], normalize_embeddings=True)

app = FastAPI(title="BGE-M3 Embedding Microservice")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME, "device": device, "fp16": use_fp16}


@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    input_data = request.texts if request.texts is not None else request.text

    if input_data is None:
        raise HTTPException(status_code=422, detail="Request payload must contain either 'texts' or 'text'.")

    input_texts = [input_data] if isinstance(input_data, str) else input_data
    cleaned_texts = [t.strip() for t in input_texts if isinstance(t, str) and t.strip()]

    if not cleaned_texts:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    with torch.inference_mode():
        embeddings_array = model.encode(cleaned_texts, batch_size=128, normalize_embeddings=True)

    embeddings_list = embeddings_array.tolist()
    return EmbeddingResponse(embeddings=embeddings_list, dimensions=len(embeddings_list[0]))