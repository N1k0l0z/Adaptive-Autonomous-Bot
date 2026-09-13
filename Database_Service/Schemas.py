from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any

class MessageCreate(BaseModel):
    conv_id: str
    role: str
    message: str

class ChunkInsert(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    doc_title: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = {}

class SimilarityQuery(BaseModel):
    query_embedding: List[float]
    top_k: int = 5
    min_sim: float = 0.0
