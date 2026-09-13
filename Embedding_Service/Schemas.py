from pydantic import BaseModel, Field
from typing import List, Union, Optional

class EmbeddingRequest(BaseModel):
    text: Optional[Union[str, List[str]]] = Field(default=None)
    texts: Optional[List[str]] = Field(default=None)


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int