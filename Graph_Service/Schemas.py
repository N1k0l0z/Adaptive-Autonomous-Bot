from typing import List, Dict, Any, TypedDict, Optional
from pydantic import BaseModel

class ProcessQueryRequest(BaseModel):
    question: str
    conv_id: Optional[str] = None

class ProcessQueryResponse(BaseModel):
    question: str
    conv_id: Optional[str]
    history_loaded_count: int
    detected_intent: str