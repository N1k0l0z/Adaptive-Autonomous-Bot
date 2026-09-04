from typing import Any, Dict, List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
import operator


class GraphState(TypedDict):
    question: str
    conv_id: str
    run_id: str
    chat_history: List[Dict[str, Any]]
    blueprint: Any
    planner_trace: Any
    node_outputs: Dict[str, Any]
    evaluation_logs: List[Dict[str, Any]]
    execution_timeline: List[Dict[str, Any]]  # Standard list (no operator.add)
    iteration_count: int
    final_answer: str
    status: str
    clarification_question: Any
    clarification_reasoning: Any

class ProcessQueryRequest(BaseModel):
    question: str
    conv_id: Optional[str] = None


class AuditTrail(BaseModel):
    run_id: str
    conv_id: str
    question: str
    status: str
    clarification_question: Optional[str] = None
    clarification_reasoning: Optional[str] = None
    planner: Dict[str, Any] = Field(
        ..., description="{'blueprint': {'nodes': [...], 'edges': [...], 'clarification_reasoning': ...}} as originally planned"
    )
    steps: List[Dict[str, Any]] = Field(
        ..., description="Every agent that actually ran, in order: planner -> parallel DAG nodes -> evaluator pass(es). "
                          "Each entry carries input, prompt_used, runtime_output, reasoning (if any), status, and "
                          "started_at/finished_at/duration_seconds."
    )
    evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    final_answer: str


class ProcessQueryResponse(BaseModel):
    run_id: str
    conv_id: str
    question: str
    status: str
    final_answer: str

    clarification_question: Optional[str] = None
    clarification_reasoning: Optional[str] = None
    execution_timeline: List[Dict[str, Any]] = []

  


class HistoryResponse(BaseModel):
    conv_id: str
    messages: List[Dict[str, Any]]