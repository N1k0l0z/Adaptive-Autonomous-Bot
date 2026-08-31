from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    question: str
    conv_id: str
    run_id: str
    chat_history: List[Dict[str, Any]]
    blueprint: Optional[Dict[str, Any]]
    planner_trace: Optional[Dict[str, Any]]
    node_outputs: Dict[str, Any]
    evaluation_logs: List[Dict[str, Any]]
    iteration_count: int
    final_answer: str
    status: str  # PROCESSING | EXECUTING | CLARIFICATION_NEEDED | NEEDS_REVISION | APPROVED
    clarification_question: Optional[str]
    clarification_reasoning: Optional[str]


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


class ProcessQueryResponse(AuditTrail):
    """Adds back the old top-level fields (blueprint/nodes/edges/node_outputs/
    retrieved_chunks/execution_trace) so existing UI code written against the
    previous pipeline's response shape keeps working unchanged, alongside the
    new planner/steps/evaluations structure."""
    blueprint: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    node_outputs: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]
    evaluation_logs: List[Dict[str, Any]]


class HistoryResponse(BaseModel):
    conv_id: str
    messages: List[Dict[str, Any]]