from typing import List, Optional
from pydantic import BaseModel, Field


class TaskNode(BaseModel):
    id: str = Field(description="Unique node identifier, e.g., 'node_1', 'node_2'")
    node_type: str = Field(
        description="Capability key or strategy descriptor (e.g., 'vector_search', 'fetch_history', 'query_decomposition', 'synthesis', 'clarify_user_intent', 'plan_validation')."
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Search query string if vector retrieval is needed."
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Top K document chunks to retrieve for vector search (1 to 20)."
    )
    fetch_history_limit: Optional[int] = Field(
        default=None,
        description="Number of past conversation turns to pull from history database."
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="LLM prompt template for reasoning/synthesis/validation nodes. Can reference parent outputs like '{node_1}'."
    )
    question_to_ask: Optional[str] = Field(
        default=None,
        description="Clarification question to display to user if requirements are underspecified."
    )
    validation_criteria: Optional[str] = Field(
        default=None,
        description="Criteria or rules for validator nodes to assess output completeness and accuracy."
    )


class GraphEdge(BaseModel):
    source: str = Field(description="Source node ID (e.g., 'node_1')")
    target: str = Field(description="Target node ID (e.g., 'node_2')")


class AutonomousExecutionBlueprint(BaseModel):
    clarification_reasoning: Optional[str] = Field(
        default=None,
        description="Detailed reasoning explaining why clarification is required, or null if execution can proceed."
    )
    nodes: List[TaskNode] = Field(description="Execution steps forming the DAG")
    edges: List[GraphEdge] = Field(default_factory=list, description="Dependencies between execution steps")


class EvaluatorDecision(BaseModel):
    action: str = Field(
        description="Must be one of: 'APPROVE', 'NEEDS_REVISION', or 'NEEDS_CLARIFICATION'"
    )
    is_sufficient: bool = Field(
        description="True if the final answer fully covers all aspects of the user query; False otherwise."
    )
    reasoning: str = Field(
        description="Comprehensive analytical critique covering completeness, accuracy, implicit intent coverage, and factual alignment."
    )
    question_to_ask: Optional[str] = Field(
        default=None,
        description="Required ONLY when action == 'NEEDS_CLARIFICATION'. The exact follow-up question for the user.",
    )
    evaluator_proposed_nodes: List[TaskNode] = Field(
        default_factory=list,
        description="New nodes recommended for the Planner when action == 'NEEDS_REVISION'.",
    )
    evaluator_proposed_edges: List[GraphEdge] = Field(
        default_factory=list,
        description="New edges recommended for the Planner when action == 'NEEDS_REVISION'.",
    )