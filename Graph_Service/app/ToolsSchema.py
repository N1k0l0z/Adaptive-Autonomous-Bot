from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TaskNode(BaseModel):
    id: str = Field(
        description="Unique sequential node identifier (e.g., 'node_1', 'node_2')."
    )
    node_type: str = Field(
        description=(
            "Node category. Must be one of the built-in service keywords ('vector_search', "
            "'fetch_history', 'clarify_user_intent') or a descriptive keyword for an LLM "
            "reasoning step (e.g., 'synthesis', 'query_decomposition', 'analysis')."
        )
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Target search string required if node_type is 'vector_search'.",
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of document chunks to retrieve (1 to 20) for 'vector_search'.",
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional LLM model name to use for reasoning nodes. If not specified, the default model will be used. Must be a valid model name supported by the LLM service."
        ),
    )
    fetch_history_limit: Optional[int] = Field(
        default=5,
        description="Number of prior conversation turns to retrieve for 'fetch_history'.",
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description=(
            "Required for custom LLM reasoning nodes. Prompt template string that MUST "
            "reference parent node outputs via '{node_id}' placeholders (e.g., '{node_1}')."
        ),
    )
    question_to_ask: Optional[str] = Field(
        default=None,
        description="Direct clarification question string required if node_type is 'clarify_user_intent'.",
    )


class GraphEdge(BaseModel):
    source: str = Field(description="Upstream dependency node ID (e.g., 'node_1').")
    target: str = Field(description="Downstream dependent node ID (e.g., 'node_2').")


class AutonomousExecutionBlueprint(BaseModel):
    clarification_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning explaining why user clarification is necessary before planning can proceed, if applicable.",
    )
    nodes: List[TaskNode] = Field(
        description="Execution nodes forming the Directed Acyclic Graph (DAG)."
    )
    edges: List[GraphEdge] = Field(
        default_factory=list,
        description="Directed dependency links connecting source nodes to target nodes.",
    )


from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EvaluatorDecision(BaseModel):
    action: Literal["APPROVE", "NEEDS_REVISION", "NEEDS_CLARIFICATION"] = Field(
        ..., 
        description="Governance decision: APPROVE if answer is fully grounded and complete; NEEDS_REVISION if missing facts or ungrounded claims exist; NEEDS_CLARIFICATION if user intent is ambiguous."
    )
    is_sufficient: bool = Field(
        ..., 
        description="True only if answer fully satisfies the user request with 100% strict grounding in retrieved data."
    )
    coverage_score: float = Field(
        ..., 
        description="Numerical ratio (0.0 to 1.0) measuring how thoroughly the answer covers all sub-questions asked."
    )
    ungrounded_claims: List[str] = Field(
        default_factory=list,
        description="List of specific claims or facts found in the final answer that DO NOT exist in the provided chunks or history (parametric hallucinations)."
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="List of specific facts or details requested by the user that were omitted from the final answer."
    )
    recommendation_for_planner: Optional[str] = Field(
        default=None,
        description="Actionable, concrete instructions for the Planner on what specific queries to search or nodes to execute during revision."
    )
    question_to_ask: Optional[str] = Field(
        default=None,
        description="Direct clarification question to surface to the user if action is NEEDS_CLARIFICATION."
    )
    reasoning: str = Field(
        ..., 
        description="Detailed analytical critique explaining the audit logic, coverage gaps, and grounding validation."
    )