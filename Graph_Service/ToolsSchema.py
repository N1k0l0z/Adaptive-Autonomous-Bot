from typing import List, Optional, Literal
from pydantic import BaseModel, Field

NodeType = Literal["SEARCH_RAG", "SYNTHESIZE", "CLARIFY", "USER_INPUT", "DECOMPOSE_QUERY"]

class TaskNode(BaseModel):
    id: str = Field(description="Sequential ID e.g. node_1")
    type: NodeType = Field(description="Node operation type")
    search_query: Optional[str] = Field(
        default=None, 
        description="Atomic search phrase. Only for SEARCH_RAG."
    )
    prompt_template: Optional[str] = Field(
        default=None, 
        description="Instruction with {node_id} vars. Only for SYNTHESIZE or DECOMPOSE_QUERY."
    )
    question_to_ask: Optional[str] = Field(
        default=None, 
        description="Question to user. Only for CLARIFY or USER_INPUT."
    )

class GraphEdge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")

class AutonomousExecutionBlueprint(BaseModel):
    nodes: List[TaskNode] = Field(description="Execution nodes")
    edges: List[GraphEdge] = Field(description="Directed edges")