import uuid
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.graph import app_graph
from app.schemas import (
    GraphState,
    HistoryResponse,
    ProcessQueryRequest,
    ProcessQueryResponse,
)
from app.services import fetch_conversation_history, log_message_to_history

app = FastAPI(title="Autonomous Multi-Agent System", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process", response_model=ProcessQueryResponse)
def process_query(payload: ProcessQueryRequest):
    raw_question = payload.question

    conv_id = (
        payload.conv_id.strip()
        if payload.conv_id and payload.conv_id.strip()
        else f"conv_{uuid.uuid4().hex[:12]}"
    )
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    history = fetch_conversation_history(conv_id=conv_id, limit=5)

    log_message_to_history(conv_id=conv_id, role="user", message_payload=raw_question)

    initial_state: GraphState = {
        "question": raw_question,
        "conv_id": conv_id,
        "run_id": run_id,
        "chat_history": history,
        "blueprint": None,
        "planner_trace": None,
        "node_outputs": {},
        "evaluation_logs": [],
        "iteration_count": 0,
        "final_answer": "",
        "status": "PROCESSING",
        "clarification_question": None,
        "clarification_reasoning": None,
    }

    final_state = app_graph.invoke(initial_state)

    status = final_state.get("status", "APPROVED")
    final_answer = final_state.get("final_answer", "")
    blueprint = final_state.get("blueprint") or {}
    planner_trace = final_state.get("planner_trace") or {}
    evaluation_logs = final_state.get("evaluation_logs") or []
    clarification_question = final_state.get("clarification_question")
    clarification_reasoning = final_state.get("clarification_reasoning")

    dag_nodes = blueprint.get("nodes", [])
    
    # Normalize edges whether they arrive as dicts or Pydantic models
    raw_edges = blueprint.get("edges", [])
    edges = []
    for e in raw_edges:
        e_dict = e.model_dump() if hasattr(e, "model_dump") else e
        if e_dict.get("source") != e_dict.get("target"):
            edges.append(e_dict)

    evaluator_entries: List[Dict[str, Any]] = []
    for ev in evaluation_logs:
        evaluator_entries.append(
            {
                "id": f"evaluator_iter_{ev.get('iteration', len(evaluator_entries) + 1)}",
                "node_type": "evaluator_agent",
                "step_description": (
                    "Judge whether the synthesized answer covers every sub-question, "
                    "and decide APPROVE / NEEDS_CLARIFICATION / REVISE_PLAN"
                ),
                "prompt_used": ev.get("prompt_used"),
                "status": "EXECUTED",
                "runtime_output": {
                    "action": ev.get("action"),
                    "is_sufficient": ev.get("is_sufficient"),
                },
                "reasoning": ev.get("reasoning"),
                "started_at": ev.get("started_at"),
                "finished_at": ev.get("finished_at"),
                "duration_seconds": ev.get("duration_seconds"),
            }
        )

    full_trace: List[Dict[str, Any]] = []
    if planner_trace:
        full_trace.append(planner_trace)
    
    # Normalize node items
    for n in dag_nodes:
        full_trace.append(n.model_dump() if hasattr(n, "model_dump") else n)
        
    full_trace.extend(evaluator_entries)

    for n in full_trace:
        n.setdefault("assigned_agent", n.get("assigned_agent") or n.get("node_type"))
        n.setdefault("step_description", n.get("step_description") or n.get("node_type", ""))

    node_outputs: Dict[str, Any] = {
        n["id"]: n.get("runtime_output") for n in full_trace if n.get("id")
    }
    retrieved_chunks: List[Dict[str, Any]] = []
    execution_trace: List[Dict[str, Any]] = []
    
    for n in full_trace:
        node_type = n.get("node_type") or n.get("assigned_agent")
        if node_type == "planner_agent":
            execution_trace.append(
                {"agent": "PlannerAgent", "node_id": n.get("id"), "status": n.get("status")}
            )
        elif node_type in ("vector_search", "search_rag"):
            chunks = n.get("runtime_output") or []
            if isinstance(chunks, list):
                retrieved_chunks.extend(chunks)
            execution_trace.append(
                {
                    "agent": "VectorSearchAgent",
                    "node_id": n.get("id"),
                    "search_query": n.get("search_query"),
                    "retrieved_chunk_count": len(chunks) if isinstance(chunks, list) else 0,
                }
            )
        elif node_type == "fetch_history":
            execution_trace.append({"agent": "HistoryFetchAgent", "node_id": n.get("id")})
        elif node_type == "evaluator_agent":
            execution_trace.append(
                {
                    "agent": "EvaluatorAgent",
                    "node_id": n.get("id"),
                    "action": (n.get("runtime_output") or {}).get("action"),
                    "reasoning": n.get("reasoning"),
                }
            )
        else:
            output_text = n.get("runtime_output") or ""
            execution_trace.append(
                {
                    "agent": node_type or "UnknownAgent",
                    "node_id": n.get("id"),
                    "output_preview": (
                        output_text[:150]
                        if isinstance(output_text, str)
                        else str(output_text)[:150]
                    ),
                }
            )

    evaluation_logs_ui = [
        {**ev, "verdict": ev.get("action"), "feedback": ev.get("reasoning")}
        for ev in evaluation_logs
    ]

    audit_trail: Dict[str, Any] = {
        "run_id": run_id,
        "conv_id": conv_id,
        "question": raw_question,
        "status": status,
        "clarification_question": clarification_question,
        "clarification_reasoning": clarification_reasoning,
        "planner": {
            "blueprint": {
                "nodes": dag_nodes,
                "edges": edges,
                "clarification_reasoning": blueprint.get("clarification_reasoning"),
            }
        },
        "steps": full_trace,
        "evaluations": evaluation_logs,
        "final_answer": final_answer,
        "blueprint": {"nodes": full_trace, "edges": edges, "question": raw_question},
        "nodes": full_trace,
        "edges": edges,
        "node_outputs": node_outputs,
        "retrieved_chunks": retrieved_chunks,
        "execution_trace": execution_trace,
        "evaluation_logs": evaluation_logs_ui,
    }

    validated = ProcessQueryResponse(**audit_trail)

    log_message_to_history(
        conv_id=conv_id, role="assistant", message_payload=validated.model_dump()
    )

    return validated.model_dump()