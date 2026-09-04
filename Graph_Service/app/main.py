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
        "execution_timeline": [],
        "iteration_count": 0,
        "final_answer": "",
        "status": "PROCESSING",
        "clarification_question": None,
        "clarification_reasoning": None,
    }

    final_state = app_graph.invoke(initial_state)

    audit_trail = {
        "run_id": run_id,
        "conv_id": conv_id,
        "question": raw_question,
        "status": final_state.get("status", "APPROVED"),
        "final_answer": final_state.get("final_answer", "") or final_state.get("clarification_question", "") or "",
        "execution_timeline": final_state.get("execution_timeline", []),
    }

    validated = ProcessQueryResponse(**audit_trail)

    log_message_to_history(
        conv_id=conv_id, role="assistant", message_payload=validated.model_dump()
    )

    return validated