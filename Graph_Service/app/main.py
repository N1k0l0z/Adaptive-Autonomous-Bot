import base64
import re
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


def sanitize_id(val: str) -> str:
    if not val:
        return "node"
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(val))
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def group_tasks_by_batch(task_items: List[dict]) -> List[List[dict]]:
    if not task_items:
        return []

    batches: List[List[dict]] = []
    current_batch: List[dict] = []
    current_batch_id = None

    for item in task_items:
        batch_id = item.get("batch_id")

        if batch_id is not None:
            if current_batch_id == batch_id:
                current_batch.append(item)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_batch_id = batch_id
        else:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_id = None
            batches.append([item])

    if current_batch:
        batches.append(current_batch)

    return batches


def generate_unrolled_mermaid(execution_timeline: list) -> str:
    lines = ["flowchart TD", '    Start(["Start (User Query)"])']

    if not execution_timeline or not isinstance(execution_timeline, list):
        lines.append('    Task["Process Query"]')
        lines.append("    Start --> Task")
        lines.append('    Task --> End(["End"])')
        return "\n".join(lines)

    valid_steps = []
    for item in execution_timeline:
        agent = str(
            item.get("agent")
            or item.get("assigned_agent")
            or item.get("node_type")
            or ""
        ).lower()
        phase = str(item.get("phase") or "").upper()
        if agent in ["user", "system"] or phase == "USER_INPUT":
            continue
        valid_steps.append(item)

    if not valid_steps:
        lines.append('    Task["Process Query"]')
        lines.append("    Start --> Task")
        lines.append('    Task --> End(["End"])')
        return "\n".join(lines)

    phases: List[List[dict]] = []
    current_phase: List[dict] = []

    for item in valid_steps:
        agent = str(
            item.get("agent")
            or item.get("assigned_agent")
            or item.get("node_type")
            or ""
        ).lower()

        if current_phase and ("planner" in agent):
            phases.append(current_phase)
            current_phase = [item]
        else:
            current_phase.append(item)

    if current_phase:
        phases.append(current_phase)

    prev_parents = ["Start"]

    for phase_idx, phase in enumerate(phases):
        planners = []
        middle_tasks = []
        evaluators = []

        for item in phase:
            agent = str(
                item.get("agent")
                or item.get("assigned_agent")
                or item.get("node_type")
                or ""
            ).lower()
            if "planner" in agent:
                planners.append(item)
            elif "evaluat" in agent:
                evaluators.append(item)
            else:
                middle_tasks.append(item)

        if planners:
            planner_item = planners[0]
            status = planner_item.get("status", "EXECUTED")
            planner_id = f"planner_p{phase_idx + 1}"
            p_label = (
                f"Planner Agent (Revision {phase_idx})"
                if phase_idx > 0
                else "Planner Agent"
            )
            lines.append(f'    {planner_id}["{p_label} [{status}]"]')

            for parent in prev_parents:
                if parent == "Start":
                    lines.append(f"    Start --> {planner_id}")
                else:
                    lines.append(
                        f'    {parent} -->|"Needs Revision"| {planner_id}'
                    )

            prev_parents = [planner_id]

        task_batches = group_tasks_by_batch(middle_tasks)
        task_counter = 0

        for batch in task_batches:
            batch_node_ids = []

            for item in batch:
                agent_name = (
                    item.get("agent")
                    or item.get("assigned_agent")
                    or item.get("node_type")
                    or "Task"
                )
                clean_name = str(agent_name).replace("_", " ").title()
                status = item.get("status", "EXECUTED")
                node_id = f"task_p{phase_idx + 1}_{task_counter}"
                task_counter += 1

                lines.append(f'    {node_id}["{clean_name} [{status}]"]')
                batch_node_ids.append(node_id)

                for parent in prev_parents:
                    lines.append(f"    {parent} --> {node_id}")

            prev_parents = batch_node_ids

        if evaluators:
            evaluator_id = f"evaluator_p{phase_idx + 1}"
            ev_status = evaluators[0].get("status", "APPROVED")
            lines.append(f'    {evaluator_id}["Evaluator Agent [{ev_status}]"]')
            for parent in prev_parents:
                lines.append(f"    {parent} --> {evaluator_id}")
            prev_parents = [evaluator_id]

    lines.append('    End(["End"])')
    for parent in prev_parents:
        if "evaluator" in parent:
            lines.append(f'    {parent} -->|"Approved"| End')
        else:
            lines.append(f"    {parent} --> End")

    return "\n".join(lines)


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
    log_message_to_history(
        conv_id=conv_id, role="user", message_payload=raw_question
    )

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

    initial_state.get("execution_timeline", []).append({
        "step": 1,
        "phase": "USER_INPUT",
        "agent": "User",
        "node_id": "user_query",
        "message": raw_question,
        "status": "RECEIVED",
    })

    final_state = app_graph.invoke(initial_state)

    execution_timeline = final_state.get("execution_timeline", [])
    status = final_state.get("status", "APPROVED")

    mermaid_output = generate_unrolled_mermaid(execution_timeline)

    encoded_mermaid = base64.b64encode(
        mermaid_output.encode("utf-8")
    ).decode("utf-8")
    graph_image_url = f"https://mermaid.ink/img/{encoded_mermaid}"

    audit_trail = {
        "run_id": run_id,
        "conv_id": conv_id,
        "question": raw_question,
        "status": status,
        "final_answer": (
            final_state.get("final_answer", "")
            or final_state.get("clarification_question", "")
            or ""
        ),
        "execution_timeline": execution_timeline,
        "graph_mermaid": mermaid_output,
        "graph_image_url": graph_image_url,
    }

    validated = ProcessQueryResponse(**audit_trail)

    log_message_to_history(
        conv_id=conv_id,
        role="assistant",
        message_payload=validated.model_dump(),
    )

    return validated