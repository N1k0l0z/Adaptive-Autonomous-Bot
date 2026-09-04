import json
from typing import Any, Dict, List

def parse_raw_conversation_history(raw_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_turns: List[Dict[str, Any]] = []

    for item in raw_history:
        role = item.get("role")
        raw_msg = item.get("message")

        if role == "user":
            cleaned_turns.append({
                "role": "user",
                "message": raw_msg,
                "created_at": item.get("created_at")
            })
            continue

        if role == "assistant":
            payload: Dict[str, Any] = {}
            if isinstance(raw_msg, str):
                try:
                    payload = json.loads(raw_msg)
                except json.JSONDecodeError:
                    payload = {"final_answer": raw_msg, "status": "APPROVED"}
            elif isinstance(raw_msg, dict):
                payload = raw_msg

            status = payload.get("status", "APPROVED")
            question = payload.get("question", "")

            if status == "CLARIFICATION_NEEDED":
                q_asked = payload.get("clarification_question") or payload.get("final_answer")
                c_reason = payload.get("clarification_reasoning")

                if not c_reason or c_reason == q_asked:
                    bp = payload.get("planner", {}).get("blueprint", {})
                    c_reason = (
                        bp.get("clarification_reasoning")
                        or "User prompt requires intent clarification or missing parameter parameters."
                    )

                cleaned_turns.append({
                    "role": "assistant",
                    "status": "CLARIFICATION_NEEDED",
                    "question_asked_to_user": q_asked,
                    "clarification_reasoning": c_reason
                })

            elif status == "NEEDS_REVISION":
                eval_logs = payload.get("evaluations") or payload.get("evaluation_logs") or []
                revision_reason = payload.get("revision_reasoning", "")
                
                if not revision_reason:
                    for ev in eval_logs:
                        if isinstance(ev, dict) and ev.get("action") == "REVISE_PLAN":
                            revision_reason = ev.get("reasoning") or ev.get("feedback") or ""
                            break

                cleaned_turns.append({
                    "role": "assistant",
                    "status": "NEEDS_REVISION",
                    "user_original_question": question,
                    "previous_incomplete_answer": payload.get("final_answer", ""),
                    "revision_reasoning": revision_reason or "Evaluator requested structural plan additions."
                })

            else:
                cleaned_turns.append({
                    "role": "assistant",
                    "status": "APPROVED",
                    "final_answer": payload.get("final_answer", "")
                })

    return cleaned_turns


def format_history_for_planner_prompt(cleaned_turns: List[Dict[str, Any]]) -> str:
    if not cleaned_turns:
        return "No prior conversation history."

    formatted_lines: List[str] = []

    for idx, turn in enumerate(cleaned_turns, start=1):
        role = turn["role"].upper()

        if role == "USER":
            formatted_lines.append(f"Turn {idx} [USER]: {turn.get('message')}")

        elif role == "ASSISTANT":
            status = turn.get("status")
            if status == "CLARIFICATION_NEEDED":
                formatted_lines.append(
                    f"Turn {idx} [ASSISTANT - CLARIFICATION REQUESTED]:\n"
                    f"  - Question Asked: {turn.get('question_asked_to_user')}\n"
                    f"  - Reason: {turn.get('clarification_reasoning')}"
                )
            elif status == "NEEDS_REVISION":
                formatted_lines.append(
                    f"Turn {idx} [ASSISTANT - FAILED EVALUATION / REVISION REQUIRED]:\n"
                    f"  - Original Question: {turn.get('user_original_question')}\n"
                    f"  - Incomplete Output: {turn.get('previous_incomplete_answer')}\n"
                    f"  - Evaluation Critique: {turn.get('revision_reasoning')}"
                )
            else:
                formatted_lines.append(
                    f"Turn {idx} [ASSISTANT - COMPLETED]:\n"
                    f"  - Final Answer: {turn.get('final_answer')}"
                )

    return "\n\n".join(formatted_lines)



from typing import Any, Dict, List

def build_ui_graph_payload(
    planner_trace: Dict[str, Any],
    dag_nodes: List[Dict[str, Any]],
    evaluator_entries: List[Dict[str, Any]],
    raw_edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Filter out self-referential edges
    edges = []
    for e in raw_edges:
        e_dict = e.model_dump() if hasattr(e, "model_dump") else e
        if e_dict.get("source") != e_dict.get("target"):
            edges.append(e_dict)

    if not dag_nodes:
        return edges

    # Identify DAG entry and leaf nodes
    dag_ids = [n.get("id") if isinstance(n, dict) else getattr(n, "id") for n in dag_nodes]
    internal_sources = {e["source"] for e in edges}
    internal_targets = {e["target"] for e in edges}

    entry_nodes = [n_id for n_id in dag_ids if n_id not in internal_targets]
    leaf_nodes = [n_id for n_id in dag_ids if n_id not in internal_sources]

    # Connect Planner -> DAG Entry Nodes
    if planner_trace and planner_trace.get("id"):
        planner_id = planner_trace["id"]
        for entry_id in (entry_nodes or dag_ids[:1]):
            edges.append({"source": planner_id, "target": entry_id, "label": "generates plan"})

    # Connect DAG Leaf Nodes -> Evaluator
    if evaluator_entries:
        first_eval_id = evaluator_entries[0]["id"]
        for leaf_id in (leaf_nodes or dag_ids[-1:]):
            edges.append({"source": leaf_id, "target": first_eval_id, "label": "audits result"})

    return edges