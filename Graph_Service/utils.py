import json
from typing import Any, Dict, List

from app.services import perform_vector_search


def parse_raw_conversation_history(
    raw_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cleaned_turns: List[Dict[str, Any]] = []

    for item in raw_history:
        role = item.get("role")
        raw_msg = item.get("message")

        if role == "user":
            cleaned_turns.append({
                "role": "user",
                "message": raw_msg,
                "created_at": item.get("created_at"),
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

            # Extract chunk metadata preserved in execution timeline nodes
            sources_metadata: List[Dict[str, Any]] = []
            timeline = payload.get("execution_timeline") or []
            for step in timeline:
                if step.get("node_type") == "vector_search":
                    runtime_output = step.get("runtime_output") or []
                    if isinstance(runtime_output, list):
                        for chunk in runtime_output:
                            if isinstance(chunk, dict) and "metadata" in chunk:
                                meta = chunk["metadata"]
                                # Avoid duplicate metadata entries
                                if meta not in sources_metadata:
                                    sources_metadata.append(meta)

            if status == "CLARIFICATION_NEEDED":
                q_asked = payload.get(
                    "clarification_question"
                ) or payload.get("final_answer")
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
                    "clarification_reasoning": c_reason,
                })

            elif status == "NEEDS_REVISION":
                eval_logs = (
                    payload.get("evaluations")
                    or payload.get("evaluation_logs")
                    or []
                )
                revision_reason = payload.get("revision_reasoning", "")

                if not revision_reason:
                    for ev in eval_logs:
                        if (
                            isinstance(ev, dict)
                            and ev.get("action") == "REVISE_PLAN"
                        ):
                            revision_reason = (
                                ev.get("reasoning")
                                or ev.get("feedback")
                                or ""
                            )
                            break

                cleaned_turns.append({
                    "role": "assistant",
                    "status": "NEEDS_REVISION",
                    "user_original_question": question,
                    "previous_incomplete_answer": payload.get(
                        "final_answer", ""
                    ),
                    "revision_reasoning": revision_reason
                    or "Evaluator requested structural plan additions.",
                })

            else:
                cleaned_turns.append({
                    "role": "assistant",
                    "status": "APPROVED",
                    "final_answer": payload.get("final_answer", ""),
                    "sources_metadata": sources_metadata,
                })

    return cleaned_turns


def format_history_for_planner_prompt(
    cleaned_turns: List[Dict[str, Any]],
) -> str:
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
                sources = turn.get("sources_metadata", [])
                source_str = ""
                if sources:
                    formatted_sources = [
                        f"    * {json.dumps(m, ensure_ascii=False)}"
                        for m in sources
                    ]
                    source_str = (
                        "\n  - Referenced Document Metadata:\n"
                        + "\n".join(formatted_sources)
                    )

                formatted_lines.append(
                    f"Turn {idx} [ASSISTANT - COMPLETED]:\n"
                    f"  - Final Answer: {turn.get('final_answer')}"
                    f"{source_str}"
                )

    return "\n\n".join(formatted_lines)

def sanitize_blueprint_for_evaluator(
    blueprint: Dict[str, Any],
) -> Dict[str, Any]:
    sanitized_nodes = []
    for node in blueprint.get("nodes", []):
        sn = {
            "id": node.get("id"),
            "node_type": node.get("node_type"),
            "status": node.get("status"),
        }
        if node.get("search_query"):
            sn["search_query"] = node.get("search_query")
        if node.get("question_to_ask"):
            sn["question_to_ask"] = node.get("question_to_ask")
        sanitized_nodes.append(sn)

    return {
        "nodes": sanitized_nodes,
        "edges": blueprint.get("edges", []),
    }


def run_llm_prompt_with_retry(prompt: str, max_retries: int = 3) -> str:
    delay = 2
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME, contents=prompt
            ).text
        except APIError as e:
            if getattr(e, "code", None) == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e


def execute_vector_search_agent(
    query: str, top_k: int = 5, min_sim: float = 0.0
) -> List[Dict[str, Any]]:
    raw_results = perform_vector_search(
        query=query, top_k=top_k, min_sim=min_sim
    )

    formatted_chunks = []
    for idx, c in enumerate(raw_results):
        meta = c.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        formatted_chunks.append({
            "chunk_id": c.get("chunk_id") or c.get("id") or (idx + 1),
            "content": c.get("content")
            or c.get("text")
            or c.get("page_content")
            or "",
            "metadata": meta,
        })

    return formatted_chunks


DEFAULT_MODEL = "gemini-2.5-flash"

def resolve_model_name(model_alias: Any) -> str:
    """Safely extracts target model string from input or model_registry."""
    if not model_alias:
        return MODEL_NAME

    key = str(model_alias).strip()
    if isinstance(model_registry, dict) and key in model_registry:
        return key

    return key


def build_model_guidelines_string(registry: Dict[str, Any]) -> str:
    if not isinstance(registry, dict) or not registry:
        return (
            "- Target Model: `gemini-2.5-flash`\n- Target Model:"
            " `gemini-2.5-flash-lite`"
        )

    lines = []
    for model_name, details in registry.items():
        if isinstance(details, dict):
            desc = details.get("description", "")
            when = details.get("when_to_use", "")
            lines.append(
                f"- Target Model: `{model_name}`\n"
                f"  * Description: {desc}\n"
                f"  * Selection Criteria: {when}"
            )

    return "\n".join(lines)