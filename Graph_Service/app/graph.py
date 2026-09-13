import copy
import json
import yaml
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from google.genai.errors import APIError
from langgraph.graph import END, StateGraph

from app.config import MODEL_NAME, client
from app.PromptSchema import get_evaluator_prompts, get_planner_prompts
from app.schemas import GraphState
from app.services import fetch_conversation_history, perform_vector_search
from app.ToolsSchema import AutonomousExecutionBlueprint, EvaluatorDecision
from utils import (
    format_history_for_planner_prompt,
    parse_raw_conversation_history,
    execute_vector_search_agent,
    build_model_guidelines_string,
)

MAX_REVISION_ITERATIONS = 2

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
model_registry = config.get("model_registry", {})

model_guidelines = build_model_guidelines_string(model_registry)


def resolve_model_name(model_alias: Any) -> str:
    """Safely extracts model string regardless of registry layout (dict or str)."""
    if not model_alias:
        return MODEL_NAME

    key = str(model_alias).lower().strip()
    entry = model_registry.get(key) or model_registry.get("default")

    if isinstance(entry, dict):
        return entry.get("model_name", MODEL_NAME)
    if isinstance(entry, str):
        return entry

    return MODEL_NAME


def sanitize_blueprint_for_evaluator(blueprint: Dict[str, Any]) -> Dict[str, Any]:
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


def run_llm(model: str, prompt: str, max_retries: int = 3) -> str:
    selected_model = resolve_model_name(model)
    delay = 2

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=selected_model, contents=prompt
            ).text
        except APIError as e:
            if getattr(e, "code", None) == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e


def resolve_prompt_template(
    template: str, node_outputs: Dict[str, Any]
) -> str:
    if not template or not isinstance(template, str):
        return ""

    resolved = template

    # Matches both {{node_1}} and {node_1} placeholders cleanly
    pattern = r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}|\{\s*([a-zA-Z0-9_]+)\s*\}"
    matches = re.findall(pattern, template)

    for match in matches:
        ph = match[0] or match[1]
        matched_output = None

        if ph in node_outputs:
            matched_output = node_outputs[ph]
        else:
            for nid, output in node_outputs.items():
                short_nid = (
                    f"node_{nid.rsplit('_', 1)[-1]}" if "_" in nid else nid
                )
                if nid == ph or short_nid == ph or ph in nid:
                    matched_output = output
                    break

        if matched_output is not None:
            if isinstance(matched_output, list):
                chunk_strings = []
                for c in matched_output:
                    if isinstance(c, dict) and ("chunk_id" in c or "content" in c):
                        chunk_id = c.get("chunk_id", "N/A")
                        content = c.get("content", "")
                        meta = c.get("metadata", {})

                        meta_str = (
                            json.dumps(meta, ensure_ascii=False)
                            if isinstance(meta, dict) and meta
                            else "{}"
                        )

                        chunk_block = (
                            f"--- Chunk ID: {chunk_id} ---\n"
                            f"Metadata: {meta_str}\n"
                            f"Content:\n{content}"
                        )
                        chunk_strings.append(chunk_block)
                    else:
                        chunk_strings.append(str(c))
                rendered = "\n\n".join(chunk_strings)
            else:
                rendered = str(matched_output)

            # Replaces exact {{ph}} or {ph} without leaving trailing braces
            sub_pattern = (
                r"\{\{\s*" + re.escape(ph) + r"\s*\}\}|\{\s*" + re.escape(ph) + r"\s*\}"
            )
            resolved = re.sub(sub_pattern, lambda m: rendered, resolved)

    return resolved


def _execute_task_node(
    node: Dict[str, Any],
    node_outputs: Dict[str, Any],
    raw_question: str,
    conv_id: str,
) -> Dict[str, Any]:
    node_type = node.get("node_type")
    started = time.time()

    try:
        if node_type == "vector_search":
            raw_query = node.get("search_query") or raw_question
            # Interpolate query if search_query contains placeholders
            query = resolve_prompt_template(raw_query, node_outputs) or raw_query
            top_k = node.get("top_k") or 5

            node["search_query"] = query
            node["runtime_output"] = execute_vector_search_agent(
                query=query, top_k=top_k, min_sim=0.0
            )
            node["status"] = "EXECUTED"

        elif node_type == "fetch_history":
            raw_limit = node.get("fetch_history_limit") or 5
            limit = max(int(raw_limit), 5)

            raw_history = fetch_conversation_history(conv_id=conv_id, limit=limit)
            parsed_history = parse_raw_conversation_history(raw_history)
            formatted_history = format_history_for_planner_prompt(parsed_history)

            node["runtime_output"] = formatted_history
            node["status"] = "EXECUTED"
            node["prompt_template"] = None
            node["resolved_prompt"] = None

        elif node_type == "clarify_user_intent":
            node["runtime_output"] = node.get("question_to_ask")
            node["reasoning"] = "Clarification requested mid-plan."
            node["status"] = "EXECUTED"
            node["prompt_template"] = None
            node["resolved_prompt"] = None

        elif node.get("prompt_template"):
            context_dict = dict(node_outputs)
            context_dict.setdefault("question", raw_question)

            resolved = resolve_prompt_template(
                node["prompt_template"], context_dict
            )
            node["resolved_prompt"] = resolved
            node["runtime_output"] = run_llm(node.get("model"), resolved)
            node["status"] = "EXECUTED"

        else:
            node["status"] = "SKIPPED_UNKNOWN_TYPE"
            node["runtime_output"] = None
            node["reasoning"] = (
                f"No handler for node_type '{node_type}' and no prompt_template."
            )

    except Exception as e:
        node["status"] = "FAILED"
        node["error"] = str(e)
        node["runtime_output"] = None

    finished = time.time()
    node.update({
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": round(finished - started, 3),
    })
    return node


def planner_agent_node(state: GraphState) -> Dict[str, Any]:
    timeline = list(state.get("execution_timeline") or [])

    raw_history = state.get("chat_history", [])
    parsed_history = parse_raw_conversation_history(raw_history)
    formatted_history = format_history_for_planner_prompt(parsed_history)

    state["retry_count"] = state.get("iteration_count", 0)
    sys_instruction, user_prompt = get_planner_prompts(state, formatted_history)

    is_revision = state.get("status") == "NEEDS_REVISION"
    current_iteration = state.get("iteration_count", 0)
    started = time.time()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config={
            "system_instruction": sys_instruction,
            "response_mime_type": "application/json",
            "response_schema": AutonomousExecutionBlueprint,
            "temperature": 0,
        },
    )
    raw_response_text = response.text
    blueprint = json.loads(raw_response_text)
    finished = time.time()

    nodes = blueprint.get("nodes", [])
    prefix = f"rev{current_iteration}_" if is_revision else ""
    id_map = {}

    for idx, n in enumerate(nodes):
        old_id = n.get("id")
        new_id = f"{prefix}node_{idx + 1}"
        if old_id:
            id_map[old_id] = new_id
        n["id"] = new_id
        n.setdefault("status", "PENDING")

    raw_edges = blueprint.get("edges", [])
    updated_edges = []
    for e in raw_edges:
        src = id_map.get(e.get("source"))
        tgt = id_map.get(e.get("target"))
        if src and tgt and src != tgt:
            updated_edges.append({"source": src, "target": tgt})

    blueprint["nodes"] = nodes
    blueprint["edges"] = updated_edges

    planner_trace = {
        "step": len(timeline) + 1,
        "phase": (
            f"REPLANNING_ITERATION_{current_iteration}"
            if is_revision
            else "INITIAL_PLANNING"
        ),
        "agent": "PlannerAgent",
        "node_id": (
            f"planner_rev_{current_iteration}"
            if is_revision
            else "planner_initial"
        ),
        "status": "EXECUTED",
        "blueprint": copy.deepcopy(blueprint),
        "prompt_used": user_prompt,
        "system_instruction": sys_instruction,
        "runtime_output": raw_response_text,
        "input": {
            "question": state["question"],
            "history": formatted_history,
        },
        "duration_seconds": round(finished - started, 3),
    }

    timeline.append(planner_trace)

    clarify_node = next(
        (n for n in nodes if n.get("node_type") == "clarify_user_intent"), None
    )

    if clarify_node:
        clarification_text = (
            clarify_node.get("question_to_ask")
            or "Hello! How can I help you today?"
        )

        return {
            "blueprint": blueprint,
            "planner_trace": planner_trace,
            "status": "CLARIFICATION_NEEDED",
            "clarification_question": clarification_text,
            "final_answer": clarification_text,
            "execution_timeline": timeline,
        }

    return {
        "blueprint": blueprint,
        "planner_trace": planner_trace,
        "status": "EXECUTING",
        "execution_timeline": timeline,
    }


def executor_agent_node(state: GraphState) -> Dict[str, Any]:
    blueprint = state.get("blueprint") or {}

    nodes = copy.deepcopy(blueprint.get("nodes", []))
    edges = blueprint.get("edges", [])
    node_outputs = dict(state.get("node_outputs") or {})
    raw_question = state["question"]
    conv_id = state["conv_id"]
    current_iteration = state.get("iteration_count", 0)

    nodes_by_id = {n["id"]: n for n in nodes}

    deps = {}
    for n_id in nodes_by_id:
        source_ids = set()
        for e in edges:
            src = e.get("source") if isinstance(e, dict) else getattr(e, "source", None)
            tgt = e.get("target") if isinstance(e, dict) else getattr(e, "target", None)
            if tgt == n_id and src in nodes_by_id and src != n_id:
                source_ids.add(src)
        deps[n_id] = source_ids

    executed = {n_id for n_id, n in nodes_by_id.items() if n.get("status") == "EXECUTED"}
    remaining = set(nodes_by_id) - executed
    wave_counter = 1

    while remaining:
        ready = [n_id for n_id in remaining if deps[n_id].issubset(executed)]

        if not ready:
            for n_id in remaining:
                nodes_by_id[n_id]["status"] = "SKIPPED_UNRESOLVED_DEPENDENCY"
            break

        is_parallel_wave = len(ready) > 1
        batch_id = (
            f"batch_p{current_iteration + 1}_w{wave_counter}"
            if is_parallel_wave
            else None
        )

        for n_id in ready:
            nodes_by_id[n_id]["batch_id"] = batch_id

        with ThreadPoolExecutor(max_workers=max(len(ready), 1)) as pool:
            futures = {
                pool.submit(
                    _execute_task_node,
                    nodes_by_id[n_id],
                    dict(node_outputs),
                    raw_question,
                    conv_id,
                ): n_id
                for n_id in ready
            }

            for fut in as_completed(futures):
                n_id = futures[fut]
                res = fut.result()
                nodes_by_id[n_id] = res
                node_outputs[n_id] = res.get("runtime_output")

        executed.update(ready)
        remaining -= set(ready)
        wave_counter += 1

    nodes_ordered = list(nodes_by_id.values())

    synthesis_nodes = [
        n
        for n in nodes_ordered
        if n.get("status") == "EXECUTED" and n.get("prompt_template")
    ]

    if synthesis_nodes:
        final_ans = synthesis_nodes[-1].get("runtime_output", "")
    elif nodes_ordered:
        final_ans = nodes_ordered[-1].get("runtime_output", "")
    else:
        final_ans = ""

    if not final_ans:
        failed_nodes = [n for n in nodes_ordered if n.get("status") == "FAILED"]
        if failed_nodes:
            errors = [f"[{n.get('id')}] {n.get('error')}" for n in failed_nodes]
            final_ans = f"Execution error: {'; '.join(errors)}"

    if isinstance(final_ans, list):
        final_ans = str(final_ans)

    timeline = list(state.get("execution_timeline") or [])

    for n in nodes_ordered:
        node_type = n.get("node_type", "")
        formatted_node_type = (
            node_type.replace("_", " ").title() if node_type else ""
        )

        agent_name = (
            n.get("assigned_agent")
            or (f"{formatted_node_type} Agent" if formatted_node_type else None)
            or "ExecutorAgent"
        )

        raw_output = n.get("runtime_output")

        # Resolve model name for timeline tracing
        assigned_model = (
            resolve_model_name(n.get("model"))
            if node_type not in ("vector_search", "fetch_history", "clarify_user_intent")
            else None
        )

        sanitized_node_data = dict(n)
        if node_type == "vector_search":
            sanitized_node_data["runtime_output"] = raw_output

        timeline.append({
            "step": len(timeline) + 1,
            "agent": agent_name,
            "node_id": n.get("id"),
            "node_type": node_type,
            "model": assigned_model,  
            "batch_id": n.get("batch_id"),
            "status": n.get("status", "EXECUTED"),
            "runtime_output": raw_output,
            "prompt_template": n.get("prompt_template"),
            "resolved_prompt": n.get("resolved_prompt"),
            "search_query": n.get("search_query"),
            "top_k": n.get("top_k"),
            "reasoning": n.get("reasoning"),
            "duration_seconds": n.get("duration_seconds"),
            "node_data": sanitized_node_data,
        })

    return {
        "blueprint": {**blueprint, "nodes": nodes_ordered},
        "node_outputs": node_outputs,
        "final_answer": final_ans,
        "status": "EXECUTING",
        "execution_timeline": timeline,
    }


def evaluator_agent_node(state: GraphState) -> Dict[str, Any]:
    raw_history = state.get("chat_history", [])
    parsed_history = parse_raw_conversation_history(raw_history)
    formatted_history = format_history_for_planner_prompt(parsed_history)

    eval_state = copy.deepcopy(state)
    eval_state["blueprint"] = sanitize_blueprint_for_evaluator(
        state.get("blueprint", {})
    )

    sys_instruction, user_prompt = get_evaluator_prompts(
        eval_state, formatted_history
    )

    started = time.time()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config={
            "system_instruction": sys_instruction,
            "response_mime_type": "application/json",
            "response_schema": EvaluatorDecision,
        },
    )
    decision = json.loads(response.text)
    finished = time.time()

    action = decision.get("action", "APPROVE")
    is_sufficient = decision.get("is_sufficient", True)
    reasoning = decision.get("reasoning", "")
    recommendation = decision.get("recommendation_for_planner", "")
    question_to_ask = decision.get("question_to_ask")

    full_revision_critique = (
        f"{reasoning}\n\nPlanner Recommendation: {recommendation}"
        if recommendation
        else reasoning
    )

    current_iteration = state.get("iteration_count", 0) + 1

    eval_record = {
        "iteration": current_iteration,
        "action": action,
        "is_sufficient": is_sufficient,
        "reasoning": reasoning,
        "recommendation_for_planner": recommendation,
        "prompt_used": user_prompt,
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": round(finished - started, 3),
    }

    updated_logs = list(state.get("evaluation_logs") or []) + [eval_record]
    timeline = list(state.get("execution_timeline") or [])

    timeline.append({
        "step": len(timeline) + 1,
        "phase": f"EVALUATION_ITERATION_{current_iteration}",
        "agent": "EvaluatorAgent",
        "node_id": f"evaluator_iter_{current_iteration}",
        "status": action,
        "reasoning": reasoning,
        "action": action,
        "is_sufficient": is_sufficient,
        "question_to_ask": question_to_ask,
        "duration_seconds": round(finished - started, 3),
    })

    if action == "NEEDS_REVISION":
        return {
            "status": "NEEDS_REVISION",
            "revision_reasoning": full_revision_critique,
            "iteration_count": current_iteration,
            "evaluation_logs": updated_logs,
            "execution_timeline": timeline,
        }

    elif action == "NEEDS_CLARIFICATION":
        return {
            "status": "CLARIFICATION_NEEDED",
            "clarification_question": question_to_ask,
            "clarification_reasoning": reasoning,
            "final_answer": question_to_ask or reasoning,
            "evaluation_logs": updated_logs,
            "execution_timeline": timeline,
        }

    return {
        "status": "APPROVED",
        "evaluation_logs": updated_logs,
        "execution_timeline": timeline,
    }


def route_after_planner(state: GraphState) -> str:
    status = state.get("status")
    if status in ("CLARIFICATION_NEEDED", "APPROVED"):
        return END
    return "executor"


def route_after_evaluation(state: GraphState) -> str:
    if state.get("status") == "NEEDS_REVISION":
        if state.get("iteration_count", 0) >= MAX_REVISION_ITERATIONS:
            return END
        return "planner"
    return END


builder = StateGraph(GraphState)

builder.add_node("planner", planner_agent_node)
builder.add_node("executor", executor_agent_node)
builder.add_node("evaluator", evaluator_agent_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "planner", route_after_planner, {"executor": "executor", END: END}
)
builder.add_edge("executor", "evaluator")
builder.add_conditional_edges(
    "evaluator", route_after_evaluation, {"planner": "planner", END: END}
)

app_graph = builder.compile()