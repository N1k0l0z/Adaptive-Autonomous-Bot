import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from app.config import MODEL_NAME, client
from app.PromptSchema import get_evaluator_prompts, get_planner_prompts
from app.schemas import GraphState
from app.services import fetch_conversation_history, perform_vector_search
from app.ToolsSchema import AutonomousExecutionBlueprint, EvaluatorDecision

MAX_REVISION_ITERATIONS = 2


def execute_vector_search_agent(
    query: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    raw_results = perform_vector_search(query=query, top_k=top_k)
    return [
        {
            "chunk_id": c.get("id") or c.get("chunk_id") or idx + 1,
            "content": c.get("content", ""),
            "metadata": c.get("metadata", {}),
        }
        for idx, c in enumerate(raw_results)
    ]


def resolve_prompt_template(template: str, node_outputs: Dict[str, Any]) -> str:
    resolved = template
    for prev_id, prev_output in node_outputs.items():
        placeholder = f"{{{prev_id}}}"
        if placeholder not in resolved:
            continue
        if isinstance(prev_output, list):
            chunk_str_list = []
            for c in prev_output:
                if isinstance(c, dict) and "chunk_id" in c:
                    chunk_str_list.append(
                        f"Chunk [{c.get('chunk_id', 'N/A')}]:\n{c.get('content', '')}"
                    )
                else:
                    chunk_str_list.append(str(c))
            resolved = resolved.replace(
                placeholder, "\n\n".join(chunk_str_list)
            )
        else:
            resolved = resolved.replace(placeholder, str(prev_output))
    return resolved


def run_llm_prompt(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME, contents=prompt
    )
    return response.text


def _execute_task_node(
    node: Dict[str, Any],
    node_outputs: Dict[str, Any],
    raw_question: str,
    conv_id: str,
) -> Dict[str, Any]:
    node_type = node.get("node_type")
    started = time.time()

    try:
        if node_type in ("vector_search", "search_rag"):
            query = node.get("search_query") or raw_question
            top_k = node.get("top_k") or 5
            chunks = execute_vector_search_agent(query=query, top_k=top_k)
            node["runtime_output"] = chunks
            node["status"] = "EXECUTED"

        elif node_type == "fetch_history":
            limit = node.get("fetch_history_limit") or 5
            hist = fetch_conversation_history(conv_id=conv_id, limit=limit)
            node["runtime_output"] = hist
            node["status"] = "EXECUTED"

        elif node_type == "clarify_user_intent":
            node["runtime_output"] = node.get("question_to_ask")
            node["reasoning"] = "Clarification requested mid-plan."
            node["status"] = "EXECUTED"

        elif node.get("prompt_template"):
            resolved = resolve_prompt_template(
                node["prompt_template"], node_outputs
            )
            text = run_llm_prompt(resolved)
            node["resolved_prompt"] = resolved
            node["runtime_output"] = text
            node["status"] = "EXECUTED"
            if (
                node_type == "plan_validation"
                and node.get("validation_criteria")
            ):
                node["reasoning"] = (
                    f"Validated against: {node['validation_criteria']}"
                )

        else:
            node["status"] = "SKIPPED_UNKNOWN_TYPE"
            node["runtime_output"] = None
            node["reasoning"] = (
                f"No handler for node_type '{node_type}' and no prompt_template to fall back on."
            )

    except Exception as e:
        node["status"] = "FAILED"
        node["error"] = str(e)
        node["runtime_output"] = None

    finished = time.time()
    node["started_at"] = started
    node["finished_at"] = finished
    node["duration_seconds"] = round(finished - started, 3)
    return node


def planner_agent_node(state: GraphState) -> Dict[str, Any]:
    raw_question = state["question"]
    history = state.get("chat_history", [])

    system_instruction, user_prompt = get_planner_prompts(raw_question, history)

    started = time.time()
    raw_response_text = None
    error = None

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": AutonomousExecutionBlueprint,
            },
        )
        raw_response_text = response.text
        blueprint = json.loads(response.text)
        planner_status = "EXECUTED"
    except Exception as e:
        error = str(e)
        planner_status = "FALLBACK_USED"
        blueprint = {
            "clarification_reasoning": None,
            "nodes": [
                {
                    "id": "node_1",
                    "node_type": "vector_search",
                    "search_query": raw_question,
                    "top_k": 5,
                },
                {
                    "id": "node_2",
                    "node_type": "synthesis",
                    "prompt_template": f"Answer the question using this context: {{node_1}}. Question: {raw_question}",
                },
            ],
            "edges": [{"source": "node_1", "target": "node_2"}],
        }

    finished = time.time()

    nodes = blueprint.get("nodes", [])
    for idx, n in enumerate(nodes):
        n.setdefault("id", f"node_{idx + 1}")
        n.setdefault("status", "PENDING")
    blueprint["nodes"] = nodes
    blueprint.setdefault("edges", [])

    planner_trace = {
        "id": "planner",
        "assigned_agent": "planner_agent",
        "step_description": "Contextualize/decompose the question against chat history and design a parallel-execution DAG",
        "input": {
            "question": raw_question,
            "history_turns_considered": len(history[-5:]),
        },
        "prompt_used": user_prompt,
        "system_instruction": system_instruction,
        "status": planner_status,
        "runtime_output": (
            raw_response_text
            if raw_response_text is not None
            else {"error": error, "fallback_used": True}
        ),
        "error": error,
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": round(finished - started, 3),
    }

    clarification_reasoning = blueprint.get("clarification_reasoning")
    clarify_node = next(
        (n for n in nodes if n.get("node_type") == "clarify_user_intent"), None
    )

    if clarify_node or clarification_reasoning:
        q_to_ask = (
            (clarify_node.get("question_to_ask") if clarify_node else None)
            or clarification_reasoning
        )
        return {
            "blueprint": blueprint,
            "planner_trace": planner_trace,
            "status": "CLARIFICATION_NEEDED",
            "clarification_question": (
                clarify_node.get("question_to_ask") if clarify_node else None
            ),
            "clarification_reasoning": clarification_reasoning
            or (clarify_node.get("question_to_ask") if clarify_node else None),
            "final_answer": q_to_ask,
        }

    return {
        "blueprint": blueprint,
        "planner_trace": planner_trace,
        "status": "EXECUTING",
    }


def executor_agent_node(state: GraphState) -> Dict[str, Any]:
    blueprint = state.get("blueprint") or {}
    nodes = blueprint.get("nodes", [])
    edges = blueprint.get("edges", [])
    node_outputs = dict(state.get("node_outputs", {}))
    raw_question = state["question"]
    conv_id = state["conv_id"]

    nodes_by_id = {n["id"]: n for n in nodes}
    deps: Dict[str, set] = {nid: set() for nid in nodes_by_id}
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src == tgt:
            continue
        if tgt in deps and src in nodes_by_id:
            deps[tgt].add(src)

    executed = {
        nid for nid, n in nodes_by_id.items() if n.get("status") == "EXECUTED"
    }
    remaining = set(nodes_by_id) - executed

    while remaining:
        ready = [nid for nid in remaining if deps[nid].issubset(executed)]
        if not ready:
            for nid in remaining:
                nodes_by_id[nid]["status"] = "SKIPPED_UNRESOLVED_DEPENDENCY"
            break

        with ThreadPoolExecutor(max_workers=max(len(ready), 1)) as pool:
            futures = {
                pool.submit(
                    _execute_task_node,
                    nodes_by_id[nid],
                    dict(node_outputs),
                    raw_question,
                    conv_id,
                ): nid
                for nid in ready
            }
            for fut in as_completed(futures):
                nid = futures[fut]
                result_node = fut.result()
                node_outputs[nid] = result_node.get("runtime_output")

        executed.update(ready)
        remaining -= set(ready)

    nodes_ordered = list(nodes_by_id.values())

    synthesis_like = [
        n
        for n in nodes_ordered
        if n.get("status") == "EXECUTED"
        and n.get("prompt_template")
        and n.get("node_type") != "plan_validation"
    ]
    if synthesis_like:
        final_ans = synthesis_like[-1].get("runtime_output", "")
    elif nodes_ordered:
        final_ans = nodes_ordered[-1].get("runtime_output", "")
    else:
        final_ans = ""

    if isinstance(final_ans, list):
        final_ans = str(final_ans)

    return {
        "blueprint": {**blueprint, "nodes": nodes_ordered},
        "node_outputs": node_outputs,
        "final_answer": final_ans or "",
        "status": "EXECUTING",
    }


def evaluator_agent_node(state: GraphState) -> Dict[str, Any]:
    blueprint = state.get("blueprint") or {}
    nodes = blueprint.get("nodes", [])
    final_answer = state.get("final_answer", "")
    iteration = state.get("iteration_count", 0)
    logs = list(state.get("evaluation_logs", []))
    raw_question = state["question"]

    sub_questions = [
        n.get("search_query")
        for n in nodes
        if n.get("node_type") in ("vector_search", "search_rag")
        and n.get("search_query")
    ]

    if iteration >= MAX_REVISION_ITERATIONS:
        now = time.time()
        logs.append(
            {
                "iteration": iteration,
                "action": "APPROVE",
                "is_sufficient": True,
                "reasoning": f"Max revision iterations ({MAX_REVISION_ITERATIONS}) reached — auto-approved without a model call.",
                "prompt_used": None,
                "started_at": now,
                "finished_at": now,
                "duration_seconds": 0.0,
            }
        )
        return {"status": "APPROVED", "evaluation_logs": logs}

    eval_system, eval_prompt = get_evaluator_prompts(
        question=raw_question,
        sub_questions=sub_questions,
        final_answer=final_answer,
    )

    started = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=eval_prompt,
            config={
                "system_instruction": eval_system,
                "response_mime_type": "application/json",
                "response_schema": EvaluatorDecision,
            },
        )
        decision = json.loads(response.text)
        action = decision.get("action", "APPROVE")
    except Exception as e:
        decision = {
            "action": "APPROVE",
            "is_sufficient": True,
            "reasoning": f"Evaluation call failed, auto-approved: {str(e)}",
        }
        action = "APPROVE"
    finished = time.time()

    logs.append(
        {
            "iteration": iteration,
            "action": action,
            "is_sufficient": decision.get("is_sufficient", True),
            "reasoning": decision.get("reasoning", ""),
            "prompt_used": eval_prompt,
            "started_at": started,
            "finished_at": finished,
            "duration_seconds": round(finished - started, 3),
        }
    )

    result: Dict[str, Any] = {
        "evaluation_logs": logs,
        "iteration_count": iteration + 1,
    }

    if action == "NEEDS_CLARIFICATION":
        result["status"] = "CLARIFICATION_NEEDED"
        result["clarification_question"] = decision.get("question_to_ask")
        result["clarification_reasoning"] = decision.get("reasoning")
        result["final_answer"] = decision.get("question_to_ask") or decision.get(
            "reasoning"
        )
        return result

    if action == "REVISE_PLAN":
        additional_nodes = decision.get("additional_nodes") or []
        additional_edges = decision.get("additional_edges") or []
        existing_ids = {n["id"] for n in nodes}
        for idx, n in enumerate(additional_nodes):
            if not n.get("id") or n["id"] in existing_ids:
                n["id"] = f"node_rev{iteration}_{idx + 1}"
            n.setdefault("status", "PENDING")
            existing_ids.add(n["id"])
        nodes.extend(additional_nodes)
        blueprint["nodes"] = nodes
        blueprint["edges"] = blueprint.get("edges", []) + additional_edges
        result["blueprint"] = blueprint
        result["status"] = "NEEDS_REVISION"
        return result

    result["status"] = "APPROVED"
    return result


def route_after_planner(state: GraphState) -> str:
    if state.get("status") == "CLARIFICATION_NEEDED":
        return END
    return "executor"


def route_after_evaluation(state: GraphState) -> str:
    if state.get("status") == "NEEDS_REVISION":
        return "executor"
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
    "evaluator", route_after_evaluation, {"executor": "executor", END: END}
)

app_graph = builder.compile()