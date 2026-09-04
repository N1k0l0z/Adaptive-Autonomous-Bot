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
from utils import format_history_for_planner_prompt, parse_raw_conversation_history

MAX_REVISION_ITERATIONS = 2


# ============================================================================
# HELPER FUNCTIONS & EXECUTORS
# ============================================================================

def execute_vector_search_agent(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    raw_results = perform_vector_search(query=query, top_k=top_k)
    formatted_chunks = []
    for idx, c in enumerate(raw_results):
        formatted_chunks.append({
            "chunk_id": c.get("id") or c.get("chunk_id") or idx + 1,
            "content": c.get("content", ""),
            "metadata": c.get("metadata", {}),
        })
    return formatted_chunks


def resolve_prompt_template(template: str, node_outputs: Dict[str, Any]) -> str:
    resolved = template
    for node_id, output in node_outputs.items():
        key = f"{{{node_id}}}"
        if key not in resolved:
            continue

        if isinstance(output, list):
            chunk_strings = []
            for c in output:
                if isinstance(c, dict) and "chunk_id" in c:
                    chunk_strings.append(
                        f"Chunk [{c.get('chunk_id', 'N/A')}]:\n{c.get('content', '')}"
                    )
                else:
                    chunk_strings.append(str(c))
            rendered = "\n\n".join(chunk_strings)
        else:
            rendered = str(output)
            
        resolved = resolved.replace(key, rendered)
    return resolved


def run_llm_prompt(prompt: str) -> str:
    return client.models.generate_content(model=MODEL_NAME, contents=prompt).text


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
            node["runtime_output"] = execute_vector_search_agent(query=query, top_k=top_k)
            node["status"] = "EXECUTED"

        elif node_type == "fetch_history":
            limit = node.get("fetch_history_limit") or 5
            node["runtime_output"] = fetch_conversation_history(conv_id=conv_id, limit=limit)
            node["status"] = "EXECUTED"

        elif node_type == "clarify_user_intent":
            node["runtime_output"] = node.get("question_to_ask")
            node["reasoning"] = "Clarification requested mid-plan."
            node["status"] = "EXECUTED"

        elif node.get("prompt_template"):
            resolved = resolve_prompt_template(node["prompt_template"], node_outputs)
            node["resolved_prompt"] = resolved
            node["runtime_output"] = run_llm_prompt(resolved)
            node["status"] = "EXECUTED"
            if node_type == "plan_validation" and node.get("validation_criteria"):
                node["reasoning"] = f"Validated against: {node['validation_criteria']}"

        else:
            node["status"] = "SKIPPED_UNKNOWN_TYPE"
            node["runtime_output"] = None
            node["reasoning"] = f"No handler for node_type '{node_type}' and no prompt_template."

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


# ============================================================================
# GRAPH NODES
# ============================================================================

def planner_agent_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    raw_history = state.get("chat_history", [])

    parsed_history = parse_raw_conversation_history(raw_history)
    formatted_history = format_history_for_planner_prompt(parsed_history)

    sys_instruction, user_prompt = get_planner_prompts(state, formatted_history)

    is_revision = state.get("status") == "NEEDS_REVISION"
    current_iteration = state.get("iteration_count", 0)
    started = time.time()

    timeline = list(state.get("execution_timeline") or [])

    # Step 1: User input (added only once at graph entry)
    if not timeline:
        timeline.append({
            "step": 1,
            "phase": "USER_INPUT",
            "agent": "User",
            "node_id": "user_query",
            "output_preview": question,
            "status": "RECEIVED",
        })

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config={
            "system_instruction": sys_instruction,
            "response_mime_type": "application/json",
            "response_schema": AutonomousExecutionBlueprint,
        },
    )
    raw_response_text = response.text
    blueprint = json.loads(raw_response_text)
    finished = time.time()

    nodes = blueprint.get("nodes", [])
    prefix = f"rev{current_iteration}_" if is_revision else ""
    for idx, n in enumerate(nodes):
        if not n.get("id") or (is_revision and not n["id"].startswith("rev")):
            n["id"] = f"{prefix}node_{idx + 1}"
        n.setdefault("status", "PENDING")

    blueprint["nodes"] = nodes
    blueprint.setdefault("edges", [])

    # Record Planner execution trace + full blueprint schema
    planner_trace = {
        "step": len(timeline) + 1,
        "phase": f"REPLANNING_ITERATION_{current_iteration}" if is_revision else "INITIAL_PLANNING",
        "agent": "PlannerAgent",
        "node_id": f"planner_rev_{current_iteration}" if is_revision else "planner_initial",
        "status": "EXECUTED",
        "output_preview": f"Generated plan with {len(nodes)} nodes and {len(blueprint.get('edges', []))} edges.",
        "blueprint": blueprint,
        "prompt_used": user_prompt,
        "system_instruction": sys_instruction,
        "runtime_output": raw_response_text,
        "input": {"question": question, "history_turns_considered": len(parsed_history)},
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": round(finished - started, 3),
    }

    timeline.append(planner_trace)

    return {
        "blueprint": blueprint,
        "planner_trace": planner_trace,
        "status": "EXECUTING",
        "execution_timeline": timeline,
    }


def executor_agent_node(state: GraphState) -> Dict[str, Any]:
    blueprint = state.get("blueprint") or {}
    nodes = blueprint.get("nodes", [])
    edges = blueprint.get("edges", [])
    node_outputs = dict(state.get("node_outputs") or {})
    raw_question = state["question"]
    conv_id = state["conv_id"]

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

    executed = set()
    for n_id, n in nodes_by_id.items():
        if n.get("status") == "EXECUTED":
            executed.add(n_id)

    remaining = set(nodes_by_id) - executed

    while remaining:
        ready = [n_id for n_id in remaining if deps[n_id].issubset(executed)]

        if not ready:
            for n_id in remaining:
                nodes_by_id[n_id]["status"] = "SKIPPED_UNRESOLVED_DEPENDENCY"
            break

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

    nodes_ordered = list(nodes_by_id.values())

    synthesis_nodes = [
        n for n in nodes_ordered
        if n.get("status") == "EXECUTED" and n.get("prompt_template") and n.get("node_type") != "plan_validation"
    ]

    if synthesis_nodes:
        final_ans = synthesis_nodes[-1].get("runtime_output", "")
    elif nodes_ordered:
        final_ans = nodes_ordered[-1].get("runtime_output", "")
    else:
        final_ans = ""

    if isinstance(final_ans, list):
        final_ans = str(final_ans)

    timeline = list(state.get("execution_timeline") or [])
    
    # Store complete metadata per DAG node execution
    for n in nodes_ordered:
        out_val = n.get("runtime_output") or n.get("reasoning") or ""
        
        if isinstance(out_val, list):
            preview = f"Retrieved {len(out_val)} RAG chunks."
        else:
            preview = str(out_val)[:200]

        timeline.append({
            "step": len(timeline) + 1,
            "agent": n.get("assigned_agent") or n.get("node_type") or "ExecutorAgent",
            "node_id": n.get("id"),
            "node_type": n.get("node_type"),
            "status": n.get("status", "EXECUTED"),
            "output_preview": preview,
            "runtime_output": n.get("runtime_output"),  # Preserves full RAG search output
            "prompt_template": n.get("prompt_template"),
            "resolved_prompt": n.get("resolved_prompt"),
            "search_query": n.get("search_query"),
            "top_k": n.get("top_k"),
            "reasoning": n.get("reasoning"),
            "duration_seconds": n.get("duration_seconds"),
            "node_data": dict(n),
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

    sys_instruction, user_prompt = get_evaluator_prompts(state, formatted_history)

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
    raw_response_text = response.text
    decision = json.loads(raw_response_text)
    finished = time.time()

    action = decision.get("action", "APPROVE")
    is_sufficient = decision.get("is_sufficient", True)
    reasoning = decision.get("reasoning", "")
    question_to_ask = decision.get("question_to_ask")
    proposed_nodes = decision.get("evaluator_proposed_nodes", [])
    proposed_edges = decision.get("evaluator_proposed_edges", [])

    current_iteration = state.get("iteration_count", 0) + 1

    eval_record = {
        "iteration": current_iteration,
        "action": action,
        "is_sufficient": is_sufficient,
        "reasoning": reasoning,
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
        "output_preview": reasoning,
        "reasoning": reasoning,
        "action": action,
        "is_sufficient": is_sufficient,
        "question_to_ask": question_to_ask,
        "proposed_nodes": proposed_nodes,
        "proposed_edges": proposed_edges,
        "prompt_used": user_prompt,
        "system_instruction": sys_instruction,
        "duration_seconds": round(finished - started, 3),
        "evaluation_record": eval_record,
    })

    if action == "NEEDS_REVISION":
        return {
            "status": "NEEDS_REVISION",
            "revision_reasoning": reasoning,
            "evaluator_proposed_nodes": proposed_nodes,
            "evaluator_proposed_edges": proposed_edges,
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


# ============================================================================
# CONDITIONAL ROUTING & GRAPH COMPILATION
# ============================================================================

def route_after_planner(state: GraphState) -> str:
    return END if state.get("status") == "CLARIFICATION_NEEDED" else "executor"


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
    "planner",
    route_after_planner,
    {"executor": "executor", END: END}
)
builder.add_edge("executor", "evaluator")
builder.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {"planner": "planner", END: END}
)

app_graph = builder.compile()