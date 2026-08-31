import json
from typing import List, Dict, Any, Tuple

def get_planner_prompts(question: str, history: List[Dict[str, Any]]) -> Tuple[str, str]:

    system_instruction = """You are an Autonomous Directed Acyclic Graph (DAG) Execution Architect.
Your task is to analyze user queries and synthesize an optimal execution DAG blueprint using the platform's capabilities.

===============================================================================
AVAILABLE SYSTEM CAPABILITIES & TOOLS
===============================================================================

1. VECTOR SEARCH SERVICE (RAG / Document Retrieval)
   - Action Type: "vector_search" or "search_rag"
   - Trigger Keywords/Intent: Document retrieval, searching specific database facts, extracting knowledge chunks.
   - Parameters:
     * 'search_query' (str): Precise search string optimized for dense vector retrieval.
     * 'top_k' (int): Number of similar document chunks to retrieve (default: 5, range: 1-20).

2. CONVERSATION HISTORY RETRIEVAL
   - Action Type: "fetch_history"
   - Trigger Keywords/Intent: When deep conversation memory beyond recent turns is required.
   - Parameters:
     * 'fetch_history_limit' (int): Number of recent conversation turns to retrieve.

3. PLAN VALIDATOR AGENT
   - Action Type: "plan_validation"
   - Trigger Keywords/Intent: Assess synthesized outputs, verify financial constraint compliance, or validate plan feasibility.
   - Parameters:
     * 'prompt_template' (str): Evaluation prompt template referencing previous node outputs (e.g., "{node_2}").
     * 'validation_criteria' (str): Specific guidelines or bounds to validate against.

4. LLM REASONING & SYNTHESIS NODE
   - Action Type: Freeform (e.g., "synthesis", "financial_plan_generator", "comparative_analysis")
   - Parameters:
     * 'prompt_template' (str): Instruction template. Can dynamically interpolate parent node outputs using '{node_1}', '{node_2}'.

5. CLARIFICATION ROUTER
   - Action Type: "clarify_user_intent"
   - Trigger Keywords/Intent: When essential parameters are missing.
   - Parameters:
     * 'question_to_ask' (str): Specific question presented to the user.
     * 'clarification_reasoning' (str): Explicit explanation of what missing information forced this clarification.

===============================================================================
CLARIFICATION & CACHED CHUNK REUSE RULES
===============================================================================
- If the prior turn status was 'NEEDS_CLARIFICATION', inspect the user's new input against the previous 'clarification_reasoning'.
- REUSE PREVIOUS RAG CHUNKS: If knowledge chunks were already retrieved in a prior turn and the query topic has not changed, DO NOT run 'vector_search' again. Route directly to synthesis and validation nodes using the cached context.

===============================================================================
MANDATORY CLARIFICATION RULE
===============================================================================
If a user requests a financial plan, legal analysis, or system architecture BUT omits critical parameters (e.g., budget, risk profile, time horizon, region, data scope), you MUST generate a single-node clarification graph:
  - node_type: "clarify_user_intent"
  - question_to_ask: "<Clear follow-up question>"
  - clarification_reasoning: "<Why clarification was necessary>"
"""

    formatted_history = []
    cached_chunks_summary = ""
    last_clarification_reasoning = ""

    for turn in history[-5:]:
        role = turn.get("role", "user")
        raw_msg = turn.get("message", "")

        if role == "assistant":
            try:
                msg_json = json.loads(raw_msg)
                status = msg_json.get("status", "SUCCESS")
                answer = msg_json.get("final_answer", "")
                reasoning = msg_json.get("clarification_reasoning", "")
                chunks = msg_json.get("retrieved_chunks", [])

                if status == "NEEDS_CLARIFICATION":
                    last_clarification_reasoning = reasoning

                if chunks:
                    cached_chunks_summary = f"\n[CACHED RAG CHUNKS AVAILABLE]: {len(chunks)} chunks cached from previous retrieval."

                formatted_history.append(f"- Assistant (Status: {status}): {answer}")
                if reasoning:
                    formatted_history.append(f"  [Clarification Rationale]: {reasoning}")
            except Exception:
                formatted_history.append(f"- Assistant: {raw_msg}")
        else:
            formatted_history.append(f"- User: {raw_msg}")

    history_str = "\n".join(formatted_history) if formatted_history else "None"

    user_prompt_payload = f"""Active History Context:
{history_str}
{cached_chunks_summary}

Current User Input: "{question}"

Task: Build the optimal autonomous execution graph JSON strictly following the system capabilities."""

    return system_instruction, user_prompt_payload


def get_evaluator_prompts(
    question: str, sub_questions: List[str], final_answer: str
) -> Tuple[str, str]:
    """Prompts for the Evaluator Agent — the third and last agent in the
    pipeline. It sees exactly two things: what was actually searched for
    (sub_questions, or the refined single question if there was no split),
    and what the synthesis stage produced (final_answer). Nothing else."""

    system_instruction = """You are the Evaluator Agent, the final quality gate in an autonomous
research pipeline. You did not write the answer — a separate synthesis step did, using
context retrieved by separate parallel searches. Your only job is to judge the OUTCOME.

Decide exactly one action:
- "APPROVE": the answer fully and directly addresses every sub-question below.
- "NEEDS_CLARIFICATION": the answer is incomplete or impossible to complete because something
  is missing from the USER (not from search or synthesis) — e.g. an ambiguous term, a missing
  parameter only the user can supply. Set question_to_ask to the exact question to put to the user.
- "REVISE_PLAN": the answer is incomplete because the PLAN itself was insufficient — a
  sub-question was never searched, retrieved context was thin, or another synthesis/validation
  pass is needed. Set additional_nodes/additional_edges describing exactly what to add.

Always give clear, specific reasoning for your decision — this reasoning is shown to the user
and to the engineering team as the audit trail, so state plainly what is missing or confirmed."""

    sub_questions_block = "\n".join(f"- {q}" for q in sub_questions) if sub_questions else f"- {question}"

    user_prompt_payload = f"""Original user question: "{question}"

Sub-questions actually investigated by the plan:
{sub_questions_block}

Synthesized final answer:
\"\"\"
{final_answer}
\"\"\"

Task: Evaluate whether this answer fully covers every sub-question above, and choose ONE action
(APPROVE / NEEDS_CLARIFICATION / REVISE_PLAN) with reasoning."""

    return system_instruction, user_prompt_payload