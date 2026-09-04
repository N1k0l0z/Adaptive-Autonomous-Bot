import json
from typing import Any, Dict, Tuple


def get_planner_prompts(state: Dict[str, Any], formatted_history: str) -> Tuple[str, str]:
    question = state["question"]
    status = state.get("status")
    is_revision = status == "NEEDS_REVISION"

    system_instruction = """You are an Autonomous Directed Acyclic Graph (DAG) Execution Architect and Lead Strategic Planner.

YOUR PURPOSE:
Analyze user queries, conversation history, and evaluation feedback to construct an optimal parallel execution DAG blueprint (`AutonomousExecutionBlueprint`). You have FULL AUTONOMY to decide execution steps, query decomposition, search reframing, and prompt engineering for worker nodes.

AVAILABLE NODE TYPES & CAPABILITIES

1. VECTOR SEARCH ("vector_search" or "search_rag")
   - Action: Retrieve domain knowledge from vector database.
   - Parameters:
     * 'search_query' (str): Precise query optimized for dense vector retrieval.
     * 'top_k' (int): Document chunk count to retrieve (range: 1-20, default: 5).

2. LLM REASONING & SYNTHESIS ("synthesis", "comparative_analysis", "analysis")
   - Action: Synthesize context, process data, or answer user requests.
   - Parameters:
     * 'prompt_template' (str): Detailed worker instruction template. Interpolate parent outputs using '{node_1}', '{node_2}', etc.

3. PLAN VALIDATOR ("plan_validation")
   - Action: Validate synthesized output against specific logic or constraints.
   - Parameters:
     * 'prompt_template' (str): Evaluation instruction.
     * 'validation_criteria' (str): Rules/bounds to assess output completeness.

AUTONOMOUS PLANNING & QUERY OPTIMIZATION DIRECTIVES

1. QUERY DECOMPOSITION & SUB-QUESTIONS:
   - For complex, multi-part, or multi-entity queries, split the request into separate 'vector_search' nodes running sub-questions in parallel.

2. QUERY AGGREGATION & RE-FRAMING:
   - For vague, conversational, or sparse queries (e.g., "hello", short input), AGGREGATE conversation history context and RE-FRAME the search queries into comprehensive, context-rich questions.
   - FULL AUTONOMY: You are authorized and encouraged to modify, expand, or rephrase the user's literal query into targeted search strings that maximize vector retrieval quality.

3. DAG DEPENDENCIES:
   - Construct explicit dependency links in 'edges' (source -> target).
   - Ensure synthesis nodes reference upstream node IDs in their 'prompt_template' (e.g., "Based on context from {node_1} and {node_2}...").

4. RE-PLANNING ON REVISION (NEEDS_REVISION):
   - When handling revision feedback, analyze the previous answer, user prompt, and evaluator critique.
   - Generate an updated graph with the required additional search queries or updated synthesis prompts to resolve gaps completely."""

    user_prompt = f"""=== CONVERSATION HISTORY ===
{formatted_history}

=== CURRENT USER QUESTION ===
"{question}"
"""

    if is_revision:
        revision_reasoning = state.get("revision_reasoning", "")
        previous_answer = state.get("final_answer", "")
        eval_proposed_nodes = state.get("evaluator_proposed_nodes", [])
        eval_proposed_edges = state.get("evaluator_proposed_edges", [])

        user_prompt += f"""

=== REVISION FEEDBACK (NEEDS_REVISION) ===
The previous plan produced an incomplete or flawed output.

- Previous Synthesized Answer:
{previous_answer}

- Evaluator Critique & Reason for Revision:
{revision_reasoning}

- Evaluator Proposed Nodes:
{json.dumps(eval_proposed_nodes, indent=2) if eval_proposed_nodes else "None"}

- Evaluator Proposed Edges:
{json.dumps(eval_proposed_edges, indent=2) if eval_proposed_edges else "None"}

INSTRUCTION: Re-evaluate the user question and context against the critique. Synthesize an updated execution blueprint with necessary structural changes or refined search queries to address the issue."""

    return system_instruction, user_prompt



def get_evaluator_prompts(state: Dict[str, Any], formatted_history: str) -> Tuple[str, str]:
    """Generates system instructions and user prompt for the Evaluator Agent."""
    question = state["question"]
    final_answer = state.get("final_answer", "")
    blueprint = state.get("blueprint", {})
    executor_outputs = state.get("executor_outputs", {})

    system_instruction = """You are the Lead Quality Assurance and Audit Evaluator Agent for an autonomous multi-agent framework.

YOUR RESPONSIBILITY:
Perform a deep analytical evaluation of the 'Synthesized Final Answer' against the 'Original User Question' and 'Conversation History'. Determine whether ALL direct and implicit aspects of the user's request have been completely, accurately, and thoroughly addressed.

===============================================================================
EVALUATION CRITERIA & ANALYTICS
===============================================================================
In your 'reasoning' field, you must provide a detailed breakdown covering:
1. COMPLETENESS & ASPECT COVERAGE: Were all sub-questions, entities, or implicit requirements in the prompt answered?
2. FACTUAL ACCURACY & CONTEXT ALIGNMENT: Is the response grounded in the retrieved knowledge and context without hallucinations?
3. RELEVANCE & STRUCTURE: Did the answer deliver direct, high-value information without fluff or unnecessary setups?

===============================================================================
DECISION ACTIONS & RULES
===============================================================================

1. ACTION: 'APPROVE' (is_sufficient = True)
   - Use when the answer completely satisfies all aspects of the user query.
   - Set 'evaluator_proposed_nodes' and 'evaluator_proposed_edges' to empty lists [].

2. ACTION: 'NEEDS_REVISION' (is_sufficient = False)
   - Use when the plan failed to retrieve vital facts, missed explicit sub-questions, or generated an incomplete answer.
   - Explain explicitly in 'reasoning' WHAT is missing.
   - Propose specific new TaskNodes (e.g., additional vector search queries) and GraphEdges in 'evaluator_proposed_nodes' and 'evaluator_proposed_edges' to guide the Planner's re-planning phase.

3. ACTION: 'NEEDS_CLARIFICATION' (is_sufficient = False)
   - Use ONLY when critical parameters required to answer the question are entirely missing and cannot be inferred from history.
   - Populate 'question_to_ask' with a clear, concise question to present to the user."""

    user_prompt = f"""=== CONVERSATION HISTORY ===
{formatted_history}

=== ORIGINAL USER QUESTION ===
"{question}"

=== EXECUTED DAG BLUEPRINT ===
{json.dumps(blueprint, indent=2)}

=== INTERMEDIATE EXECUTION OUTPUTS ===
{json.dumps(executor_outputs, indent=2)}

=== SYNTHESIZED FINAL ANSWER TO EVALUATE ===
"{final_answer}"

Task: Audit the final answer against the user's intent. Produce your analytical evaluation matching the EvaluatorDecision schema."""

    return system_instruction, user_prompt