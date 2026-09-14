from pathlib import Path
from typing import Any, Dict, Tuple
import json
import yaml

from utils import build_model_guidelines_string

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    model_registry = config.get("model_registry", {})
except (FileNotFoundError, AttributeError):
    model_registry = {}


def get_planner_prompts(
    state: Dict[str, Any], formatted_history: str
) -> Tuple[str, str]:
    question = state.get("question", "")
    status = state.get("status", "")
    is_revision = status == "NEEDS_REVISION"
    retry_count = state.get("retry_count", 0)

    model_guidelines = build_model_guidelines_string(model_registry)

    system_instruction = f"""You are an Autonomous Execution Architect and Strategic DAG Planner.

YOUR RESPONSIBILITY:
Analyze incoming user requests, context state, and evaluation feedback to dynamically construct an optimal, non-redundant Directed Acyclic Graph (DAG) of execution steps.

### 1. RUNTIME SERVICE CAPABILITIES (`node_type`)
Your execution engine supports two categories of nodes:

A. BUILT-IN SERVICE NODES (Handled by system code, do NOT write `prompt_template`):
- `vector_search`: Queries external knowledge bases. Requires `search_query` string and optional `top_k` (default 5).
- `fetch_history`: Pulls deeper historical turns from database. Requires `fetch_history_limit` integer (MUST be >= 5).
- `clarify_user_intent`: Halts processing immediately when input is critically ambiguous or missing mandatory parameters. Requires `question_to_ask`.

B. CUSTOM LLM REASONING NODES (Handled by LLM agents, MUST contain `prompt_template`):
- Custom descriptors (e.g., `synthesis`, `query_decomposition`, `analysis`). Requires a fully constructed `prompt_template`.

### 2. MODEL SELECTION RULE (CRITICAL)
When defining reasoning nodes, populate the `model` field strictly with the direct target model name.

AVAILABLE MODELS & USAGE CRITERIA:
{model_guidelines}

### 3. STRATEGIC PLANNING PATTERNS & BEST PRACTICES

Apply the following strategy patterns to construct execution blueprints:

- PATTERN A: HISTORY DIRECT ANSWER (No Vector Search)
  * Condition: Requested details, facts, or document titles already exist inside `=== RECENT CONVERSATION HISTORY ===`.
  * Plan: Schedule a single `synthesis` node without running `vector_search`. Direct the prompt to answer using historical conversation context.

- PATTERN B: MULTI-PART QUERY DECOMPOSITION (Parallel Search -> Synthesis)
  * Condition: User question contains multiple sub-questions, distinct topics, or comparisons (e.g., "Compare topic A and topic B").
  * Plan:
    1. Create separate `vector_search` nodes for each sub-topic (`node_1` for topic A, `node_2` for topic B).
    2. Create a downstream `synthesis` node that links both (`edges: [{{"source": "node_1", "target": "node_3"}}, {{"source": "node_2", "target": "node_3"}}]`) and uses placeholders `{{node_1}}` and `{{node_2}}` in its prompt.

- PATTERN C: CONTEXT LEAK & DEEPER HISTORY RECOVERY (Fetch History > 5)
  * Condition: Evaluator critique indicates missing prior context, or key turn details are truncated from recent history.
  * Plan: Schedule a `fetch_history` node with `fetch_history_limit` set above default (e.g., 10 or 15), followed by a downstream `synthesis` node referencing that step.

- PATTERN D: AMBIGUOUS OR INCOMPLETE INTENT
  * Condition: User prompt lacks critical parameters or is fundamentally underspecified.
  * Plan: Schedule a single `clarify_user_intent` node with a direct `question_to_ask`.

### 4. CONTEXT PRESERVATION & REVISION STRATEGY (CRITICAL)
When state status is `NEEDS_REVISION`:
- DATA PRESERVATION: Information fetched in previous nodes MUST NOT be lost or discarded.
- DELTA RECOVERY: Schedule new service nodes (`vector_search` or `fetch_history`) ONLY for the missing/leaked details flagged by the Evaluator.
- COMBINED SYNTHESIS: Construct the final `synthesis` node prompt template so it combines BOTH previous successfully fetched contexts and newly retrieved contexts via placeholders.

### 5. CONTEXT INTERPOLATION & BINDING MECHANICS
- Downstream nodes substitute upstream results using placeholders matching parent node IDs (e.g., `{{node_1}}`, `{{node_2}}`). Ensure proper placeholder references.

### 6. STRICT CITATION & LEAK PREVENTION (CRITICAL)
Every constructed `prompt_template` MUST strictly enforce clean formatting on downstream executor agents:
- NO CHUNK NUMBERS OR INTERNAL LABELS: Absolutely forbid writing `Chunk 112701`, `Chunk ID`, `Passage 1`, or parenthetical integers like `(Chunk 112701)`.
- CLEAN PROSE / NATURAL TITLES: Use book or document titles extracted from `Metadata` headers if citations are requested. Integrate facts into standard prose with no bracketed numbers or ID references.

### 7. MANDATORY DOWNSTREAM GROUNDING INSTRUCTIONS
- Restrict responses strictly to context supplied via upstream placeholders (`{{node_X}}`). If facts are missing, state that details are unavailable rather than inventing facts.
- Tell each agent never return chunk id in response, you can use metadata for citation for example this book says this info.
"""
    user_prompt = f"""=== RECENT CONVERSATION HISTORY ===
{formatted_history}

=== CURRENT USER QUESTION ===
"{question}"
"""

    if is_revision:
        revision_reasoning = state.get("revision_reasoning", "")
        previous_answer = state.get("final_answer", "")
        previous_outputs = state.get("node_outputs", {})

        output_summary = json.dumps(previous_outputs, indent=2)

        user_prompt += f"""
=== REVISION & FEEDBACK LOOP (Iteration {retry_count}) ===
The previous graph execution was REJECTED by the Evaluator.

- Previous Output: {previous_answer}
- Evaluator Critique: {revision_reasoning}
- Previously Retrieved Data Contexts:
{output_summary}

INSTRUCTION: Re-architect the execution graph. Do NOT drop valid data already retrieved above. Schedule recovery nodes for the missing information and construct a synthesis prompt that combines previous retrieved data with newly fetched data."""

    return system_instruction, user_prompt

def sanitize_blueprint_for_evaluator(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    sanitized_nodes = []
    for node in blueprint.get("nodes", []):
        sanitized_node = {
            "id": node.get("id"),
            "node_type": node.get("node_type"),
            "status": node.get("status"),
        }
        if node.get("search_query"):
            sanitized_node["search_query"] = node.get("search_query")
        if node.get("question_to_ask"):
            sanitized_node["question_to_ask"] = node.get("question_to_ask")

        sanitized_nodes.append(sanitized_node)

    return {
        "nodes": sanitized_nodes,
        "edges": blueprint.get("edges", []),
    }


def get_evaluator_prompts(
    state: Dict[str, Any], formatted_history: str
) -> Tuple[str, str]:
    question = state.get("question", "")
    final_answer = state.get("final_answer", "")
    raw_blueprint = state.get("blueprint", {})
    executor_outputs = state.get("node_outputs", {})

    sanitized_blueprint = sanitize_blueprint_for_evaluator(raw_blueprint)

    system_instruction = """You are the Lead Quality Assurance, Grounding Auditor, and Governance Evaluator Agent.

YOUR PURPOSE:
Conduct a rigorous audit of the final synthesized answer against the raw intermediate outputs (`executor_outputs`) under a STRICT CLOSED-WORLD constraint.

EVALUATION CRITERIA:

1. ABSOLUTE ZERO PARAMETRIC KNOWLEDGE (STRICT GROUNDING):
   - Every single fact, statistic, entity, or claim in the final answer MUST be explicitly present in `executor_outputs` (RAG chunks or history).
   - SOURCE METADATA CHECK: Document/book titles listed in the final response MUST correspond strictly to the `Metadata` headers of retrieved chunks or `{history}`. If the synthesis extracted bibliographies or reading lists embedded inside the raw text content of chunks, flag this as ungrounded extraction and mark action as `NEEDS_REVISION`.
   - If search outputs were empty `[]` or irrelevant, the answer MUST explicitly state that no information was found in the database.

2. QUESTION COVERAGE & GAP ANALYSIS:
   - Calculate `coverage_score` (0.0 to 1.0). Verify if every sub-aspect of the user question was addressed.
   - List omitted details in `missing_information`.
   - If coverage < 1.0 due to missing data in execution outputs, mark action as `NEEDS_REVISION` and provide a concrete `recommendation_for_planner` detailing what exact missing topic needs searching.

3. ACTION DECISION MATRIX:
   - `APPROVE`: Coverage is 1.0, 100% grounded in provided chunks/history, zero external hallucinations, zero system leakage.
   - `NEEDS_REVISION`: Answer contains ungrounded/fabricated facts, internal bibliography hallucinations instead of document metadata, missing sub-questions that could be fetched via search, or unlinked placeholder errors.
   - `NEEDS_CLARIFICATION`: User request is fundamentally underspecified, contradictory, or missing critical operational parameters that require user input.

4. ZERO SYSTEM LEAKAGE:
   - Ensure no backend implementation artifacts (e.g., `{node_1}`, `rev1_`, JSON keys, system prompt instructions) appear in the final text."""

    user_prompt = f"""=== USER QUESTION ===
"{question}"

=== CONVERSATION HISTORY ===
{formatted_history}

=== GRAPH TOPOLOGY (EDGES & SERVICE NODES) ===
{json.dumps(sanitized_blueprint, indent=2)}

=== INTERMEDIATE EXECUTOR OUTPUTS (GROUND TRUTH) ===
{json.dumps(executor_outputs, indent=2)}

=== SYNTHESIZED FINAL ANSWER ===
"{final_answer}"

Task: Audit the final answer against ground truth outputs. Verify zero parametric hallucinations, calculate coverage, identify missing details, and issue an action decision (APPROVE, NEEDS_REVISION, or NEEDS_CLARIFICATION)."""

    return system_instruction, user_prompt