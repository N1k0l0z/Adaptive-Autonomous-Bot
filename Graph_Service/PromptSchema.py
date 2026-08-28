from typing import List, Dict, Any, Tuple

def get_planner_prompts(question: str, history: List[Dict[str, Any]]) -> Tuple[str, str]:

    system_instruction = """You are a High-Speed Autonomous Execution Graph Architect.
Design a minimal Directed Acyclic Graph (DAG) for the user request.

### STRICT LATENCY & SPEED RULES (CRITICAL)
1. MAXIMUM 15 WORDS PER STRING: Keep all 'prompt_template', 'question_to_ask', and 'search_query' extremely brief.
2. NO VERBOSE EXPLANATIONS: Generate zero structural explanations.
3. ATOMIC SEARCH QUERIES: One 3-6 word search string per SEARCH_RAG node. Parallelize multi-topic requests across root nodes.

### NODE RULES
- SEARCH_RAG: Set 'search_query'. Other optional fields must be null.
- SYNTHESIZE: Set 'prompt_template' referencing parent variables like '{node_1}'. Other optional fields must be null.
- CLARIFY / USER_INPUT: Set 'question_to_ask'. Other optional fields must be null.
- DECOMPOSE_QUERY: Set 'prompt_template'. Only use if sub-queries are non-deterministic.

### GRAPH RULES
- Always flow forward (DAG). No circular loops.
- Return single CLARIFY node if the user query is completely ambiguous.
"""

    if history:
        # Keep history string minimal to conserve prompt tokens
        history_str = "\n".join([f"- {msg.get('role', 'user')}: {msg.get('message', '')}" for msg in history[-3:]])
    else:
        history_str = "None"

    user_prompt_payload = f"""History:
{history_str}

User Question: "{question}"

Task: Build execution DAG JSON."""

    return system_instruction, user_prompt_payload