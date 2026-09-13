import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import requests

# 1. Resolve exact project paths
CURRENT_DIR = Path(__file__).resolve().parent
GRAPH_SERVICE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = GRAPH_SERVICE_DIR.parent

# 2. Load GCP key & environment variables FIRST
KEY_PATH = PROJECT_ROOT / "gcp-key.json"
if KEY_PATH.exists():
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        key_data = json.load(f)
        project_id = key_data.get("project_id")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get(
    "GOOGLE_CLOUD_LOCATION", "us-central1"
)

# 3. Add path and import config AFTER credentials are set
if str(GRAPH_SERVICE_DIR) not in sys.path:
    sys.path.append(str(GRAPH_SERVICE_DIR))

from app.config import MODEL_NAME, client

DB_SERVICE_URL = os.environ.get("DB_SERVICE_URL", "http://127.0.0.1:8000")
EMBEDDING_SERVICE_URL = os.environ.get(
    "EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001"
)
QUERY_REWRITE_PROMPT = """You are an expert search query reformulator.
Given the conversation history and a follow-up user question, rewrite the follow-up question into a single, complete, and fully self-contained search query optimized for vector retrieval.

=== CONVERSATION HISTORY ===
{chat_history}

=== FOLLOW-UP QUESTION ===
{question}

=== STANDALONE QUERY ===
"""

SYNTHESIS_PROMPT = """You are a senior domain researcher tasked with producing exhaustive, precise, and fully grounded technical analyses.

=== STRICT GROUNDING RULES ===
1. ZERO OUTSIDE KNOWLEDGE: You must rely EXCLUSIVELY on the provided retrieved context chunks. Do NOT use any external parametric memory, general knowledge, unstated assumptions, or outside facts under any circumstances.
2. ABSOLUTE CONTEXT ADHERENCE: Every statement, formula, metric, and claim you make must be directly backed by the retrieved context block.

=== EXTRACTION & SYNTHESIS DIRECTIVES ===
1. DO NOT STEP BACK OR ARTIFICIALLY TRUNCATE: Do not write brief, generic, or high-level summaries. Extract and synthesize every single relevant theoretical concept, mathematical variable, empirical finding, and structural mechanism contained in the chunks.
2. MAXIMIZE TECHNICAL DEPTH: Push the depth of your synthesis to the absolute limit supported by the context. Thoroughly exhaust all details available in the chunks without holding back depth or structure.
3. PARTIAL COVERAGE HANDLING: If the context answers only part of the question, exhaustively answer every supported part using the context. Clearly and explicitly state which specific sub-points are missing from the retrieved context rather than issuing a total refusal.

=== RETRIEVED CONTEXT ===
{context_block}

=== QUESTION ===
{question}

=== ANSWER ===
"""


def fetch_conversation_history_http(
    conv_id: str, limit: int = 5
) -> List[Dict[str, str]]:
    try:
        url = f"{DB_SERVICE_URL}/history/{conv_id}?limit={limit}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def perform_vector_search_http(
    query: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    try:
        embed_res = requests.post(
            f"{EMBEDDING_SERVICE_URL}/embed", json={"text": query}, timeout=5
        )
        if embed_res.status_code != 200:
            return []
        vector = embed_res.json().get("embeddings", [[]])[0]
        if not vector:
            return []

        search_res = requests.post(
            f"{DB_SERVICE_URL}/rag/search",
            json={"query_embedding": vector, "top_k": top_k, "min_sim": 0.0},
            timeout=5,
        )
        if search_res.status_code == 200:
            return search_res.json()
    except Exception:
        pass
    return []


def format_history(history_list: List[Dict[str, str]]) -> str:
    if not history_list:
        return "No prior conversation history."
    formatted = []
    for msg in history_list:
        if isinstance(msg, dict):
            formatted.append(
                f"{msg.get('role', 'User')}: {msg.get('content', '')}"
            )
        else:
            formatted.append(str(msg))
    return "\n".join(formatted)


def run_standard_rag(
    question: str, conv_id: str = "default_conv"
) -> Dict[str, Any]:
    start_time = time.time()
    history = fetch_conversation_history_http(conv_id=conv_id, limit=5)
    formatted_hist = format_history(history)
    rewrite_input = QUERY_REWRITE_PROMPT.format(
        chat_history=formatted_hist, question=question
    )

    try:
        standalone_query = (
            client.models.generate_content(
                model=MODEL_NAME, contents=rewrite_input
            )
            .text.strip()
            .replace('"', "")
        )
    except Exception:
        standalone_query = question

    retrieved_chunks = perform_vector_search_http(
        query=standalone_query, top_k=5
    )

    chunk_contents = []
    for idx, chunk in enumerate(retrieved_chunks):
        content = chunk.get("content", str(chunk))
        chunk_contents.append(f"[Chunk {idx+1}]: {content}")

    context_block = (
        "\n\n".join(chunk_contents)
        if chunk_contents
        else "No external context found."
    )
    synth_input = SYNTHESIS_PROMPT.format(
        context_block=context_block, question=question
    )

    try:
        final_answer = client.models.generate_content(
            model=MODEL_NAME, contents=synth_input
        ).text.strip()
    except Exception as e:
        final_answer = f"Error during synthesis: {str(e)}"

    return {
        "conv_id": conv_id,
        "question": question,
        "std_latency_sec": round(time.time() - start_time, 3),
        "std_final_answer": final_answer,
    }


def main():
    benchmark_questions = [
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_01",
        "question": "Using the Mundell-Fleming model, analyze how high capital mobility constrains monetary policy autonomy under a pegged exchange rate regime during an external interest rate shock, and evaluate the trade-offs between reserve depletion and imposing capital controls."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_02",
        "question": "Analyze how financial dollarization impairs the interest rate channel of central bank monetary policy transmission, specifically tracing how policy rate adjustments fail to pass through to commercial lending rates during sudden-stop foreign exchange shocks."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_03",
        "question": "Deconstruct the structural channels of 'Dutch Disease' in resource-exporting emerging economies, examining how real exchange rate appreciation squeezes non-resource tradable sectors and alters long-run productivity."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_04",
        "question": "Evaluate the mathematical formulation and structural limitations of the standard Taylor Rule when applied to small open economies subject to terms-of-trade volatility and supply-side energy shocks."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_05",
        "question": "Examine the mechanism of Lucas Critique in economic policy evaluation, specifically demonstrating how private sector rational expectations defeat central bank attempts to exploit short-run Phillips Curve trade-offs."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_06",
        "question": "Analyze liquidity trap conditions at the Zero Lower Bound (ZLB), assessing how quantitative easing (QE) and forward guidance alter term premiums and long-term sovereign bond yields."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_07",
        "question": "Explain how unanchored inflation expectations shift the Aggregate Supply (AS) curve upward, and analyze central bank credibility requirements for executing a low-sacrifice-ratio disinflation program."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_08",
        "question": "Assess the disintermediation risks to commercial bank balance sheets when a central bank introduces a direct-to-consumer Retail Central Bank Digital Currency (CBDC) during times of systemic banking sector stress."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_09",
        "question": "Evaluate the transmission of central bank quantitative tightening (QT) on interbank money market liquidity, reverse repo utility, and short-term yield volatility."
    },
    {
        "category": "Macroeconomic Transmission & Monetary Policy",
        "conv_id": "econ_macro_10",
        "question": "Analyze the structural determinants of the Natural Rate of Interest (r*) and evaluate how demographic aging and lower Total Factor Productivity (TFP) constrain central bank terminal policy rates."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_11",
        "question": "Apply Jacob Viner's customs union theory to evaluate trade creation versus trade diversion effects for South Caucasus economies participating in regional free trade agreements along the Middle Corridor."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_12",
        "question": "Compare the logistics bottlenecks, customs clearance latency, infrastructure investment budgets, and container transit times across the Trans-Caspian route versus traditional Northern Corridor rail transit."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_13",
        "question": "Analyze the Balassa-Samuelson effect, showing mathematically how productivity growth differentials between tradable and non-tradable sectors lead to domestic real exchange rate appreciation."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_14",
        "question": "Evaluate the gravity model of international trade when incorporating multilateral resistance terms, estimating how trade friction elasticities change under sudden sanctions regimes."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_15",
        "question": "Examine the Marshall-Lerner condition and the J-curve effect following a currency devaluation, outlining the temporal lags required for trade volume adjustments to outweigh price effects."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_16",
        "question": "Assess Global Value Chain (GVC) integration metrics (forward vs. backward participation rates) in middle-income economies and analyze how rising supply chain regionalization impacts foreign direct investment (FDI)."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_17",
        "question": "Analyze the economic impact of Carbon Border Adjustment Mechanisms (CBAM) on heavy-industry exporters in developing nations lacking domestic carbon pricing markets."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_18",
        "question": "Examine the port infrastructure capacity, berth productivity metrics, and rail intermodal connections comparing Poti and Batumi ports in Georgia against Baku International Sea Trade Port in Azerbaijan."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_19",
        "question": "Evaluate nearshoring and friendshoring shifts in global trade routes, identifying how trade redirection impacts transit tariff revenues for intermediate transit corridors."
    },
    {
        "category": "International Trade & Geoeconomics",
        "conv_id": "econ_trade_20",
        "question": "Analyze terms of trade (ToT) shocks in commodity-dependent developing economies and evaluate how fiscal rules buffer domestic consumption against severe revenue volatility."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_21",
        "question": "Analyze the 'Original Sin' hypothesis in emerging market sovereign debt, evaluating how currency mismatches exacerbate fiscal sustainability risks and sovereign credit rating downgrades during balance of payments crises."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_22",
        "question": "Examine the sovereign-bank 'doom loop' under Basel III capital requirements, analyzing how domestic bank holdings of local sovereign debt amplify systemic risk during sudden fiscal deteriorations."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_23",
        "question": "Assess how macroprudential tools—such as reserve requirements on foreign currency liabilities and countercyclical capital buffers (CCyB)—mitigate systemic credit risks in heavily dollarized banking sectors."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_24",
        "question": "Apply Guillermo Calvo's model of sudden stops in capital flows to examine how real exchange rate depreciation forces sharp current account reversals and domestic balance sheet compression."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_25",
        "question": "Evaluate Debt Sustainability Analysis (DSA) frameworks under stochastic interest-rate-growth differentials (r - g) and primary fiscal deficit trajectories in non-resource emerging markets."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_26",
        "question": "Analyze systemic financial contagion using CoVaR and SRISK methodologies, demonstrating how capital buffer deficiencies in tier-one banks propagate systemic liquidity risk."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_27",
        "question": "Evaluate the resolution strategies for Non-Performing Loans (NPLs) during post-crisis credit contractions, contrasting centralized Asset Management Companies (AMCs) against out-of-court debt restructuring frameworks."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_28",
        "question": "Analyze the impact of interest coverage ratio (ICR) decay across corporate sectors on banking sector capitalization under severe macro-stress testing scenarios."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_29",
        "question": "Assess the effectiveness of IMF reserve adequacy (ARA metric) indicators in predicting currency crisis vulnerabilities across emerging market economies."
    },
    {
        "category": "Sovereign Debt & Financial Stability",
        "conv_id": "econ_fin_30",
        "question": "Examine shadow banking expansion and money market fund liquidity vulnerabilities when central bank emergency lending facilities (Lender of Last Resort) are legally restricted to commercial banks."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_31",
        "question": "Compare the monetary policy frameworks, inflation targeting performance, foreign reserve adequacy metrics (ARA metric), and exchange rate intervention strategies of Georgia, Armenia, and Azerbaijan over recent external shock periods."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_32",
        "question": "Evaluate the comparative impact of migrant remittances versus foreign direct investment (FDI) inflows on current account balances and domestic currency real appreciation across South Caucasus economies."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_33",
        "question": "Compare the GDP growth rates, domestic inflation rates, public debt ratios, and primary fiscal balances of Georgia, Armenia, and Azerbaijan over recent fiscal cycles."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_34",
        "question": "Contrast the structural governance, asset allocation strategies, and fiscal rule withdrawal mechanisms of sovereign wealth funds in oil-exporting states (e.g., SOFAZ in Azerbaijan) against non-resource stabilization funds."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_35",
        "question": "Compare banking sector financial soundness metrics—specifically Loan-to-Deposit (LTD) ratios, NPL ratios, capital adequacy ratios (CAR), and FX loan shares—between Georgia and Armenia."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_36",
        "question": "Evaluate trade tax revenue share, value-added tax (VAT) efficiency ratios, and corporate income tax regime structures across South Caucasus economies."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_37",
        "question": "Compare energy matrix composition, electricity generation capacities, and energy import dependency metrics between Georgia, Armenia, and Azerbaijan."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_38",
        "question": "Analyze comparative labor market dynamics across the South Caucasus, comparing labor force participation rates, structural youth unemployment, and worker remittance intensity."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_39",
        "question": "Contrast institutional mandates and structural reform progress under EU Association Agreements versus regional customs unions across South Caucasus economies."
    },
    {
        "category": "Multi-Entity Comparative Analysis",
        "conv_id": "econ_comp_40",
        "question": "Compare the transmission of regional food and energy price shocks to consumer price indices (CPI) across Georgia, Armenia, and Azerbaijan based on consumption basket weightings."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_41",
        "question": "Formulate a Structural Vector Autoregression (SVAR) model using Sign Restrictions to disentangle aggregate demand, aggregate supply, and monetary policy shocks in a small open economy."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_42",
        "question": "Explain the implementation of a Difference-in-Differences (DiD) design with staggered treatment timing, detailing how parallel trends violations distort causal estimates of minimum wage hikes."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_43",
        "question": "Deconstruct a Dynamic Stochastic General Equilibrium (DSGE) model featuring Calvo price stickiness, demonstrating how shock persistence parameters alter the impulse response functions of consumption and investment."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_44",
        "question": "Analyze the conditions required for valid Instrumental Variables (IV) estimation in two-stage least squares (2SLS), detailing weak instrument diagnostics (Cragg-Donald statistic) when estimating returns to education."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_45",
        "question": "Evaluate Johansen Cointegration tests and Vector Error Correction Models (VECM) to estimate long-run Purchasing Power Parity (PPP) and speed-of-adjustment parameters."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_46",
        "question": "Detail a Regression Discontinuity Design (RDD) framework (sharp vs. fuzzy) used to evaluate microfinance access limits based on asset eligibility thresholds."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_47",
        "question": "Compare Solow-Swan growth accounting methods against endogenous growth models (Romer model) when calculating Total Factor Productivity (TFP) growth in transitioning developing nations."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_48",
        "question": "Analyze real options valuation frameworks for major infrastructure investments under price uncertainty, contrasting Net Present Value (NPV) decisions against deferral option values."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_49",
        "question": "Evaluate GARCH and EGARCH econometric models in modeling asymmetric conditional volatility in emerging market foreign exchange rates during political shocks."
    },
    {
        "category": "Applied Econometrics & Structural Modeling",
        "conv_id": "econ_em_50",
        "question": "Analyze Synthetic Control Methods (SCM) versus conventional panel fixed-effects models when evaluating the macro-level impact of major trade embargoes on sovereign growth."
    }
]

    fieldnames = [
        "category",
        "conv_id",
        "question",
        "std_latency_sec",
        "std_final_answer",
    ]

    output_file = CURRENT_DIR / "standard_rag_result.csv"

    print(f"Starting Benchmark Evaluation using {MODEL_NAME}...\n")

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, test in enumerate(benchmark_questions, start=1):
            print(f"[{idx}/{len(benchmark_questions)}] Processing: {test['question'][:60]}...")
            res = run_standard_rag(test["question"], test["conv_id"])
            writer.writerow({
                "category": test["category"],
                "conv_id": res["conv_id"],
                "question": res["question"],
                "std_latency_sec": res["std_latency_sec"],
                "std_final_answer": res["std_final_answer"],
            })
            print(f"    Finished in {res['std_latency_sec']}s\n")

    print(f"Done! Results written to {output_file}")


if __name__ == "__main__":
    main()