import csv
import os
from pathlib import Path
import time
from typing import Any, Dict
import requests

CURRENT_DIR = Path(__file__).resolve().parent
DB_SERVICE_URL = os.environ.get("DB_SERVICE_URL", "http://127.0.0.1:8003")
AUTONOMOUS_PROCESS_URL = f"{DB_SERVICE_URL}/process"


def run_autonomous_rag(question: str, conv_id: str) -> Dict[str, Any]:
    start_time = time.time()
    payload = {"question": question, "conv_id": conv_id}

    try:
        response = requests.post(
            AUTONOMOUS_PROCESS_URL, json=payload, timeout=300
        )
        latency = round(time.time() - start_time, 3)

        if response.status_code == 200:
            data = response.json()
            mermaid_diagram = data.get("graph_mermaid", "")
            revision_cycles = mermaid_diagram.count("Needs Revision")

            return {
                "dag_latency_sec": latency,
                "dag_revision_cycles": revision_cycles,
                "dag_final_answer": data.get("final_answer", ""),
                "dag_graph_mermaid": mermaid_diagram,
                "dag_graph_image_url": data.get("graph_image_url", ""),
            }
        else:
            return {
                "dag_latency_sec": latency,
                "dag_revision_cycles": 0,
                "dag_final_answer": f"HTTP Error {response.status_code}: {response.text}",
                "dag_graph_mermaid": "",
                "dag_graph_image_url": "",
            }
    except Exception as e:
        return {
            "dag_latency_sec": round(time.time() - start_time, 3),
            "dag_revision_cycles": 0,
            "dag_final_answer": f"Request Failed: {str(e)}",
            "dag_graph_mermaid": "",
            "dag_graph_image_url": "",
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
        "dag_latency_sec",
        "dag_revision_cycles",
        "dag_final_answer",
        "dag_graph_mermaid",
        "dag_graph_image_url",
    ]

    output_file = CURRENT_DIR /"autonomous_rag_results.csv"
    print(output_file)

    print(f"Connecting to target server: {AUTONOMOUS_PROCESS_URL}")
    print(f"Starting evaluation of {len(benchmark_questions)} benchmark questions...\n")

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, test in enumerate(benchmark_questions, start=1):
            print(f"[{idx}/{len(benchmark_questions)}] Processing: {test['question'][:60]}...")
            
            res = run_autonomous_rag(test["question"], test["conv_id"])
            
            writer.writerow({
                "category": test["category"],
                "conv_id": test["conv_id"],
                "question": test["question"],
                "dag_latency_sec": res["dag_latency_sec"],
                "dag_revision_cycles": res["dag_revision_cycles"],
                "dag_final_answer": res["dag_final_answer"],
                "dag_graph_mermaid": res["dag_graph_mermaid"],
                "dag_graph_image_url": res["dag_graph_image_url"],
            })
            
            print(f"    Status: Completed in {res['dag_latency_sec']}s | Revision Cycles: {res['dag_revision_cycles']}\n")

    print(f"Benchmark finished successfully! Results saved to: {output_file}")


if __name__ == "__main__":
    main()