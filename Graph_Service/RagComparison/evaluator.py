import csv
import json
import os
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Dict, List
import pandas as pd
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

CURRENT_DIR = Path(__file__).resolve().parent
GRAPH_SERVICE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = GRAPH_SERVICE_DIR.parent

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

if str(GRAPH_SERVICE_DIR) not in sys.path:
    sys.path.append(str(GRAPH_SERVICE_DIR))

from app.config import MODEL_NAME, client
from google.genai import types


class PerformanceEvaluation(BaseModel):
    coverage_score: int = Field(
        description="Score (1-10) rating how thoroughly and completely the answer covers the question."
    )
    groundedness_score: int = Field(
        description="Score (1-10) rating strict context adherence with zero ungrounded outside knowledge or hallucination."
    )
    rigor_score: int = Field(
        description="Score (1-10) rating technical accuracy, clarity, and depth of analysis."
    )
    final_performance_score: float = Field(
        description="Aggregated final performance score (1.0 - 10.0) weighted as: 0.4*Coverage + 0.4*Groundedness + 0.2*Rigor."
    )
    rationale: str = Field(
        description="Concise 1-2 sentence breakdown justifying the performance evaluation components."
    )


RUBRIC_PROMPT = """You are an expert technical evaluator performing multi-component LLM performance estimation.
Assess the given Answer against the Question based on three key evaluation dimensions on a strict 1-10 scale:

1. COMPONENT 1: COVERAGE & COMPLETENESS (1-10)
   - Evaluates whether every sub-question, core constraint, and required domain aspect is addressed.

2. COMPONENT 2: GROUNDEDNESS & FAITHFULNESS (1-10)
   - Evaluates strict adherence to retrieved facts without adding ungrounded outside knowledge or hallucinations.
   - If answer states context is missing/refuses, assign 1.

3. COMPONENT 3: TECHNICAL RIGOR & PRECISION (1-10)
   - Evaluates analytical depth, formula/parameter accuracy, structural logic, and clarity.

CALCULATE FINAL PERFORMANCE SCORE:
- Final Score = (0.40 * coverage_score) + (0.40 * groundedness_score) + (0.20 * rigor_score)
Round the final_performance_score to 2 decimal places.
"""


def evaluate_single_answer(question: str, answer: str) -> PerformanceEvaluation:
    user_prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=RUBRIC_PROMPT,
                response_mime_type="application/json",
                response_schema=PerformanceEvaluation,
                temperature=0.0,
            ),
        )
        return PerformanceEvaluation.model_validate_json(response.text)
    except Exception as e:
        return PerformanceEvaluation(
            coverage_score=1,
            groundedness_score=1,
            rigor_score=1,
            final_performance_score=1.0,
            rationale=f"Evaluation failed due to exception: {str(e)}"
        )


def save_summary_statistics(df: pd.DataFrame):
    summary = pd.DataFrame({
        "Metric": [
            "Final Performance Score (1-10)",
            "Coverage Component (1-10)",
            "Groundedness Component (1-10)",
            "Rigor Component (1-10)",
            "Average Word Count",
        ],
        "Standard RAG": [
            round(df["std_final_score"].mean(), 2),
            round(df["std_coverage_score"].mean(), 2),
            round(df["std_groundedness_score"].mean(), 2),
            round(df["std_rigor_score"].mean(), 2),
            round(df["std_word_count"].mean(), 1),
        ],
        "Autonomous RAG": [
            round(df["auto_final_score"].mean(), 2),
            round(df["auto_coverage_score"].mean(), 2),
            round(df["auto_groundedness_score"].mean(), 2),
            round(df["auto_rigor_score"].mean(), 2),
            round(df["auto_word_count"].mean(), 1),
        ],
        "Absolute Delta": [
            round(df["auto_final_score"].mean() - df["std_final_score"].mean(), 2),
            round(df["auto_coverage_score"].mean() - df["std_coverage_score"].mean(), 2),
            round(df["auto_groundedness_score"].mean() - df["std_groundedness_score"].mean(), 2),
            round(df["auto_rigor_score"].mean() - df["std_rigor_score"].mean(), 2),
            round(df["auto_word_count"].mean() - df["std_word_count"].mean(), 1),
        ],
    })
    summary.to_csv(CURRENT_DIR / "evaluation_summary.csv", index=False)


def run_llm_evaluator():
    std_file = CURRENT_DIR / "standard_rag_result.csv"
    auto_file = CURRENT_DIR / "autonomous_rag_results.csv"
    output_file = CURRENT_DIR / "evaluated_comparison_results.csv"

    if not std_file.exists():
        raise FileNotFoundError(f"Missing input file: {std_file}")
    if not auto_file.exists():
        raise FileNotFoundError(f"Missing input file: {auto_file}")

    std_df = pd.read_csv(std_file)
    auto_df = pd.read_csv(auto_file)

    merged = pd.merge(
        std_df,
        auto_df,
        on=["category", "conv_id", "question"],
        suffixes=("_std", "_auto"),
    )

    std_final, std_cov, std_ground, std_rig, std_rat = [], [], [], [], []
    auto_final, auto_cov, auto_ground, auto_rig, auto_rat = [], [], [], [], []

    for idx, row in merged.iterrows():
        question = str(row["question"])
        std_ans = str(row["std_final_answer"])
        auto_ans = str(row["dag_final_answer"])

        print(f"[{idx + 1}/{len(merged)}] Evaluating ID: {row['conv_id']}...")

        std_eval = evaluate_single_answer(question, std_ans)
        std_final.append(std_eval.final_performance_score)
        std_cov.append(std_eval.coverage_score)
        std_ground.append(std_eval.groundedness_score)
        std_rig.append(std_eval.rigor_score)
        std_rat.append(std_eval.rationale)

        auto_eval = evaluate_single_answer(question, auto_ans)
        auto_final.append(auto_eval.final_performance_score)
        auto_cov.append(auto_eval.coverage_score)
        auto_ground.append(auto_eval.groundedness_score)
        auto_rig.append(auto_eval.rigor_score)
        auto_rat.append(auto_eval.rationale)

        print(
            f"  └─ Standard: {std_eval.final_performance_score}/10 | Autonomous: {auto_eval.final_performance_score}/10"
        )

    merged["std_final_score"] = std_final
    merged["std_coverage_score"] = std_cov
    merged["std_groundedness_score"] = std_ground
    merged["std_rigor_score"] = std_rig
    merged["std_rationale"] = std_rat

    merged["auto_final_score"] = auto_final
    merged["auto_coverage_score"] = auto_cov
    merged["auto_groundedness_score"] = auto_ground
    merged["auto_rigor_score"] = auto_rig
    merged["auto_rationale"] = auto_rat

    merged["std_word_count"] = merged["std_final_answer"].apply(
        lambda x: len(str(x).split())
    )
    merged["auto_word_count"] = merged["dag_final_answer"].apply(
        lambda x: len(str(x).split())
    )

    merged.to_csv(output_file, index=False)
    save_summary_statistics(merged)


if __name__ == "__main__":
    run_llm_evaluator()