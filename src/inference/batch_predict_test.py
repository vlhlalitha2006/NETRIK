from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.explainability import explainer_service as es


TEST_CSV_PATH = Path("data/raw/TEST.csv")
OUTPUT_CSV_PATH = Path("artifacts/test_predictions_with_explanations.csv")


def clean_feature_name(raw_name: str) -> str:
    explicit_map = {
        "num__Credit_History": "Credit History",
        "num__LoanAmount": "Loan Amount",
        "num__ApplicantIncome": "Applicant Income",
        "num__CoapplicantIncome": "Co-applicant Income",
        "num__Loan_Amount_Term": "Loan Amount Term",
        "coapplicant_income_state": "Co-applicant Income Pattern",
        "applicant_income_state": "Applicant Income Pattern",
        "total_income_state": "Total Income Pattern",
        "dependents_state": "Dependents Impact",
    }
    if raw_name in explicit_map:
        return explicit_map[raw_name]

    if raw_name.startswith("cat__Property_Area_"):
        suffix = raw_name.replace("cat__Property_Area_", "").replace("_", " ")
        return f"Property Area ({suffix.title()})"

    if raw_name.startswith("cat__"):
        label = raw_name.replace("cat__", "").replace("_", " ")
        return label.title()
    if raw_name.startswith("num__"):
        label = raw_name.replace("num__", "").replace("_", " ")
        return label.title()

    return raw_name.replace("_", " ").title()


def _build_explanation_text(
    decision: str,
    risk_score: float,
    top_tabular_features: list[str],
) -> str:
    f1, f2, f3 = top_tabular_features[:3]
    if decision == "Rejected":
        return (
            f"Loan rejected because {f1}, {f2}, and {f3} significantly increased the risk score. "
            "Behavioral sequence patterns also contributed to risk. "
            f"Overall risk score was {risk_score:.4f}."
        )
    return (
        f"Loan approved because {f1}, {f2}, and {f3} positively influenced approval. "
        "Financial behavior indicators were stable. "
        f"Overall risk score was {risk_score:.4f}."
    )


def _ensure_test_dataset() -> pd.DataFrame:
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing TEST dataset at: {TEST_CSV_PATH}")
    dataframe = pd.read_csv(TEST_CSV_PATH)
    if "Loan_ID" not in dataframe.columns:
        raise ValueError("TEST.csv must contain Loan_ID column.")
    return dataframe


def _validate_artifact_alignment(test_df: pd.DataFrame) -> None:
    sequence_lookup = es._require_initialized("sequence_lookup", es._SEQUENCE_LOOKUP)
    graph_lookup = es._require_initialized("graph_lookup", es._GRAPH_LOOKUP)

    missing_sequence_ids: list[str] = []
    missing_graph_ids: list[str] = []
    for loan_id in test_df["Loan_ID"].astype(str).tolist():
        if loan_id not in sequence_lookup and str(loan_id) not in sequence_lookup:
            missing_sequence_ids.append(loan_id)
        if loan_id not in graph_lookup and str(loan_id) not in graph_lookup:
            missing_graph_ids.append(loan_id)

    if missing_sequence_ids:
        preview = ", ".join(missing_sequence_ids[:10])
        raise ValueError(
            "Some TEST Loan_ID values are missing in sequence artifacts. "
            f"Count={len(missing_sequence_ids)}. Examples: {preview}"
        )
    if missing_graph_ids:
        preview = ", ".join(missing_graph_ids[:10])
        raise ValueError(
            "Some TEST Loan_ID values are missing in graph embeddings. "
            f"Count={len(missing_graph_ids)}. Examples: {preview}"
        )


def _predict_fusion_risk_score(
    tabular_logit: float,
    sequence_embedding: np.ndarray,
    graph_embedding: np.ndarray,
) -> float:
    fusion_input = np.concatenate(
        [
            np.array([tabular_logit], dtype=np.float32),
            sequence_embedding.astype(np.float32),
            graph_embedding.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    if fusion_input.shape[0] != 65:
        raise ValueError(f"Expected fused input dimension 65, got {fusion_input.shape[0]}")

    fusion_model = es._require_initialized("fusion_model", es._FUSION_MODEL)
    with torch.no_grad():
        tensor = torch.tensor(fusion_input[None, :], dtype=torch.float32, device=es._DEVICE)
        approval_logit = float(fusion_model(tensor).item())
    approval_probability = 1.0 / (1.0 + np.exp(-approval_logit))
    return float(1.0 - approval_probability)


def _explain_single_applicant(
    tabular_row: pd.DataFrame,
    tabular_logit: float,
    sequence_raw: np.ndarray,
    graph_embedding: np.ndarray,
) -> tuple[list[str], list[str], float]:
    tabular_explanations = es._compute_tabular_shap(tabular_row=tabular_row)
    _, feature_importance = es._ig_sequence_attributions(
        sequence=sequence_raw,
        tabular_logit=tabular_logit,
        graph_embedding=graph_embedding,
    )
    sequence_explanations = es._top_sequence_features(feature_importance)
    graph_influence = es._graph_influence_score(graph_embedding)

    top_tabular = [item["feature"] for item in tabular_explanations[:3]]
    top_sequence = [item["feature"] for item in sequence_explanations[:3]]
    return top_tabular, top_sequence, float(graph_influence)


def run_batch_inference_with_explanations() -> Path:
    es._initialize_caches()
    test_df = _ensure_test_dataset()
    _validate_artifact_alignment(test_df)

    tabular_pipeline = es._require_initialized("tabular_pipeline", es._TABULAR_PIPELINE)
    results: list[dict[str, Any]] = []

    for _, row in test_df.iterrows():
        loan_id = str(row["Loan_ID"])
        tabular_row = pd.DataFrame([row.to_dict()])

        tabular_logit = float(es.compute_tabular_logits(tabular_pipeline, tabular_row)[0, 0])
        sequence_raw = es._load_sequence_for_id(loan_id=loan_id)
        sequence_embedding = es._compute_lstm_embedding(sequence=sequence_raw)
        graph_embedding = es._load_graph_embedding_for_id(loan_id=loan_id)

        risk_score = _predict_fusion_risk_score(
            tabular_logit=tabular_logit,
            sequence_embedding=sequence_embedding,
            graph_embedding=graph_embedding,
        )

        # Decision rule is intentionally based on user's requested threshold semantics.
        decision = "Approved" if risk_score >= 0.5 else "Rejected"

        top_tabular_raw, top_sequence_raw, graph_influence = _explain_single_applicant(
            tabular_row=tabular_row,
            tabular_logit=tabular_logit,
            sequence_raw=sequence_raw,
            graph_embedding=graph_embedding,
        )
        top_tabular_clean = [clean_feature_name(name) for name in top_tabular_raw]
        top_sequence_clean = [clean_feature_name(name) for name in top_sequence_raw]
        explanation_text = _build_explanation_text(
            decision=decision,
            risk_score=risk_score,
            top_tabular_features=top_tabular_clean,
        )

        results.append(
            {
                "Loan_ID": loan_id,
                "Predicted_Loan_Status": decision,
                "Risk_Score": risk_score,
                "Top_Tabular_Features": ", ".join(top_tabular_clean),
                "Top_Sequence_Features": ", ".join(top_sequence_clean),
                "Graph_Influence_Score": graph_influence,
                "Explanation_Text": explanation_text,
            }
        )

    output_df = pd.DataFrame(results)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)

    approved_count = int((output_df["Predicted_Loan_Status"] == "Approved").sum())
    rejected_count = int((output_df["Predicted_Loan_Status"] == "Rejected").sum())
    print(f"Total applicants processed: {len(output_df)}")
    print(f"Number Approved: {approved_count}")
    print(f"Number Rejected: {rejected_count}")
    print(f"Output file path: {OUTPUT_CSV_PATH}")
    preview_columns = ["Loan_ID", "Predicted_Loan_Status", "Risk_Score", "Explanation_Text"]
    print("\nFirst 5 rows:")
    print(output_df[preview_columns].head(5).to_string(index=False))
    return OUTPUT_CSV_PATH


if __name__ == "__main__":
    run_batch_inference_with_explanations()
