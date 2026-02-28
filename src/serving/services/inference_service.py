from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.explainability import explainer_service as es


@dataclass
class InferenceServiceError(Exception):
    status_code: int
    detail: str


class InferenceService:
    def __init__(self) -> None:
        # Caches are initialized in explainer_service at import/startup.
        self._device = es._DEVICE

    @property
    def device_name(self) -> str:
        return str(self._device)

    @property
    def applicant_count(self) -> int:
        dataframe = es._require_initialized("dataframe", es._DATAFRAME)
        return int(len(dataframe))

    @property
    def artifacts_cached(self) -> bool:
        return (
            es._TABULAR_PIPELINE is not None
            and es._LSTM_MODEL is not None
            and es._GRAPH_EMBEDDINGS is not None
            and es._FUSION_MODEL is not None
            and es._SHAP_EXPLAINER is not None
        )

    def score_applicant(self, loan_id: str) -> dict[str, Any]:
        try:
            applicant_row = es._find_applicant_row(loan_id=loan_id)
            tabular_row = applicant_row.drop(columns=["Loan_Status"], errors="ignore")
            tabular_pipeline = es._require_initialized("tabular_pipeline", es._TABULAR_PIPELINE)

            tabular_logit = float(es.compute_tabular_logits(tabular_pipeline, tabular_row)[0, 0])
            sequence = es._load_sequence_for_id(loan_id=str(loan_id))
            lstm_embedding = es._compute_lstm_embedding(sequence=sequence)
            graph_embedding = es._load_graph_embedding_for_id(loan_id=str(loan_id))

            fusion_input = np.concatenate(
                [
                    np.array([tabular_logit], dtype=np.float32),
                    lstm_embedding.astype(np.float32),
                    graph_embedding.astype(np.float32),
                ],
                axis=0,
            ).astype(np.float32)
            if fusion_input.shape[0] != 65:
                raise ValueError(f"Unexpected fusion input dimension: {fusion_input.shape[0]}")

            fusion_model = es._require_initialized("fusion_model", es._FUSION_MODEL)
            with torch.no_grad():
                fusion_tensor = torch.tensor(
                    fusion_input[None, :],
                    dtype=torch.float32,
                    device=self._device,
                )
                approval_logit = float(fusion_model(fusion_tensor).item())

            approval_prob = float(1.0 / (1.0 + np.exp(-approval_logit)))
            risk_score = float(1.0 - approval_prob)
            confidence = float(max(approval_prob, 1.0 - approval_prob))
            decision = "Rejected" if risk_score >= 0.5 else "Approved"
            return {
                "risk_score": risk_score,
                "decision": decision,
                "confidence": confidence,
            }
        except KeyError as exc:
            raise InferenceServiceError(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise InferenceServiceError(
                status_code=500,
                detail=f"Unexpected scoring error: {exc}",
            ) from exc

    def explain_applicant(self, loan_id: str) -> dict[str, Any]:
        try:
            return es.explain_applicant(loan_id=loan_id)
        except KeyError as exc:
            raise InferenceServiceError(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise InferenceServiceError(
                status_code=500,
                detail=f"Unexpected explanation error: {exc}",
            ) from exc
