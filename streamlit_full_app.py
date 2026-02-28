"""
streamlit_full_app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full interactive Loan Risk prediction demo.

Runs entirely offline — loads all trained artifacts from disk and performs
multimodal inference without calling FastAPI.

Pipeline:
  1. Tabular branch  : sklearn XGBoost pipeline  → 1-dim logit
  2. Sequence branch : deterministic 5-stage sequence → LSTM → 32-dim embedding
  3. Graph branch    : precomputed GraphSAGE embeddings (zero-vector fallback)
  4. Fusion branch   : MLP(65-dim) → sigmoid → approval probability

Explainability:
  • SHAP TreeExplainer on XGBoost classifier for top-3 tabular features
  • Bar chart + gauge visualisation for results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import torch
import shap  # type: ignore
import streamlit as st
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Project root on sys.path so model modules are importable ─────────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.fusion.fusion_mlp import FusionMLP
from models.sequence.lstm import LSTMSequenceEncoder, pad_or_truncate_sequences

# ─── Artifact paths ───────────────────────────────────────────────────────────
TABULAR_PIPELINE_PATH  = ROOT / "artifacts/tabular/sklearn_xgb_pipeline.joblib"
LSTM_MODEL_PATH        = ROOT / "artifacts/sequence/lstm_encoder.pt"
GRAPH_EMBEDDINGS_PATH  = ROOT / "artifacts/graph/precomputed_node_embeddings.npy"
GRAPH_INDEX_PATH       = ROOT / "artifacts/graph/node_embedding_index.pkl"
FUSION_MODEL_PATH      = ROOT / "artifacts/fusion/fusion_mlp.pt"
SCALING_STATS_PATH     = ROOT / "artifacts/sequence/scaling_stats.json"

# Sequence dims — derived from training artifacts (shape 981×5×8)
SEQUENCE_STAGES   = 5
SEQUENCE_FEAT_DIM = 8
LSTM_HIDDEN_SIZE  = 32
GRAPH_EMBED_DIM   = 32
FUSION_INPUT_DIM  = 65  # 1 tabular logit + 32 LSTM + 32 graph

# Sequence feature names (8 features per stage)
SEQUENCE_FEATURE_NAMES = [
    "Applicant Income Pattern",
    "Co-applicant Income Pattern",
    "Total Income Pattern",
    "Loan Amount Pattern",
    "Loan Term Pattern",
    "Credit History State",
    "Dependents Impact",
    "Leverage State",
]

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanRisk AI · Live Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #09111f 0%, #0d1a30 60%, #090f1c 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a30 0%, #090f1c 100%);
    border-right: 1px solid rgba(99,179,237,0.18);
}

/* Result cards */
.result-card {
    border-radius: 18px;
    padding: 2rem 2rem 1.5rem 2rem;
    margin: 1rem 0;
    backdrop-filter: blur(12px);
}
.approved-card {
    background: linear-gradient(135deg, rgba(5,46,22,0.85), rgba(6,78,59,0.6));
    border: 2px solid #10b981;
    box-shadow: 0 0 32px rgba(16,185,129,0.25);
}
.rejected-card {
    background: linear-gradient(135deg, rgba(69,10,10,0.85), rgba(127,29,29,0.6));
    border: 2px solid #ef4444;
    box-shadow: 0 0 32px rgba(239,68,68,0.25);
}
.decision-label {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-align: center;
}
.approved-text  { color: #34d399; }
.rejected-text  { color: #f87171; }

/* Metric cards */
.kpi-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.kpi-label  { font-size: 0.74rem; color: #93c5fd; text-transform:uppercase; letter-spacing:0.07em; }
.kpi-value  { font-size: 1.9rem; font-weight: 700; color: #e2e8f0; }
.kpi-sub    { font-size: 0.7rem; color: #64748b; margin-top: 0.2rem; }

/* Section headers */
.sec-header {
    font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
    border-left: 3px solid #3b82f6; padding-left: 0.6rem;
    margin: 1.4rem 0 0.8rem 0;
}

/* Explanation box */
.explain-box {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.7;
}

/* Sidebar title */
.sidebar-title {
    text-align: center; padding: 1rem 0 1.2rem 0;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING  (cached — loaded once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_all_artifacts():
    """
    Load and cache all model artifacts.  Called once per Streamlit session.
    Returns a dict of ready-to-use model objects.
    """
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ── 1. Tabular pipeline (XGBoost sklearn pipeline) ────────────────────────
    tabular_pipeline = joblib.load(TABULAR_PIPELINE_PATH)
    preprocessor     = tabular_pipeline.named_steps["preprocessor"]
    classifier       = tabular_pipeline.named_steps["classifier"]
    feature_names    = preprocessor.get_feature_names_out().tolist()
    shap_explainer   = shap.TreeExplainer(classifier)

    # ── 2. LSTM sequence encoder ──────────────────────────────────────────────
    lstm_model = LSTMSequenceEncoder(
        feature_dim=SEQUENCE_FEAT_DIM,
        max_seq_len=SEQUENCE_STAGES,
    )
    lstm_state = torch.load(LSTM_MODEL_PATH, map_location=device)
    lstm_model.load_state_dict(lstm_state)
    lstm_model.to(device).eval()

    # ── 3. Graph embedding store + lookup index ───────────────────────────────
    graph_embeddings = np.load(GRAPH_EMBEDDINGS_PATH).astype(np.float32)  # (981, 32)
    with open(GRAPH_INDEX_PATH, "rb") as fh:
        raw_index: dict = pickle.load(fh)
    graph_lookup: dict[str, int] = {}
    for k, v in raw_index.items():
        graph_lookup[str(k)] = int(v)
        graph_lookup[k]       = int(v)

    # ── 4. Fusion MLP ─────────────────────────────────────────────────────────
    fusion_model = FusionMLP(input_dim=FUSION_INPUT_DIM)
    fusion_state = torch.load(FUSION_MODEL_PATH, map_location=device)
    fusion_model.load_state_dict(fusion_state)
    fusion_model.to(device).eval()

    # ── 5. Scaling stats (for sequence normalization) ─────────────────────────
    with open(SCALING_STATS_PATH, "r") as fh:
        scaling_stats = json.load(fh)

    return {
        "device":           device,
        "tabular_pipeline": tabular_pipeline,
        "preprocessor":     preprocessor,
        "classifier":       classifier,
        "feature_names":    feature_names,
        "shap_explainer":   shap_explainer,
        "lstm_model":       lstm_model,
        "graph_embeddings": graph_embeddings,
        "graph_lookup":     graph_lookup,
        "fusion_model":     fusion_model,
        "scaling_stats":    scaling_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE FEATURE BUILDER  (mirrors scripts/build_sequence_features.py)
# ══════════════════════════════════════════════════════════════════════════════

def _zscore_single(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise a 1-D or 2-D array (one 'batch')."""
    mean = np.mean(arr, axis=0)
    std  = np.std(arr, axis=0) + 1e-8
    return (arr - mean) / std


def build_sequence_for_single_applicant(
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_term: float,
    credit_history: float,
    dependents: float,
    scaling_stats: dict,
) -> np.ndarray:
    """
    Deterministically construct the 5-stage pseudo-temporal financial
    progression sequence for ONE new applicant, with Z-score normalization.

    Returns shape: (SEQUENCE_STAGES=5, SEQUENCE_FEAT_DIM=8)
    """
    def normalize(vals: np.ndarray, stage_key: str) -> np.ndarray:
        mean = np.array(scaling_stats[stage_key]["mean"], dtype=np.float32)
        std = np.array(scaling_stats[stage_key]["std"], dtype=np.float32)
        return (vals - mean) / std

    # Raw values
    total_income    = applicant_income + coapplicant_income
    loan_to_income  = loan_amount / (total_income + 1.0)
    credit          = np.clip(credit_history, 0.0, 1.0)
    deps            = max(dependents, 0.0)

    # Stage 1: raw normalized state
    stage1_raw = np.array([
        applicant_income, coapplicant_income, total_income,
        loan_amount, loan_term, credit, deps, loan_to_income
    ], dtype=np.float32)
    stage1 = normalize(stage1_raw, "stage1")

    # Stage 2: debt-to-income adjusted
    dti        = loan_amount / (total_income + 1.0)
    dti_clip   = np.clip(dti, 0.0, 2.0)
    capacity   = np.clip(1.0 - 0.5 * dti_clip, 0.1, 1.0)
    stage2_raw = np.array([
        applicant_income   * capacity,
        coapplicant_income * capacity,
        total_income       * capacity,
        loan_amount,
        loan_term,
        credit,
        deps,
        dti_clip,
    ], dtype=np.float32)
    stage2 = normalize(stage2_raw, "stage2")

    # Stage 3: credit-weighted
    cw  = 0.5 + 0.5 * credit
    stage3_raw = np.array([
        stage2_raw[0] * cw,
        stage2_raw[1] * cw,
        stage2_raw[2] * cw,
        stage2_raw[3] * (2.0 - cw),
        stage2_raw[4],
        credit,
        deps,
        stage2_raw[7] * (2.0 - cw),
    ], dtype=np.float32)
    stage3 = normalize(stage3_raw, "stage3")

    # Stage 4: risk interaction features
    emi_proxy          = loan_amount / (loan_term + 1.0)
    burden_ratio       = emi_proxy / (total_income + 1.0)
    co_support_ratio   = coapplicant_income / (total_income + 1.0)
    dependent_pressure = deps / (deps + 1.0)
    credit_penalty     = (1.0 - credit) * burden_ratio
    stability_proxy    = (1.0 - np.clip(burden_ratio, 0.0, 1.0)) * (0.5 + 0.5 * credit)
    dtc_interaction    = dti_clip * (1.0 - 0.5 * credit)
    loan_to_primary    = loan_amount / (applicant_income + 1.0)
    stage4_raw = np.array([
        emi_proxy, burden_ratio, co_support_ratio, dependent_pressure,
        credit_penalty, stability_proxy, dtc_interaction, loan_to_primary,
    ], dtype=np.float32)
    stage4 = normalize(stage4_raw, "stage4")

    # Stage 5: blend of stages 3 & 4
    stage5 = 0.6 * stage3 + 0.4 * stage4

    # Stack to (5, 8)
    sequence = np.stack([stage1, stage2, stage3, stage4, stage5], axis=0).astype(np.float32)
    return sequence


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_logit(prob: float, eps: float = 1e-6) -> float:
    prob = np.clip(prob, eps, 1.0 - eps)
    return float(np.log(prob / (1.0 - prob)))


def run_tabular(artifacts: dict, row_df: pd.DataFrame) -> tuple[float, float]:
    """Run the XGBoost sklearn pipeline → return (raw_prob, logit)."""
    pipeline = artifacts["tabular_pipeline"]
    prob     = float(pipeline.predict_proba(row_df)[0, 1])
    logit    = _safe_logit(prob)
    return prob, logit


def run_lstm(artifacts: dict, sequence: np.ndarray) -> np.ndarray:
    """Encode a single (5,8) sequence through the LSTM → 32-dim embedding."""
    device     = artifacts["device"]
    lstm_model = artifacts["lstm_model"]
    # pad_or_truncate_sequences expects a list of 2-D arrays
    padded, lengths = pad_or_truncate_sequences([sequence], max_seq_len=SEQUENCE_STAGES)
    x = torch.tensor(padded, dtype=torch.float32).to(device)
    l = torch.tensor(lengths, dtype=torch.int64).to(device)
    with torch.no_grad():
        emb = lstm_model(x, l)
    return emb.cpu().numpy()[0].astype(np.float32)


def run_graph_lookup(artifacts: dict, loan_id: str | None) -> np.ndarray:
    """
    Retrieve precomputed graph embedding for a known Loan_ID.
    Falls back to a zero-vector for new/unknown applicants.
    """
    graph_embeddings = artifacts["graph_embeddings"]
    graph_lookup     = artifacts["graph_lookup"]
    if loan_id and loan_id in graph_lookup:
        idx = graph_lookup[loan_id]
        return graph_embeddings[idx].astype(np.float32)
    # Zero-vector fallback for new applicants not in graph
    return np.zeros(GRAPH_EMBED_DIM, dtype=np.float32)


def run_fusion(artifacts: dict, tabular_logit: float,
               lstm_emb: np.ndarray, graph_emb: np.ndarray) -> tuple[float, float]:
    """
    Concatenate branch outputs → run FusionMLP → return (approval_prob, risk_score).
    approval_prob ∈ (0,1)  — higher = more likely to be approved
    risk_score    = 1 - approval_prob
    """
    device       = artifacts["device"]
    fusion_model = artifacts["fusion_model"]

    fusion_input = np.concatenate([
        np.array([tabular_logit], dtype=np.float32),
        lstm_emb.astype(np.float32),
        graph_emb.astype(np.float32),
    ]).astype(np.float32)

    if fusion_input.shape[0] != FUSION_INPUT_DIM:
        raise ValueError(
            f"Fusion input dim mismatch: got {fusion_input.shape[0]}, expected {FUSION_INPUT_DIM}"
        )

    tensor = torch.tensor(fusion_input[None, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = float(fusion_model(tensor).item())

    approval_prob = float(1.0 / (1.0 + np.exp(-logit)))
    risk_score    = float(1.0 - approval_prob)
    return approval_prob, risk_score


def run_shap_explainer(
    artifacts: dict, row_df: pd.DataFrame, top_k: int = 3
) -> list[dict[str, Any]]:
    """Return top-k SHAP feature contributions for the tabular branch."""
    preprocessor  = artifacts["preprocessor"]
    feature_names = artifacts["feature_names"]
    explainer     = artifacts["shap_explainer"]

    transformed = preprocessor.transform(row_df)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    shap_vals = explainer.shap_values(dense)
    # XGBoost multi-output: take positive class
    if isinstance(shap_vals, list):
        values = np.asarray(shap_vals[-1])[0]
    else:
        values = np.asarray(shap_vals)[0]

    top_indices = np.argsort(np.abs(values))[::-1][:top_k]
    return [
        {
            "feature": _clean_feature_name(feature_names[i]),
            "raw_name": feature_names[i],
            "impact":   float(values[i]),
        }
        for i in top_indices
    ]


def _clean_feature_name(raw: str) -> str:
    """Human-readable version of sklearn ColumnTransformer feature names."""
    explicit = {
        "num__Credit_History":    "Credit History",
        "num__LoanAmount":        "Loan Amount",
        "num__ApplicantIncome":   "Applicant Income",
        "num__CoapplicantIncome": "Co-applicant Income",
        "num__Loan_Amount_Term":  "Loan Amount Term",
    }
    if raw in explicit:
        return explicit[raw]
    if raw.startswith("cat__Property_Area_"):
        area = raw.replace("cat__Property_Area_", "").replace("_", " ")
        return f"Property Area ({area.title()})"
    if raw.startswith("cat__"):
        return raw.replace("cat__", "").replace("_", " ").title()
    if raw.startswith("num__"):
        return raw.replace("num__", "").replace("_", " ").title()
    return raw.replace("_", " ").title()


def run_full_inference(
    artifacts: dict,
    row_df: pd.DataFrame,
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_term: float,
    credit_history: float,
    dependents: float,
    loan_id: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end inference pipeline for one applicant.

    Parameters
    ----------
    artifacts       : loaded model artifacts dict
    row_df          : single-row DataFrame with all raw feature columns
    applicant_income / ... : individual numeric fields for sequence builder
    loan_id         : optional, used for graph embedding lookup
    """
    # ── Tabular branch ────────────────────────────────────────────────────────
    tabular_prob, tabular_logit = run_tabular(artifacts, row_df)

    # ── Sequence branch ───────────────────────────────────────────────────────
    sequence = build_sequence_for_single_applicant(
        applicant_income   = applicant_income,
        coapplicant_income = coapplicant_income,
        loan_amount        = loan_amount,
        loan_term          = loan_term,
        credit_history     = credit_history,
        dependents         = dependents,
        scaling_stats      = artifacts["scaling_stats"],
    )
    lstm_emb = run_lstm(artifacts, sequence)

    # ── Graph branch (zero-vector fallback for new applicants) ────────────────
    graph_emb = run_graph_lookup(artifacts, loan_id)
    graph_source = "lookup" if (loan_id and loan_id in artifacts["graph_lookup"]) else "zero_fallback"

    # ── Fusion branch ─────────────────────────────────────────────────────────
    approval_prob, risk_score = run_fusion(artifacts, tabular_logit, lstm_emb, graph_emb)

    # Decision threshold = 0.5 on approval probability
    decision = "Approved" if approval_prob >= 0.5 else "Rejected"
    confidence = max(approval_prob, 1.0 - approval_prob)

    # ── SHAP explanations ─────────────────────────────────────────────────────
    shap_features = run_shap_explainer(artifacts, row_df, top_k=3)

    return {
        "decision":        decision,
        "approval_prob":   approval_prob,
        "risk_score":      risk_score,
        "confidence":      confidence,
        "tabular_logit":   tabular_logit,
        "tabular_prob":    tabular_prob,
        "shap_features":   shap_features,
        "graph_source":    graph_source,
        "lstm_embedding":  lstm_emb,
        "graph_embedding": graph_emb,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_gauge(approval_prob: float) -> go.Figure:
    """Plotly gauge chart showing approval probability."""
    pct = approval_prob * 100
    color = "#10b981" if approval_prob >= 0.5 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 34, "color": "#e2e8f0"}},
        delta={"reference": 50, "increasing": {"color": "#10b981"},
               "decreasing": {"color": "#ef4444"},
               "font": {"size": 14}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#64748b",
                      "tickfont": {"color": "#94a3b8", "size": 11}},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(99,179,237,0.2)",
            "steps": [
                {"range": [0, 50],  "color": "rgba(239,68,68,0.12)"},
                {"range": [50, 100], "color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {
                "line":  {"color": "#fbbf24", "width": 2.5},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Approval Probability", "font": {"color": "#94a3b8", "size": 13}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        height=260,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


def plot_shap_bar(shap_features: list[dict]) -> go.Figure:
    """Horizontal bar chart of top SHAP contributions."""
    names   = [f["feature"]  for f in shap_features]
    impacts = [f["impact"]   for f in shap_features]
    colors  = ["#ef4444" if v > 0 else "#10b981" for v in impacts]

    fig = go.Figure(go.Bar(
        y=names,
        x=impacts,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.4f}" for v in impacts],
        textposition="outside",
        textfont={"color": "#cbd5e1", "size": 12},
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=80, t=10, b=10),
        height=220,
        xaxis=dict(
            title="SHAP Impact (+ increases risk, − lowers risk)",
            gridcolor="rgba(99,179,237,0.1)",
            zerolinecolor="rgba(99,179,237,0.35)",
            zerolinewidth=1.5,
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_branch_contributions(
    tabular_prob: float, lstm_emb: np.ndarray, graph_emb: np.ndarray
) -> go.Figure:
    """Simple bar chart showing relative L2 contribution of each branch."""
    t_mag = abs(tabular_prob - 0.5)  # distance from centre
    s_mag = float(np.linalg.norm(lstm_emb))
    g_mag = float(np.linalg.norm(graph_emb))

    labels = ["Tabular (XGBoost)", "Sequence (LSTM)", "Graph (GraphSAGE)"]
    values = [t_mag, s_mag, g_mag]
    colors = ["#3b82f6", "#f59e0b", "#10b981"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#cbd5e1", "size": 12},
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        yaxis=dict(title="Signal Magnitude", gridcolor="rgba(99,179,237,0.1)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict[str, Any]:
    """Render all input widgets in the sidebar. Returns collected values."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">
            <div style="font-size:2.2rem;">🏦</div>
            <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-top:0.2rem;">LoanRisk AI</div>
            <div style="font-size:0.72rem;color:#64748b;">Multimodal · Offline · Live Predictor</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 👤 Applicant Details")

        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        married = st.selectbox("Married", ["Yes", "No"], index=0)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0)
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"], index=0
        )

        st.markdown("---")
        st.markdown("### 💰 Financial Details")

        applicant_income = st.number_input(
            "Applicant Income (₹/month)", min_value=0, max_value=1_000_000,
            value=3812, step=100,
            help="Monthly income of the primary applicant. Default: Training Median."
        )
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹/month)", min_value=0, max_value=500_000,
            value=1188, step=100,
            help="Monthly income of the co-applicant. Default: Training Median."
        )
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)", min_value=1, max_value=5000,
            value=128, step=5,
            help="Requested loan principal in thousands of ₹. Default: Training Median."
        )
        loan_term = st.number_input(
            "Loan Amount Term (months)", min_value=12, max_value=480,
            value=360, step=12,
            help="Repayment period in months."
        )
        credit_history = st.selectbox(
            "Credit History",
            options=[1.0, 0.0],
            format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)",
            help="1 = meets guidelines, 0 = does not meet guidelines."
        )

        st.markdown("---")
        with st.expander("🔬 Advanced (Graph Lookup)", expanded=False):
            loan_id_input = st.text_input(
                "Loan ID (optional)",
                value="",
                help="If you enter a Loan_ID from the training set, its precomputed "
                     "GraphSAGE embedding will be used. Otherwise a zero-vector fallback is used."
            )

        predict_clicked = st.button(
            "🔍  Predict Loan Approval",
            use_container_width=True,
            type="primary",
        )

    dep_numeric = float(dependents.replace("+", "")) if dependents != "3+" else 3.0

    return {
        "gender":             gender,
        "married":            married,
        "dependents":         dependents,
        "education":          education,
        "self_employed":      self_employed,
        "property_area":      property_area,
        "applicant_income":   float(applicant_income),
        "coapplicant_income": float(coapplicant_income),
        "loan_amount":        float(loan_amount),
        "loan_term":          float(loan_term),
        "credit_history":     float(credit_history),
        "dep_numeric":        dep_numeric,
        "loan_id":            loan_id_input.strip() or None,
        "predict_clicked":    predict_clicked,
    }


def build_tabular_row(inputs: dict) -> pd.DataFrame:
    """Construct a single-row DataFrame matching the training schema."""
    return pd.DataFrame([{
        "Loan_ID":           inputs["loan_id"] or "DUMMY",
        "Gender":            inputs["gender"],
        "Married":           inputs["married"],
        "Dependents":        inputs["dependents"],
        "Education":         inputs["education"],
        "Self_Employed":     inputs["self_employed"],
        "ApplicantIncome":   inputs["applicant_income"],
        "CoapplicantIncome": inputs["coapplicant_income"],
        "LoanAmount":        inputs["loan_amount"],
        "Loan_Amount_Term":  inputs["loan_term"],
        "Credit_History":    inputs["credit_history"],
        "Property_Area":     inputs["property_area"],
    }])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def render_results(result: dict, inputs: dict) -> None:
    """Render full results panel after inference is complete."""
    decision     = result["decision"]
    approval_prob = result["approval_prob"]
    risk_score   = result["risk_score"]
    confidence   = result["confidence"]
    shap_features = result["shap_features"]

    is_approved = decision == "Approved"
    card_cls    = "approved-card" if is_approved else "rejected-card"
    text_cls    = "approved-text"  if is_approved else "rejected-text"
    icon        = "✅"             if is_approved else "❌"

    # ── Hero decision card ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="decision-label {text_cls}">{icon} {decision.upper()}</div>
        <div style="text-align:center;font-size:0.9rem;color:#94a3b8;margin-top:0.6rem;">
            Based on multimodal AI analysis of your application
        </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, (label, val, sub) in zip([k1, k2, k3, k4], [
        ("Approval Probability", f"{approval_prob:.1%}", "P(Approved)"),
        ("Risk Score",           f"{risk_score:.4f}",   "1 − P(Approved)"),
        ("Confidence",           f"{confidence:.1%}",   "Distance from 50%"),
        ("Tabular XGB Score",    f"{result['tabular_prob']:.1%}", "XGBoost branch"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Confidence bar ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Confidence</div>', unsafe_allow_html=True)
    st.progress(int(confidence * 100))
    st.caption(
        f"Model is **{confidence:.1%} confident** in this decision "
        f"(approval probability = {approval_prob:.4f}, threshold = 0.50)"
    )

    # ── Gauge + SHAP bar ──────────────────────────────────────────────────────
    g_col, s_col = st.columns([1, 1])

    with g_col:
        st.markdown('<div class="sec-header">Approval Probability Gauge</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_gauge(approval_prob), use_container_width=True)

    with s_col:
        st.markdown('<div class="sec-header">Top 3 SHAP Feature Impacts</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_shap_bar(shap_features), use_container_width=True)

    # ── Branch contribution chart ─────────────────────────────────────────────
    st.markdown('<div class="sec-header">Multimodal Branch Signal Strengths</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        plot_branch_contributions(
            result["tabular_prob"],
            result["lstm_embedding"],
            result["graph_embedding"],
        ),
        use_container_width=True,
    )

    # ── SHAP feature table ────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">SHAP Feature Detail</div>',
                unsafe_allow_html=True)
    shap_df = pd.DataFrame([
        {
            "Feature": f["feature"],
            "SHAP Impact": f"{f['impact']:+.5f}",
            "Direction": "⬆ Increases Risk" if f["impact"] > 0 else "⬇ Lowers Risk",
        }
        for f in shap_features
    ])
    st.dataframe(
        shap_df, use_container_width=True, hide_index=True,
        column_config={
            "SHAP Impact": st.column_config.NumberColumn(format="%.5f"),
        }
    )

    # ── Natural language explanation ──────────────────────────────────────────
    st.markdown('<div class="sec-header">📝 Explanation</div>', unsafe_allow_html=True)
    f1, f2, f3 = [f["feature"] for f in shap_features[:3]]
    if is_approved:
        explanation = (
            f"Loan **approved** because **{f1}**, **{f2}**, and **{f3}** positively "
            f"influenced the approval decision. Financial behavior indicators were stable. "
            f"Overall approval probability: **{approval_prob:.2%}** "
            f"(risk score: {risk_score:.4f})."
        )
    else:
        explanation = (
            f"Loan **rejected** because **{f1}**, **{f2}**, and **{f3}** significantly "
            f"increased the risk score. Behavioral sequence patterns also contributed to risk. "
            f"Overall approval probability: **{approval_prob:.2%}** "
            f"(risk score: {risk_score:.4f})."
        )
    st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

    # ── Graph embedding source note ───────────────────────────────────────────
    if result["graph_source"] == "zero_fallback":
        st.info(
            "ℹ️ No precomputed GraphSAGE embedding was found for this applicant — "
            "a zero-vector fallback was used for the graph branch. "
            "Enter a valid Loan ID from the training set to use the real graph embedding.",
            icon="ℹ️",
        )
    else:
        st.success(
            f"✅ Graph embedding loaded from precomputed store (Loan ID: {inputs['loan_id']}).",
            icon="✅"
        )


def render_placeholder() -> None:
    """Shown before any prediction is submitted."""
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
        <div style="font-size:4rem;">🏦</div>
        <h2 style="color:#94a3b8;margin-top:1rem;font-weight:500;">
            Multimodal Loan Risk Predictor
        </h2>
        <p style="color:#475569;max-width:480px;margin:0.8rem auto 0 auto;font-size:0.9rem;line-height:1.7;">
            Fill in all applicant details in the sidebar and click
            <b style="color:#93c5fd;">Predict Loan Approval</b> to run the full
            XGBoost + LSTM + GraphSAGE + Fusion MLP pipeline locally.
        </p>
        <div style="margin-top:2rem;display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;">
            <span style="background:rgba(59,130,246,0.15);border:1px solid #3b82f6;border-radius:8px;
                         padding:0.3rem 0.8rem;font-size:0.78rem;color:#93c5fd;">🟦 XGBoost Tabular</span>
            <span style="background:rgba(245,158,11,0.15);border:1px solid #f59e0b;border-radius:8px;
                         padding:0.3rem 0.8rem;font-size:0.78rem;color:#fcd34d;">🟨 LSTM Sequence</span>
            <span style="background:rgba(16,185,129,0.15);border:1px solid #10b981;border-radius:8px;
                         padding:0.3rem 0.8rem;font-size:0.78rem;color:#6ee7b7;">🟩 GraphSAGE</span>
            <span style="background:rgba(139,92,246,0.15);border:1px solid #8b5cf6;border-radius:8px;
                         padding:0.3rem 0.8rem;font-size:0.78rem;color:#c4b5fd;">🟥 Fusion MLP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Page header
    st.markdown("""
    <div style="margin-bottom:0.5rem;">
        <h1 style="font-size:1.8rem;font-weight:700;color:#e2e8f0;margin:0;">
            🏦 LoanRisk AI — Live Applicant Predictor
        </h1>
        <p style="color:#64748b;margin:0.2rem 0 0 0;font-size:0.85rem;">
            Offline multimodal inference · XGBoost + LSTM + GraphSAGE + Fusion MLP
        </p>
    </div>
    <hr style="border:none;border-top:1px solid rgba(99,179,237,0.15);margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    # Load artifacts (cached)
    try:
        artifacts = load_all_artifacts()
    except Exception as exc:
        st.error(f"❌ Failed to load model artifacts: {exc}")
        st.stop()

    # Sidebar form
    inputs = render_sidebar()

    # Main panel
    if inputs["predict_clicked"]:
        row_df = build_tabular_row(inputs)

        with st.spinner("Running multimodal inference…"):
            try:
                result = run_full_inference(
                    artifacts          = artifacts,
                    row_df             = row_df,
                    applicant_income   = inputs["applicant_income"],
                    coapplicant_income = inputs["coapplicant_income"],
                    loan_amount        = inputs["loan_amount"],
                    loan_term          = inputs["loan_term"],
                    credit_history     = inputs["credit_history"],
                    dependents         = inputs["dep_numeric"],
                    loan_id            = inputs["loan_id"],
                )
            except Exception as exc:
                st.error(f"❌ Inference error: {exc}")
                st.stop()

        render_results(result, inputs)

    else:
        render_placeholder()


if __name__ == "__main__":
    main()
