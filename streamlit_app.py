"""
Multimodal Loan Risk Prediction System — Streamlit Dashboard
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Risk · Multimodal AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark premium background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a1020 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1528 0%, #0a1020 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(23,37,84,0.7) 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 0.5rem;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,179,237,0.2);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #93c5fd;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 4px solid #3b82f6;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Alert badges */
.alert-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #fca5a5;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.25rem;
}
.ok-badge {
    background: linear-gradient(135deg, #064e3b, #065f46);
    color: #6ee7b7;
    border-color: #10b981;
}

/* Approved/Rejected pills */
.pill-approved {
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid #10b981;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.pill-rejected {
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid #ef4444;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
EVAL_PATH  = os.path.join(BASE, "artifacts", "evaluation_report.json")
FAIR_PATH  = os.path.join(BASE, "artifacts", "fairness_report.json")
PRED_PATH  = os.path.join(BASE, "artifacts", "test_predictions_with_explanations.csv")
TRAIN_PATH = os.path.join(BASE, "data", "raw", "TRAIN.csv")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_eval():
    with open(EVAL_PATH) as f:
        return json.load(f)

@st.cache_data
def load_fairness():
    with open(FAIR_PATH) as f:
        return json.load(f)

@st.cache_data
def load_predictions():
    return pd.read_csv(PRED_PATH)

@st.cache_data
def load_train():
    return pd.read_csv(TRAIN_PATH)

eval_data = load_eval()
fair_data = load_fairness()
pred_df   = load_predictions()
train_df  = load_train()

# ── Sidebar navigation ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 1.5rem 0;">
        <div style="font-size:2.4rem;">🏦</div>
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:0.3rem;">LoanRisk AI</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem;">Multimodal Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Model Performance", "⚖️ Fairness Analysis", "🔍 Predictions Explorer", "📈 Data Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#475569; line-height:1.6;">
        <b style="color:#94a3b8;">Model Stack</b><br>
        🟦 XGBoost (Tabular)<br>
        🟨 LSTM (Sequence)<br>
        🟩 GraphSAGE (Graph)<br>
        🟥 MLP Fusion
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
if "Performance" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2rem; font-weight:700; color:#e2e8f0; margin:0;">
            📊 Model Performance
        </h1>
        <p style="color:#64748b; margin:0.3rem 0 0 0; font-size:0.9rem;">
            Hold-out validation results · 123 samples · TRAIN.csv
        </p>
    </div>""", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Accuracy", f"{eval_data['accuracy']:.2%}", "Overall correctness"),
        ("Precision", f"{eval_data['precision']:.2%}", "True positive rate"),
        ("Recall", f"{eval_data['recall']:.2%}", "Sensitivity"),
        ("F1 Score", f"{eval_data['f1_score']:.2%}", "Harmonic mean"),
        ("ROC-AUC", f"{eval_data['roc_auc']:.4f}", "Discrimination power"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    # Confusion matrix
    with col_left:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = eval_data["confusion_matrix"]
        labels = ["Rejected", "Approved"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, "#0f172a"], [0.5, "#1e40af"], [1, "#3b82f6"]],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 28, "color": "white"},
            showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=13),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Metrics radar
    with col_right:
        st.markdown('<div class="section-header">Metrics Radar</div>', unsafe_allow_html=True)
        metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        metrics_vals  = [eval_data["accuracy"], eval_data["precision"],
                         eval_data["recall"],   eval_data["f1_score"],
                         eval_data["roc_auc"]]
        fig_radar = go.Figure(go.Scatterpolar(
            r=metrics_vals + [metrics_vals[0]],
            theta=metrics_names + [metrics_names[0]],
            fill="toself",
            fillcolor="rgba(59,130,246,0.15)",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=7, color="#60a5fa"),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0.9, 1.0],
                                gridcolor="rgba(99,179,237,0.15)", tickfont=dict(color="#64748b")),
                angularaxis=dict(gridcolor="rgba(99,179,237,0.15)", tickfont=dict(color="#94a3b8")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            margin=dict(l=40, r=40, t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Model architecture info
    st.markdown('<div class="section-header">Multimodal Fusion Architecture</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    arch = [
        ("🟦", "Tabular", "XGBoost", "1-dim logit\nImputation + scaling\nOne-hot encoding"),
        ("🟨", "Sequence", "LSTM", "32-dim embedding\nPseudo-temporal\nfinancial features"),
        ("🟩", "Graph", "GraphSAGE", "32-dim embedding\nkNN similarity graph\nPrecomputed offline"),
        ("🟥", "Fusion", "MLP", "65-dim concat input\nFinal approval logit\nSigmoid output"),
    ]
    for col, (icon, name, model, desc) in zip([mc1, mc2, mc3, mc4], arch):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-size:0.85rem; font-weight:700; color:#e2e8f0;">{name}</div>
                <div style="font-size:0.75rem; color:#3b82f6; font-weight:600; margin-bottom:0.4rem;">{model}</div>
                <div style="font-size:0.72rem; color:#64748b; white-space:pre-line;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FAIRNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Fairness" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2rem; font-weight:700; color:#e2e8f0; margin:0;">
            ⚖️ Fairness Analysis
        </h1>
        <p style="color:#64748b; margin:0.3rem 0 0 0; font-size:0.9rem;">
            Group-level bias detection · Alert threshold: ±15% from global metric
        </p>
    </div>""", unsafe_allow_html=True)

    gm = fair_data["global_metrics"]
    alerts = fair_data["alerts"]

    # Global metrics
    g1, g2, g3, g4 = st.columns(4)
    for col, (label, val) in zip([g1, g2, g3, g4], [
        ("Global Approval Rate", f"{gm['approval_rate']:.2%}"),
        ("Global Rejection Rate", f"{gm['rejection_rate']:.2%}"),
        ("False Positive Rate", f"{gm['false_positive_rate']:.2%}"),
        ("False Negative Rate", f"{gm['false_negative_rate']:.2%}"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.6rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

    # Alerts
    st.markdown('<div class="section-header">⚠️ Fairness Alerts</div>', unsafe_allow_html=True)
    if alerts:
        for a in alerts:
            st.markdown(f"""
            <span class="alert-badge">
                {a['group']} = {a['value']} · {a['metric']} = {a['group_value_metric']:.2%}
                (global {a['global_metric']:.2%}, Δ {a['absolute_difference']:.2%})
            </span>""", unsafe_allow_html=True)
    else:
        st.markdown('<span class="alert-badge ok-badge">✅ No fairness alerts</span>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Group selector
    groups = list(fair_data["group_metrics"].keys())
    selected_group = st.selectbox("Select demographic group to analyse:", groups)

    group_rows = fair_data["group_metrics"][selected_group]
    gdf = pd.DataFrame(group_rows)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Approval & Rejection Rates</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Approval Rate", x=gdf["value"], y=gdf["approval_rate"],
            marker_color="#3b82f6", marker_line_width=0,
        ))
        fig_bar.add_trace(go.Bar(
            name="Rejection Rate", x=gdf["value"], y=gdf["rejection_rate"],
            marker_color="#ef4444", marker_line_width=0,
        ))
        # Global reference line
        fig_bar.add_hline(y=gm["approval_rate"], line_dash="dot",
                          line_color="#fbbf24", annotation_text="Global approval",
                          annotation_font_color="#fbbf24")
        fig_bar.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            yaxis=dict(tickformat=".0%", gridcolor="rgba(99,179,237,0.08)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">FPR vs FNR by Subgroup</div>', unsafe_allow_html=True)
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            name="FPR", x=gdf["value"], y=gdf["false_positive_rate"],
            marker_color="#f59e0b",
        ))
        fig_err.add_trace(go.Bar(
            name="FNR", x=gdf["value"], y=gdf["false_negative_rate"],
            marker_color="#8b5cf6",
        ))
        fig_err.add_hline(y=gm["false_positive_rate"], line_dash="dot",
                          line_color="#f59e0b", annotation_text="Global FPR",
                          annotation_font_color="#f59e0b")
        fig_err.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            yaxis=dict(tickformat=".0%", gridcolor="rgba(99,179,237,0.08)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # Sample-size bubble
    st.markdown('<div class="section-header">Sample Size Distribution</div>', unsafe_allow_html=True)
    fig_bubble = px.scatter(
        gdf, x="value", y="approval_rate", size="sample_size",
        color="false_positive_rate",
        color_continuous_scale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        labels={"value": "Group Value", "approval_rate": "Approval Rate",
                "false_positive_rate": "FPR", "sample_size": "Sample Size"},
        hover_data=["sample_size", "false_positive_rate", "false_negative_rate"],
    )
    fig_bubble.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(tickformat=".0%", gridcolor="rgba(99,179,237,0.08)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTIONS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Predictions" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2rem; font-weight:700; color:#e2e8f0; margin:0;">
            🔍 Predictions Explorer
        </h1>
        <p style="color:#64748b; margin:0.3rem 0 0 0; font-size:0.9rem;">
            TEST set · Batch predictions with multimodal explanations
        </p>
    </div>""", unsafe_allow_html=True)

    # Summary KPIs
    total    = len(pred_df)
    approved = (pred_df["Predicted_Loan_Status"] == "Approved").sum()
    rejected = total - approved
    avg_risk = pred_df["Risk_Score"].mean()

    k1, k2, k3, k4 = st.columns(4)
    for col, (label, val, sub) in zip([k1, k2, k3, k4], [
        ("Total Applicants", str(total), "TEST set"),
        ("Approved", f"{approved} ({approved/total:.0%})", "Predicted approval"),
        ("Rejected", f"{rejected} ({rejected/total:.0%})", "Predicted rejection"),
        ("Avg Risk Score", f"{avg_risk:.3f}", "Mean sigmoid output"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.7rem;">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Risk score distribution
    col_dist, col_pie = st.columns([2, 1])

    with col_dist:
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pred_df[pred_df["Predicted_Loan_Status"] == "Approved"]["Risk_Score"],
            nbinsx=30, name="Approved",
            marker_color="rgba(16,185,129,0.7)",
        ))
        fig_hist.add_trace(go.Histogram(
            x=pred_df[pred_df["Predicted_Loan_Status"] == "Rejected"]["Risk_Score"],
            nbinsx=30, name="Rejected",
            marker_color="rgba(239,68,68,0.7)",
        ))
        fig_hist.update_layout(
            barmode="overlay",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(title="Risk Score (≈ P(Approval))", gridcolor="rgba(99,179,237,0.08)"),
            yaxis=dict(title="Count", gridcolor="rgba(99,179,237,0.08)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_pie:
        st.markdown('<div class="section-header">Decision Split</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Approved", "Rejected"],
            values=[approved, rejected],
            marker=dict(colors=["#10b981", "#ef4444"],
                        line=dict(color="#0a0e1a", width=3)),
            hole=0.55,
            textfont=dict(color="#e2e8f0", size=13),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Graph influence score
    st.markdown('<div class="section-header">Graph Influence Score vs Risk Score</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        pred_df,
        x="Graph_Influence_Score",
        y="Risk_Score",
        color="Predicted_Loan_Status",
        color_discrete_map={"Approved": "#10b981", "Rejected": "#ef4444"},
        hover_data=["Loan_ID", "Top_Tabular_Features"],
        labels={"Graph_Influence_Score": "Graph Influence Score",
                "Risk_Score": "Risk Score (P(Approval))"},
    )
    fig_scatter.update_traces(marker=dict(size=6, opacity=0.75))
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), title="Decision"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        xaxis=dict(gridcolor="rgba(99,179,237,0.08)"),
        yaxis=dict(gridcolor="rgba(99,179,237,0.08)"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Individual record lookup
    st.markdown('<div class="section-header">Individual Applicant Lookup</div>', unsafe_allow_html=True)
    loan_ids = pred_df["Loan_ID"].tolist()
    selected_id = st.selectbox("Select Loan ID:", loan_ids)
    row = pred_df[pred_df["Loan_ID"] == selected_id].iloc[0]

    pill_cls = "pill-approved" if row["Predicted_Loan_Status"] == "Approved" else "pill-rejected"
    icon = "✅" if row["Predicted_Loan_Status"] == "Approved" else "❌"

    ia, ib, ic = st.columns(3)
    with ia:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div class="metric-label">Decision</div>
            <span class="{pill_cls}">{icon} {row['Predicted_Loan_Status']}</span>
            <div class="metric-sub" style="margin-top:0.6rem;">Risk Score: <b style="color:#e2e8f0;">{row['Risk_Score']:.4f}</b></div>
        </div>""", unsafe_allow_html=True)
    with ib:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div class="metric-label">Top Tabular Features</div>
            <div style="font-size:0.80rem; color:#93c5fd; margin-top:0.3rem;">{row['Top_Tabular_Features']}</div>
        </div>""", unsafe_allow_html=True)
    with ic:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div class="metric-label">Graph Influence Score</div>
            <div class="metric-value" style="font-size:1.6rem;">{row['Graph_Influence_Score']:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="text-align:left; margin-top:0.5rem;">
        <div class="metric-label">💬 Explanation</div>
        <div style="font-size:0.88rem; color:#cbd5e1; margin-top:0.5rem; line-height:1.6;">{row['Explanation_Text']}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Data" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2rem; font-weight:700; color:#e2e8f0; margin:0;">
            📈 Data Insights
        </h1>
        <p style="color:#64748b; margin:0.3rem 0 0 0; font-size:0.9rem;">
            TRAIN dataset exploration · {n} applicants
        </p>
    </div>""".replace("{n}", str(len(train_df))), unsafe_allow_html=True)

    # Summary stats
    d1, d2, d3, d4 = st.columns(4)
    approved_train = (train_df["Loan_Status"] == "Y").sum()
    for col, (label, val) in zip([d1, d2, d3, d4], [
        ("Total Samples", str(len(train_df))),
        ("Approved", f"{approved_train} ({approved_train/len(train_df):.0%})"),
        ("Rejected", f"{len(train_df)-approved_train} ({(len(train_df)-approved_train)/len(train_df):.0%})"),
        ("Avg Loan Amt", f"₹{train_df['LoanAmount'].mean():.0f}K"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.7rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns(2)

    # Income distribution
    with row1_c1:
        st.markdown('<div class="section-header">Applicant Income by Loan Status</div>', unsafe_allow_html=True)
        fig_inc = go.Figure()
        violin_colors = {
            "Y": ("rgba(16,185,129,0.3)", "#10b981", "Approved"),
            "N": ("rgba(239,68,68,0.3)", "#ef4444", "Rejected"),
        }
        for status, (fill, line, label) in violin_colors.items():
            sub = train_df[train_df["Loan_Status"] == status]["ApplicantIncome"]
            fig_inc.add_trace(go.Violin(
                y=sub, name=label,
                box_visible=True, meanline_visible=True,
                fillcolor=fill,
                line_color=line,
            ))
        fig_inc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            yaxis=dict(title="Income (₹)", gridcolor="rgba(99,179,237,0.08)"),
        )
        st.plotly_chart(fig_inc, use_container_width=True)

    # Credit history impact
    with row1_c2:
        st.markdown('<div class="section-header">Credit History vs Approval</div>', unsafe_allow_html=True)
        ch = train_df.groupby(["Credit_History", "Loan_Status"]).size().reset_index(name="Count")
        ch["Credit_History"] = ch["Credit_History"].map({0.0: "Bad Credit", 1.0: "Good Credit"})
        fig_ch = px.bar(
            ch, x="Credit_History", y="Count", color="Loan_Status",
            color_discrete_map={"Y": "#10b981", "N": "#ef4444"},
            labels={"Credit_History": "", "Count": "Applicants", "Loan_Status": "Status"},
            barmode="group",
        )
        fig_ch.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), title=""),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            yaxis=dict(gridcolor="rgba(99,179,237,0.08)"),
        )
        st.plotly_chart(fig_ch, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    # Property area
    with row2_c1:
        st.markdown('<div class="section-header">Approval by Property Area</div>', unsafe_allow_html=True)
        pa = train_df.groupby(["Property_Area", "Loan_Status"]).size().reset_index(name="Count")
        fig_pa = px.bar(
            pa, x="Property_Area", y="Count", color="Loan_Status",
            color_discrete_map={"Y": "#3b82f6", "N": "#f59e0b"},
            labels={"Property_Area": "", "Count": "Applicants"},
            barmode="stack",
        )
        fig_pa.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), title=""),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            yaxis=dict(gridcolor="rgba(99,179,237,0.08)"),
        )
        st.plotly_chart(fig_pa, use_container_width=True)

    # Loan amount distribution
    with row2_c2:
        st.markdown('<div class="section-header">Loan Amount Distribution</div>', unsafe_allow_html=True)
        fig_la = go.Figure()
        fig_la.add_trace(go.Histogram(
            x=train_df["LoanAmount"].dropna(),
            nbinsx=40,
            marker_color="rgba(139,92,246,0.7)",
            marker_line_color="rgba(167,139,250,0.9)",
            marker_line_width=0.5,
        ))
        fig_la.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(title="Loan Amount (₹K)", gridcolor="rgba(99,179,237,0.08)"),
            yaxis=dict(title="Count", gridcolor="rgba(99,179,237,0.08)"),
        )
        st.plotly_chart(fig_la, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    corr = train_df[num_cols].dropna().corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#ef4444"], [0.5, "#0f172a"], [1, "#3b82f6"]],
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        showscale=True,
    ))
    fig_corr.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
        coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
