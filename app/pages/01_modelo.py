"""
01_modelo.py — PoD Bank Credit Score — Métricas e comparação de modelos
"""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="PoD Bank — Modelo",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.markdown(
    """
    <style>
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #c0c0e0;
        margin: 28px 0 10px 0;
        border-left: 4px solid #2ecc71;
        padding-left: 10px;
    }
    .metric-table th {
        background: #1e1e2e;
        color: #a0a0b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 📊 Análise Detalhada do Modelo")
st.markdown(
    "<span style='color:#a0a0b8;'>Métricas completas, comparação entre modelos e curvas de diagnóstico.</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Métricas holdout ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Métricas no Holdout (20% estratificado)</div>', unsafe_allow_html=True)

holdout_data = {
    "Métrica": ["AUC-ROC", "KS Statistic", "Gini", "Recall Adimplentes", "Recall Inadimplentes",
                "Precisão", "F1-Score", "Taxa de Aprovação", "Default Esperado (aprovados)"],
    "Valor": ["0.8223", "0.4887", "0.6447", "70.57%", "77.94%",
              "18.90%", "0.3042", "66.64%", "2.68%"],
    "Target": [">= 0.75", ">= 0.35", ">= 0.50", ">= 70%", "—",
               "—", "—", "—", "—"],
    "Status": ["✅ APROVADO", "✅ APROVADO", "✅ APROVADO", "✅ APROVADO", "—",
               "—", "—", "—", "—"],
}

import pandas as pd
df_holdout = pd.DataFrame(holdout_data)
st.dataframe(df_holdout, use_container_width=True, hide_index=True)

# ── Comparação dos 3 modelos ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Comparação dos Modelos (CV 5-fold)</div>', unsafe_allow_html=True)

models_data = {
    "Modelo": [
        "Logistic Regression (baseline)",
        "LightGBM v1",
        "XGBoost v1",
        "LightGBM v2 Tuned (champion)",
    ],
    "AUC-ROC (CV)": ["0.7559 ± 0.0027", "0.7701 ± 0.0031", "0.7698 ± 0.0036", "0.7717 ± 0.0030"],
    "KS (CV)": ["0.3857 ± 0.0054", "0.4105 ± 0.0035", "0.4065 ± 0.0083", "0.4118 ± 0.0030"],
    "Gini (CV)": ["0.5117 ± 0.0055", "0.5401 ± 0.0062", "0.5396 ± 0.0071", "0.5434 ± 0.0060"],
    "AUC-ROC (Holdout)": ["—", "—", "—", "0.8223"],
    "Status": ["✅ APROVADO", "✅ APROVADO", "✅ APROVADO", "🏆 CHAMPION"],
    "MLflow Run ID": [
        "53e25c53...7b7",
        "32a8dd5f...d1",
        "e6baf972...14f",
        "624246ba...3d9",
    ],
}

df_models = pd.DataFrame(models_data)

# Highlight champion row
def highlight_champion(row):
    if "CHAMPION" in str(row.get("Status", "")):
        return ["background-color: #1a3d2e; color: #2ecc71; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_models.style.apply(highlight_champion, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ── Gráfico comparativo ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Comparativo Visual — AUC-ROC por Modelo</div>', unsafe_allow_html=True)

model_names = ["Logistic\nRegression", "LightGBM\nv1", "XGBoost\nv1", "LightGBM\nv2 Tuned"]
auc_values = [0.7559, 0.7701, 0.7698, 0.7717]
colors = ["#4a90d9", "#4a90d9", "#4a90d9", "#2ecc71"]

fig_cmp = go.Figure(go.Bar(
    x=model_names,
    y=auc_values,
    marker_color=colors,
    text=[f"{v:.4f}" for v in auc_values],
    textposition="outside",
))
fig_cmp.add_hline(y=0.72, line_dash="dash", line_color="#f39c12",
                  annotation_text="Target mínimo (0.72)", annotation_position="top left")
fig_cmp.update_layout(
    template="plotly_dark",
    height=380,
    yaxis=dict(range=[0.70, 0.79], title="AUC-ROC"),
    margin=dict(t=40, b=10),
)
st.plotly_chart(fig_cmp, use_container_width=True)

# ── Curvas lado a lado ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Curvas de Diagnóstico</div>', unsafe_allow_html=True)

fig_cols = st.columns(2)

curves = [
    ("Curva ROC", "roc_curve.png"),
    ("Curva KS", "ks_curve.png"),
    ("Curva Lift", "lift_curve.png"),
    ("Distribuição de Scores", "score_distribution.png"),
]

for i, (title, fname) in enumerate(curves):
    col = fig_cols[i % 2]
    with col:
        st.markdown(f"**{title}**")
        img_path = FIGURES_DIR / fname
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info(f"Figura não encontrada: {fname}")

# ── Matriz de confusão ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Matriz de Confusão — Threshold 0.48</div>', unsafe_allow_html=True)

col_cm, col_cm_detail = st.columns([1, 2])

with col_cm:
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)

with col_cm_detail:
    cm_data = {
        "": ["Real: Adimplente", "Real: Inadimplente"],
        "Pred: Adimplente": ["27,924 (VP)", "768 (FN)"],
        "Pred: Inadimplente": ["11,646 (FP)", "2,714 (VP mau)"],
    }
    df_cm = pd.DataFrame(cm_data)
    st.dataframe(df_cm, use_container_width=True, hide_index=True)
    st.markdown(
        """
        - **Custo estimado** (FN×5 + FP×1): **15,486**
        - Threshold 0.48 maximiza recall de inadimplentes (77.9%) preservando aprovação de bons pagadores (70.6%)
        """
    )

# ── Top 10 Features ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Top 10 Features — Importância (Gain)</div>', unsafe_allow_html=True)

col_f1, col_f2 = st.columns(2)

features = [
    "EXT_SOURCE_2", "EXT_SOURCE_3", "ORGANIZATION_TYPE", "EXT_SOURCE_1",
    "credit_term", "OCCUPATION_TYPE", "bureau_avg_days_credit",
    "inst_late_rate", "inst_late_count", "age_years",
]
importances = [240284, 232392, 164402, 152567, 88615, 53840, 47918, 46696, 35631, 27727]

with col_f1:
    fig_fi = go.Figure(go.Bar(
        x=importances[::-1],
        y=features[::-1],
        orientation="h",
        marker=dict(
            color=importances[::-1],
            colorscale=[[0, "#1a6e3c"], [1, "#2ecc71"]],
        ),
        text=[f"{v:,.0f}" for v in importances[::-1]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis_title="Importância (Gain)",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with col_f2:
    fi_path = FIGURES_DIR / "lightgbm_tuned_feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), use_container_width=True)
    else:
        fi_path2 = FIGURES_DIR / "feature_importance_final.png"
        if fi_path2.exists():
            st.image(str(fi_path2), use_container_width=True)
