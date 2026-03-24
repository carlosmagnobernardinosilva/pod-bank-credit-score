"""
03_carteira.py — PoD Bank Credit Score — Análise de Carteira
"""
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="PoD Bank — Carteira",
    page_icon="📁",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "lightgbm_tuned.pkl")
PIPELINE_PATH = os.path.join(ROOT, "models", "scoring_pipeline.pkl")
DATA_PATH = os.path.join(ROOT, "data", "processed", "train_final.parquet")
REPORTS_PATH = os.path.join(ROOT, "reports", "figures")
THRESHOLD = 0.48
SAMPLE_N = 10_000

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .section-title {
        font-size: 1.2rem; font-weight: 600; color: #c0c0e0;
        margin: 28px 0 10px 0; border-left: 4px solid #2ecc71; padding-left: 10px;
    }
    .kpi-mini {
        background: #1e1e2e; border-radius: 10px; padding: 16px;
        text-align: center; border: 1px solid #2d2d44;
    }
    .kpi-mini-label { font-size: 0.82rem; color: #a0a0b8; text-transform: uppercase; }
    .kpi-mini-value { font-size: 1.8rem; font-weight: 700; color: #e0e0f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 📁 Análise de Carteira")
st.markdown(
    "<span style='color:#a0a0b8;'>Performance do modelo sobre amostra estratificada "
    "da base de treino — scores reais via pipeline serializado.</span>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Carregar pipeline ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo...")
def load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        art = joblib.load(PIPELINE_PATH)
        model = art["model"]
        feature_columns = art["feature_columns"]
        source = f"scoring_pipeline.pkl (v{art.get('version','?')})"
    elif os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        feature_columns = model.booster_.feature_name()
        source = "lightgbm_tuned.pkl (fallback)"
    else:
        return None, None, None, "Nenhum modelo encontrado"

    # Extrair mapeamento categórico do booster
    try:
        dump = model.booster_.dump_model()
        pandas_categorical = dump.get("pandas_categorical", [])
        df_ref = pd.read_parquet(DATA_PATH).head(1)
        cat_cols_ordered = [
            c for c in feature_columns
            if c in df_ref.columns and str(df_ref[c].dtype) in ("object", "category")
        ]
        cat_map = dict(zip(cat_cols_ordered, pandas_categorical))
    except Exception:
        cat_map = {}

    return model, feature_columns, cat_map, source


# ── Carregar e pontuar amostra ────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando e pontuando amostra...")
def load_and_score(n: int, seed: int = 42):
    """Lê amostra estratificada do parquet e aplica batch predict_proba."""
    model, feature_columns, cat_map, source = load_pipeline()
    if model is None:
        return None, None

    # Ler colunas necessárias: TARGET + todas as features do modelo
    cols_to_read = ["TARGET"] + [c for c in feature_columns if c != "TARGET"]
    # Só ler colunas que existem no parquet
    parquet_cols = pd.read_parquet(DATA_PATH).columns.tolist()
    cols_to_read = [c for c in cols_to_read if c in parquet_cols]

    df = pd.read_parquet(DATA_PATH, columns=cols_to_read)

    # Amostra estratificada por TARGET
    parts = []
    for cls, frac in [(0, 0.92), (1, 0.08)]:
        sub = df[df["TARGET"] == cls]
        count = int(n * frac)
        parts.append(sub.sample(min(count, len(sub)), random_state=seed))
    df_sample = pd.concat(parts).reset_index(drop=True)

    # Alinhar features para o modelo
    df_feat = df_sample.copy()
    for col in feature_columns:
        if col not in df_feat.columns:
            df_feat[col] = np.nan
    df_feat = df_feat[feature_columns]

    # Aplicar categorias exatas do treinamento
    for col, cats in cat_map.items():
        if col in df_feat.columns:
            df_feat[col] = pd.Categorical(df_feat[col], categories=cats)

    # Batch predict (muito mais rápido que linha por linha)
    scores = model.predict_proba(df_feat)[:, 1]

    df_result = df_sample[["TARGET"]].copy()
    df_result["score"] = scores
    return df_result, source


with st.spinner("Carregando dados e pontuando amostra..."):
    df_scored, model_source = load_and_score(SAMPLE_N)

if df_scored is None:
    st.error("Pipeline não disponível. Verifique models/scoring_pipeline.pkl")
    st.stop()

st.caption(
    f"Modelo: {model_source} | Amostra: {len(df_scored):,} registros "
    f"(estratificada 92% adimplentes / 8% inadimplentes) | Threshold: {THRESHOLD}"
)

# ── KPIs de aprovação com threshold 0.48 ─────────────────────────────────────
st.markdown('<div class="section-title">Taxa de Aprovação e Default Esperado — Threshold 0.48</div>', unsafe_allow_html=True)

approved_mask = df_scored["score"] < THRESHOLD
n_total = len(df_scored)
n_approved = approved_mask.sum()
n_rejected = n_total - n_approved
approval_rate = n_approved / n_total
default_rate_total = df_scored["TARGET"].mean()
default_rate_approved = df_scored.loc[approved_mask, "TARGET"].mean()
default_rate_rejected = df_scored.loc[~approved_mask, "TARGET"].mean()
bads_captured = df_scored.loc[~approved_mask, "TARGET"].sum() / df_scored["TARGET"].sum()

kpis = [
    ("Taxa de Aprovação", f"{approval_rate:.1%}", "#2ecc71"),
    ("Default na Carteira Aprovada", f"{default_rate_approved:.2%}", "#f39c12"),
    ("Default na Carteira Reprovada", f"{default_rate_rejected:.2%}", "#e74c3c"),
    ("% Maus Capturados (Reprovados)", f"{bads_captured:.1%}", "#3498db"),
    ("Default Geral da Base", f"{default_rate_total:.2%}", "#a0a0b8"),
]

cols_kpi = st.columns(5)
for col, (label, value, color) in zip(cols_kpi, kpis):
    with col:
        st.markdown(
            f'<div class="kpi-mini">'
            f'<div class="kpi-mini-label">{label}</div>'
            f'<div class="kpi-mini-value" style="color:{color};">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Histograma de scores por TARGET ──────────────────────────────────────────
st.markdown('<div class="section-title">Distribuição de Scores por Classe (TARGET)</div>', unsafe_allow_html=True)

fig_dist = go.Figure()
for target_val, label, color in [
    (0, "Adimplente (TARGET=0)", "#2ecc71"),
    (1, "Inadimplente (TARGET=1)", "#e74c3c"),
]:
    subset = df_scored[df_scored["TARGET"] == target_val]["score"]
    fig_dist.add_trace(go.Histogram(
        x=subset, name=label, nbinsx=50,
        marker_color=color, opacity=0.65, histnorm="percent",
    ))

fig_dist.add_vline(
    x=THRESHOLD, line_dash="dash", line_color="white", line_width=2,
    annotation_text=f"Threshold ({THRESHOLD})",
    annotation_font_color="white",
    annotation_position="top right",
)
fig_dist.update_layout(
    template="plotly_dark", barmode="overlay",
    xaxis_title="Score (P(default))", yaxis_title="% da Classe",
    height=380, legend=dict(x=0.65, y=0.95),
)
st.plotly_chart(fig_dist, use_container_width=True)

# Imagem OOF se disponível
oof_img = os.path.join(REPORTS_PATH, "lightgbm_tuned_oof_distribution.png")
if os.path.exists(oof_img):
    with st.expander("Ver distribuição OOF do modelo tuned (imagem salva)"):
        st.image(oof_img, use_container_width=True)

# ── Tabela de performance por decil ──────────────────────────────────────────
st.markdown('<div class="section-title">Performance por Decil</div>', unsafe_allow_html=True)

df_decil = df_scored[["TARGET", "score"]].dropna().copy()
df_decil["decil_num"] = pd.qcut(df_decil["score"], q=10, labels=False) + 1
df_decil = df_decil.sort_values("decil_num")

decil_stats = (
    df_decil.groupby("decil_num", observed=True)
    .agg(
        total=("TARGET", "count"),
        n_bad=("TARGET", "sum"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        score_medio=("score", "mean"),
    )
    .reset_index()
)
decil_stats["n_good"] = decil_stats["total"] - decil_stats["n_bad"]
decil_stats["taxa_default"] = (decil_stats["n_bad"] / decil_stats["total"] * 100).round(2)

total_bad = decil_stats["n_bad"].sum()
total_good = decil_stats["n_good"].sum()
decil_stats = decil_stats.sort_values("score_medio")
decil_stats["cum_bad_pct"] = (decil_stats["n_bad"].cumsum() / total_bad * 100).round(1)
decil_stats["cum_good_pct"] = (decil_stats["n_good"].cumsum() / total_good * 100).round(1)
decil_stats["ks"] = ((decil_stats["cum_bad_pct"] - decil_stats["cum_good_pct"]).abs() / 100).round(4)

# Lift: (% maus capturados até decil X) / (% população até decil X)
decil_stats["cum_pop_pct"] = (decil_stats["total"].cumsum() / decil_stats["total"].sum() * 100).round(1)
decil_stats["lift"] = (decil_stats["cum_bad_pct"] / decil_stats["cum_pop_pct"]).round(2)

display_df = decil_stats[[
    "decil_num", "score_min", "score_max", "score_medio",
    "total", "n_bad", "taxa_default", "cum_bad_pct", "cum_good_pct", "ks", "lift"
]].rename(columns={
    "decil_num": "Decil",
    "score_min": "Score Min",
    "score_max": "Score Max",
    "score_medio": "Score Médio",
    "total": "Total",
    "n_bad": "Inadimplentes",
    "taxa_default": "Default (%)",
    "cum_bad_pct": "% Maus Acum.",
    "cum_good_pct": "% Bons Acum.",
    "ks": "KS",
    "lift": "Lift",
})
for c in ["Score Min", "Score Max", "Score Médio"]:
    display_df[c] = display_df[c].round(3)

st.dataframe(
    display_df.style.background_gradient(subset=["Default (%)"], cmap="RdYlGn_r"),
    use_container_width=True,
    hide_index=True,
)

# ── Curva Lift e KS por decil ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Curva Lift e Curva KS por Decil</div>', unsafe_allow_html=True)

col_lift, col_ks = st.columns(2)

with col_lift:
    fig_lift = go.Figure()
    fig_lift.add_trace(go.Scatter(
        x=decil_stats["cum_pop_pct"],
        y=decil_stats["cum_bad_pct"],
        mode="lines+markers",
        name="Modelo",
        line=dict(color="#2ecc71", width=2.5),
        marker=dict(size=7),
    ))
    fig_lift.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines", name="Aleatório",
        line=dict(color="#666", dash="dash"),
    ))
    fig_lift.add_vline(
        x=THRESHOLD * 100, line_dash="dot", line_color="#f39c12",
        annotation_text=f"Threshold {THRESHOLD}",
        annotation_font_color="#f39c12",
    )
    fig_lift.update_layout(
        template="plotly_dark",
        title="Curva Lift (% Maus Capturados)",
        xaxis_title="% da Carteira Revisada (por score crescente)",
        yaxis_title="% dos Inadimplentes Capturados",
        height=360,
    )
    st.plotly_chart(fig_lift, use_container_width=True)

with col_ks:
    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(
        x=decil_stats["cum_pop_pct"],
        y=decil_stats["cum_bad_pct"],
        mode="lines+markers", name="% Maus Acum.",
        line=dict(color="#e74c3c", width=2),
    ))
    fig_ks.add_trace(go.Scatter(
        x=decil_stats["cum_pop_pct"],
        y=decil_stats["cum_good_pct"],
        mode="lines+markers", name="% Bons Acum.",
        line=dict(color="#2ecc71", width=2),
    ))
    max_ks_row = decil_stats.loc[decil_stats["ks"].idxmax()]
    fig_ks.add_annotation(
        x=max_ks_row["cum_pop_pct"],
        y=(max_ks_row["cum_bad_pct"] + max_ks_row["cum_good_pct"]) / 2,
        text=f"KS = {max_ks_row['ks']:.3f}",
        showarrow=True, arrowhead=2, arrowcolor="#f39c12",
        font=dict(color="#f39c12", size=13),
    )
    fig_ks.update_layout(
        template="plotly_dark",
        title="Curva KS por Decil",
        xaxis_title="% da Carteira (score crescente)",
        yaxis_title="% Acumulado",
        height=360,
    )
    st.plotly_chart(fig_ks, use_container_width=True)

# ── Taxa de default por decil (barchart) ─────────────────────────────────────
fig_bar = go.Figure(go.Bar(
    x=[f"D{int(d)}" for d in decil_stats["decil_num"]],
    y=decil_stats["taxa_default"],
    marker=dict(
        color=decil_stats["taxa_default"],
        colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
    ),
    text=decil_stats["taxa_default"].apply(lambda x: f"{x:.1f}%"),
    textposition="outside",
))
fig_bar.update_layout(
    template="plotly_dark",
    title="Taxa de Default por Decil de Score",
    xaxis_title="Decil (D1 = menor score = menor risco)",
    yaxis_title="Taxa de Default (%)",
    height=320,
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Matriz de confusão ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Matriz de Confusão — Threshold 0.48</div>', unsafe_allow_html=True)

pred = (df_scored["score"] >= THRESHOLD).astype(int)
actual = df_scored["TARGET"].astype(int)

tn = int(((pred == 0) & (actual == 0)).sum())
fp = int(((pred == 1) & (actual == 0)).sum())
fn = int(((pred == 0) & (actual == 1)).sum())
tp = int(((pred == 1) & (actual == 1)).sum())
total = tn + fp + fn + tp

col_cm, col_metrics = st.columns([2, 3])

with col_cm:
    fig_cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
        texttemplate="%{text}",
        x=["Pred: Adimplente", "Pred: Inadimplente"],
        y=["Real: Adimplente", "Real: Inadimplente"],
        colorscale=[[0, "#1e1e2e"], [0.5, "#2d4a3e"], [1, "#2ecc71"]],
        showscale=False,
        textfont={"size": 14, "color": "white"},
    ))
    fig_cm.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_metrics:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_bad = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_good = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall_bad / (precision + recall_bad) if (precision + recall_bad) > 0 else 0

    metrics = {
        "Acurácia": f"{(tn + tp) / total:.3f}",
        "Precisão (maus)": f"{precision:.3f}",
        "Recall — Maus Pagadores": f"{recall_bad:.3f}",
        "Recall — Bons Pagadores": f"{recall_good:.3f}",
        "F1-Score": f"{f1:.3f}",
        "Taxa de Aprovação": f"{(tn + fn) / total:.3f}",
        "Total avaliado": f"{total:,}",
    }
    for k, v in metrics.items():
        st.markdown(
            f"<div style='padding:5px 0; border-bottom:1px solid #2d2d44;'>"
            f"<span style='color:#a0a0b8;'>{k}:</span> "
            f"<strong style='color:#e0e0f0;'>{v}</strong></div>",
            unsafe_allow_html=True,
        )

# ── Figuras salvas do pipeline de avaliação ───────────────────────────────────
available_figs = [
    ("confusion_matrix.png", "Matriz de Confusão"),
    ("roc_curve.png", "Curva ROC"),
    ("lift_curve.png", "Curva Lift"),
]
existing = [(os.path.join(REPORTS_PATH, f), t) for f, t in available_figs if os.path.exists(os.path.join(REPORTS_PATH, f))]
if existing:
    st.markdown('<div class="section-title">Figuras do Pipeline de Avaliação</div>', unsafe_allow_html=True)
    cols_fig = st.columns(len(existing))
    for col, (img_path, title) in zip(cols_fig, existing):
        with col:
            st.markdown(f"**{title}**")
            st.image(img_path, use_container_width=True)
