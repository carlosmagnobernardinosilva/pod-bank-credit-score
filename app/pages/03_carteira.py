"""
03_carteira.py — PoD Bank Credit Score — Análise de Carteira
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="PoD Bank — Carteira",
    page_icon="📁",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_final.parquet"

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 📁 Análise de Carteira")
st.markdown(
    "<span style='color:#a0a0b8;'>Distribuição de scores, performance por decil e "
    "matriz de confusão sobre a base de treino.</span>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Carregamento de dados ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando base de dados...")
def load_train_sample(path: Path, n: int, seed: int = 42) -> pd.DataFrame:
    """Carrega amostra estratificada do train_final.parquet."""
    df = pd.read_parquet(path, columns=["TARGET", "EXT_SOURCE_2", "EXT_SOURCE_3",
                                         "EXT_SOURCE_1", "AMT_CREDIT", "credit_term",
                                         "age_years", "DAYS_BIRTH"])
    # Amostra estratificada por TARGET
    n_per_class = {0: int(n * 0.92), 1: int(n * 0.08)}
    parts = []
    for cls, count in n_per_class.items():
        sub = df[df["TARGET"] == cls]
        parts.append(sub.sample(min(count, len(sub)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


@st.cache_data(show_spinner="Gerando scores simulados...")
def compute_proxy_scores(df: pd.DataFrame) -> pd.Series:
    """
    Score proxy baseado em EXT_SOURCEs para demonstração quando o pipeline
    não está disponível.
    """
    e1 = df["EXT_SOURCE_1"].fillna(0.5)
    e2 = df["EXT_SOURCE_2"].fillna(0.5)
    e3 = df["EXT_SOURCE_3"].fillna(0.5)
    raw = 1.0 - (0.35 * e2 + 0.35 * e3 + 0.30 * e1)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.08, size=len(df))
    return pd.Series(np.clip(raw + noise, 0.0, 1.0), name="score")


# ── Tentar carregar dados reais ───────────────────────────────────────────────
_data_available = False
_score_source = "proxy"

if TRAIN_PATH.exists():
    try:
        df_sample = load_train_sample(TRAIN_PATH, SAMPLE_N)
        _data_available = True
    except Exception as e:
        st.warning(f"Não foi possível carregar o parquet: {e}")
else:
    st.warning(
        f"Arquivo `train_final.parquet` não encontrado em `{TRAIN_PATH}`. "
        "Exibindo análise com dados simulados para demonstração."
    )

# ── Tentar scoring real ───────────────────────────────────────────────────────
if _data_available:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from models.predict import predict_score

        @st.cache_data(show_spinner="Aplicando modelo aos dados...")
        def score_batch(df: pd.DataFrame) -> pd.Series:
            scores = []
            cols = df.columns.tolist()
            for _, row in df.iterrows():
                try:
                    r = predict_score(row[cols].to_dict())
                    scores.append(r["score"])
                except Exception:
                    scores.append(np.nan)
            return pd.Series(scores, name="score")

        # Para performance, usar proxy se amostra > 500
        if len(df_sample) <= 500:
            scores = score_batch(df_sample)
            _score_source = "model"
        else:
            scores = compute_proxy_scores(df_sample)
            _score_source = "proxy"
    except ImportError:
        scores = compute_proxy_scores(df_sample)
        _score_source = "proxy"
else:
    # Gerar dados sintéticos realistas
    rng = np.random.default_rng(42)
    n_good = int(SAMPLE_N * 0.92)
    n_bad = SAMPLE_N - n_good
    scores_good = np.clip(rng.beta(2.5, 5, n_good), 0, 1)
    scores_bad = np.clip(rng.beta(4, 2.5, n_bad), 0, 1)
    all_scores = np.concatenate([scores_good, scores_bad])
    all_targets = np.array([0] * n_good + [1] * n_bad)
    df_sample = pd.DataFrame({"TARGET": all_targets})
    scores = pd.Series(all_scores, name="score")
    _score_source = "synthetic"

df_sample = df_sample.copy()
df_sample["score"] = scores.values if hasattr(scores, "values") else scores

score_caption = {
    "model": "Scores gerados pelo modelo LightGBM v2 Tuned",
    "proxy": "Scores proxy baseados em EXT_SOURCE (demonstração)",
    "synthetic": "Scores sintéticos para demonstração (parquet não disponível)",
}.get(_score_source, "")
st.caption(f"Fonte dos scores: {score_caption} | Amostra: {len(df_sample):,} registros")

# ── Distribuição de scores por TARGET ────────────────────────────────────────
st.markdown('<div class="section-title">Distribuição de Scores por Classe (TARGET)</div>', unsafe_allow_html=True)

fig_dist = go.Figure()
for target_val, label, color in [(0, "Adimplente (TARGET=0)", "#2ecc71"),
                                   (1, "Inadimplente (TARGET=1)", "#e74c3c")]:
    subset = df_sample[df_sample["TARGET"] == target_val]["score"]
    if len(subset) > 0:
        fig_dist.add_trace(go.Histogram(
            x=subset,
            name=label,
            nbinsx=50,
            marker_color=color,
            opacity=0.65,
            histnorm="percent",
        ))

fig_dist.add_vline(
    x=THRESHOLD,
    line_dash="dash", line_color="white", line_width=2,
    annotation_text=f"Threshold ({THRESHOLD})",
    annotation_font_color="white",
)
fig_dist.update_layout(
    template="plotly_dark",
    barmode="overlay",
    xaxis_title="Score (P(default))",
    yaxis_title="% da Classe",
    height=380,
    legend=dict(x=0.65, y=0.95),
)
st.plotly_chart(fig_dist, use_container_width=True)

# Imagem salva se disponível
score_img = FIGURES_DIR / "lightgbm_tuned_oof_distribution.png"
if score_img.exists():
    with st.expander("Ver distribuição OOF do modelo tuned (imagem)"):
        st.image(str(score_img), use_container_width=True)

# ── Tabela de performance por decil ──────────────────────────────────────────
st.markdown('<div class="section-title">Performance por Decil</div>', unsafe_allow_html=True)

df_decil = df_sample[["TARGET", "score"]].copy().dropna()
df_decil["decil"] = pd.qcut(df_decil["score"], q=10, labels=[f"D{i}" for i in range(1, 11)])
df_decil["decil_num"] = pd.qcut(df_decil["score"], q=10, labels=False) + 1
df_decil = df_decil.sort_values("decil_num")

decil_stats = (
    df_decil.groupby("decil_num", observed=True)
    .agg(
        total=("TARGET", "count"),
        n_bad=("TARGET", "sum"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        score_mean=("score", "mean"),
    )
    .reset_index()
)
decil_stats["n_good"] = decil_stats["total"] - decil_stats["n_bad"]
decil_stats["taxa_default"] = (decil_stats["n_bad"] / decil_stats["total"] * 100).round(2)
decil_stats["pct_total"] = (decil_stats["total"] / decil_stats["total"].sum() * 100).round(1)

# KS por decil
total_bad = decil_stats["n_bad"].sum()
total_good = decil_stats["n_good"].sum()
decil_stats = decil_stats.sort_values("score_mean")
decil_stats["cum_bad_rate"] = decil_stats["n_bad"].cumsum() / total_bad
decil_stats["cum_good_rate"] = decil_stats["n_good"].cumsum() / total_good
decil_stats["ks"] = (decil_stats["cum_bad_rate"] - decil_stats["cum_good_rate"]).abs().round(4)

display_cols = {
    "decil_num": "Decil",
    "score_min": "Score Min",
    "score_max": "Score Max",
    "score_mean": "Score Médio",
    "total": "Total",
    "n_bad": "Inadimplentes",
    "taxa_default": "Taxa Default (%)",
    "cum_bad_rate": "% Maus Acum.",
    "cum_good_rate": "% Bons Acum.",
    "ks": "KS",
}
df_display = decil_stats[list(display_cols.keys())].rename(columns=display_cols)
df_display["Score Min"] = df_display["Score Min"].round(3)
df_display["Score Max"] = df_display["Score Max"].round(3)
df_display["Score Médio"] = df_display["Score Médio"].round(3)
df_display["% Maus Acum."] = (df_display["% Maus Acum."] * 100).round(1).astype(str) + "%"
df_display["% Bons Acum."] = (df_display["% Bons Acum."] * 100).round(1).astype(str) + "%"

st.dataframe(
    df_display.style.background_gradient(subset=["Taxa Default (%)"], cmap="RdYlGn_r"),
    use_container_width=True,
    hide_index=True,
)

# ── Gráfico de taxa de default por decil ─────────────────────────────────────
col_bar, col_ks = st.columns(2)

with col_bar:
    fig_decil = go.Figure(go.Bar(
        x=[f"D{i}" for i in decil_stats["decil_num"]],
        y=decil_stats["taxa_default"],
        marker=dict(
            color=decil_stats["taxa_default"],
            colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
        ),
        text=decil_stats["taxa_default"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig_decil.update_layout(
        template="plotly_dark",
        title="Taxa de Default por Decil",
        xaxis_title="Decil de Score",
        yaxis_title="Taxa de Default (%)",
        height=340,
    )
    st.plotly_chart(fig_decil, use_container_width=True)

with col_ks:
    ks_sorted = decil_stats.sort_values("score_mean")
    fig_ks_decil = go.Figure()
    fig_ks_decil.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=ks_sorted["cum_bad_rate"] * 100,
        mode="lines+markers", name="% Maus Acum.", line=dict(color="#e74c3c"),
    ))
    fig_ks_decil.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=ks_sorted["cum_good_rate"] * 100,
        mode="lines+markers", name="% Bons Acum.", line=dict(color="#2ecc71"),
    ))
    fig_ks_decil.update_layout(
        template="plotly_dark",
        title="Curva KS por Decil",
        xaxis_title="Decil", yaxis_title="% Acumulado",
        height=340,
    )
    st.plotly_chart(fig_ks_decil, use_container_width=True)

# ── Matriz de confusão ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Matriz de Confusão — Threshold 0.48</div>', unsafe_allow_html=True)

df_cm = df_sample[["TARGET", "score"]].dropna()
pred = (df_cm["score"] >= THRESHOLD).astype(int)
actual = df_cm["TARGET"].astype(int)

tn = int(((pred == 0) & (actual == 0)).sum())
fp = int(((pred == 1) & (actual == 0)).sum())
fn = int(((pred == 0) & (actual == 1)).sum())
tp = int(((pred == 1) & (actual == 1)).sum())

col_cm_plot, col_cm_metrics = st.columns([2, 3])

with col_cm_plot:
    z = [[tn, fp], [fn, tp]]
    text_z = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
    ]
    fig_cm = go.Figure(go.Heatmap(
        z=z,
        text=text_z,
        texttemplate="%{text}",
        x=["Pred: Adimplente", "Pred: Inadimplente"],
        y=["Real: Adimplente", "Real: Inadimplente"],
        colorscale=[[0, "#1e1e2e"], [0.5, "#2d4a3e"], [1, "#2ecc71"]],
        showscale=False,
        textfont={"size": 14, "color": "white"},
    ))
    fig_cm.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cm_metrics:
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_bad = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_good = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall_bad / (precision + recall_bad) if (precision + recall_bad) > 0 else 0
    approval_rate = (tn + fn) / total if total > 0 else 0

    metrics = {
        "Acurácia": f"{accuracy:.3f}",
        "Precisão (maus)": f"{precision:.3f}",
        "Recall Maus Pagadores": f"{recall_bad:.3f}",
        "Recall Bons Pagadores": f"{recall_good:.3f}",
        "F1-Score": f"{f1:.3f}",
        "Taxa de Aprovação": f"{approval_rate:.3f}",
        "Total avaliado": f"{total:,}",
    }

    for k, v in metrics.items():
        st.markdown(
            f"<div style='padding:5px 0; border-bottom:1px solid #2d2d44;'>"
            f"<span style='color:#a0a0b8;'>{k}:</span> "
            f"<strong style='color:#e0e0f0;'>{v}</strong></div>",
            unsafe_allow_html=True,
        )

# ── Imagens salvas ───────────────────────────────────────────────────────────
if (FIGURES_DIR / "confusion_matrix.png").exists() or (FIGURES_DIR / "roc_curve.png").exists():
    st.markdown('<div class="section-title">Figuras Geradas pelo Pipeline de Avaliação</div>', unsafe_allow_html=True)
    fig_imgs = st.columns(3)
    available_figs = [
        ("confusion_matrix.png", "Matriz de Confusão"),
        ("roc_curve.png", "Curva ROC"),
        ("lift_curve.png", "Curva Lift"),
    ]
    for col, (fname, title) in zip(fig_imgs, available_figs):
        img_path = FIGURES_DIR / fname
        if img_path.exists():
            with col:
                st.markdown(f"**{title}**")
                st.image(str(img_path), use_container_width=True)
