"""
02_simulador.py — PoD Bank Credit Score — Simulador de Score Individual
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="PoD Bank — Simulador",
    page_icon="🔮",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = PROJECT_ROOT / "models" / "scoring_pipeline.pkl"
FALLBACK_PATH = PROJECT_ROOT / "models" / "lightgbm_tuned.pkl"
THRESHOLD = 0.48

# ── Carregar modelo ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo...")
def load_pipeline():
    """
    Carrega scoring_pipeline.pkl (ou lightgbm_tuned.pkl como fallback).
    Extrai o mapeamento de colunas categóricas do booster para garantir
    compatibilidade exata com o treinamento.
    """
    if PIPELINE_PATH.exists():
        artifact = joblib.load(PIPELINE_PATH)
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        source = f"scoring_pipeline.pkl (v{artifact.get('version', '?')})"
    elif FALLBACK_PATH.exists():
        model = joblib.load(FALLBACK_PATH)
        try:
            feature_columns = model.booster_.feature_name()
        except AttributeError:
            feature_columns = [f"f{i}" for i in range(model.n_features_in_)]
        source = "lightgbm_tuned.pkl (fallback)"
    else:
        return None, None, None, "Nenhum modelo encontrado em models/"

    # Extrair mapeamento categórico: {coluna: [categorias]} na ordem do treino
    # O booster armazena pandas_categorical em ordem das colunas categóricas
    try:
        dump = model.booster_.dump_model()
        pandas_categorical = dump.get("pandas_categorical", [])
        # Identificar quais feature_columns são categóricas lendo uma linha do parquet
        df_ref = pd.read_parquet(
            PROJECT_ROOT / "data" / "processed" / "train_final.parquet"
        ).head(1)
        cat_cols_ordered = [
            c for c in feature_columns
            if c in df_ref.columns and str(df_ref[c].dtype) in ("object", "category")
        ]
        cat_map = dict(zip(cat_cols_ordered, pandas_categorical))
    except Exception:
        cat_map = {}

    return model, feature_columns, cat_map, source


model, feature_columns, cat_map, model_source = load_pipeline()

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .section-title {
        font-size: 1.2rem; font-weight: 600; color: #c0c0e0;
        margin: 24px 0 10px 0; border-left: 4px solid #2ecc71; padding-left: 10px;
    }
    .decision-approved {
        background: #1a3d2e; border: 2px solid #2ecc71; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .decision-rejected {
        background: #3d1a1a; border: 2px solid #e74c3c; border-radius: 12px;
        padding: 24px; text-align: center;
    }
    .decision-text { font-size: 2.6rem; font-weight: 800; letter-spacing: 0.06em; }
    .score-line { color: #a0a0b8; margin-top: 8px; font-size: 0.95rem; }
    .risk-badge {
        display: inline-block; font-size: 0.85rem; font-weight: 700;
        padding: 4px 16px; border-radius: 20px; margin-top: 10px;
    }
    .risk-baixo  { background: #2ecc71; color: #fff; }
    .risk-medio  { background: #f39c12; color: #fff; }
    .risk-alto   { background: #e74c3c; color: #fff; }
    .factor-row  { padding: 6px 0; border-bottom: 1px solid #2d2d44; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🔮 Simulador de Score Individual")
st.markdown(
    "<span style='color:#a0a0b8;'>Preencha os dados do solicitante e clique em "
    "<strong>Calcular Score</strong> para obter a decisão de crédito.</span>",
    unsafe_allow_html=True,
)

if model is None:
    st.error(f"Modelo não disponível: {model_source}")
    st.stop()

st.caption(f"Modelo carregado: {model_source} | Threshold: {THRESHOLD}")
st.markdown("---")

# ── Formulário ────────────────────────────────────────────────────────────────
with st.form("scoring_form"):
    st.markdown('<div class="section-title">Dados do Solicitante</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informações Financeiras**")
        amt_credit = st.number_input(
            "Valor do Crédito (R$)", min_value=10_000.0, max_value=5_000_000.0,
            value=500_000.0, step=10_000.0, format="%.0f",
        )
        credit_term = st.slider(
            "Prazo do Contrato (meses)", min_value=6, max_value=360, value=60,
        )

    with col2:
        st.markdown("**Perfil Pessoal**")
        age_years = st.slider("Idade (anos)", min_value=18, max_value=75, value=35)
        code_gender = st.selectbox("Gênero", options=["M", "F"])
        name_education = st.selectbox(
            "Nível de Educação",
            options=[
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ],
        )

    with col3:
        st.markdown("**Scores Externos de Bureau**")
        ext_source_2 = st.slider(
            "EXT_SOURCE_2", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        )
        ext_source_3 = st.slider(
            "EXT_SOURCE_3", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        )
        st.markdown(
            "<span style='color:#a0a0b8;font-size:0.82rem;'>"
            "Scores próximos de 1.0 indicam menor risco histórico de crédito."
            "</span>",
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button(
        "🔮 Calcular Score",
        use_container_width=True,
        type="primary",
    )

# ── Inferência ────────────────────────────────────────────────────────────────
if submitted:
    # 1. Montar dicionário com os valores do formulário
    raw = {
        "AMT_CREDIT": amt_credit,
        "credit_term": float(credit_term),
        "DAYS_BIRTH": -age_years * 365,
        "age_years": float(age_years),
        "CODE_GENDER": code_gender,
        "NAME_EDUCATION_TYPE": name_education,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
    }

    # 2. Construir DataFrame alinhado às colunas do modelo
    df_input = pd.DataFrame([raw])
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = np.nan
    df_input = df_input[feature_columns]

    # Aplicar dtype category com as categorias exatas do treinamento
    # (necessário para compatibilidade com o LightGBM serializado)
    for col, cats in cat_map.items():
        if col in df_input.columns:
            df_input[col] = pd.Categorical(df_input[col], categories=cats)

    # 3. Predição
    t0 = time.perf_counter()
    score = float(model.predict_proba(df_input)[0, 1])
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Decisão e banda de risco
    decision = "APROVADO" if score < THRESHOLD else "REPROVADO"
    if score < 0.20:
        risk_band = "BAIXO"
    elif score < THRESHOLD:
        risk_band = "MEDIO"
    else:
        risk_band = "ALTO"

    # Top 5 features por importância do modelo
    try:
        importances = model.booster_.feature_importance(importance_type="gain")
    except AttributeError:
        importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_columns).nlargest(5)
    top_factors = [
        {"feature": feat, "importance": float(imp),
         "value": df_input[feat].iloc[0]}
        for feat, imp in fi.items()
    ]

    # ── Exibição dos resultados ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Resultado da Avaliação</div>', unsafe_allow_html=True)

    res_col1, res_col2, res_col3 = st.columns([2, 2, 3])

    # Gauge chart
    with res_col1:
        bar_color = "#e74c3c" if score >= THRESHOLD else ("#f39c12" if score >= 0.20 else "#2ecc71")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f", "font": {"size": 36, "color": "#e0e0f0"}},
            title={"text": "P(inadimplência)", "font": {"size": 13, "color": "#a0a0b8"}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#555"},
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "#1e1e2e",
                "borderwidth": 0,
                "steps": [
                    {"range": [0.00, 0.20], "color": "#1a3d2e"},
                    {"range": [0.20, 0.48], "color": "#3d3000"},
                    {"range": [0.48, 1.00], "color": "#3d1a1a"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.85,
                    "value": THRESHOLD,
                },
            },
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=40, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"Inferência: {inference_ms:.1f} ms | Threshold: {THRESHOLD}")

    # Decisão e resumo
    with res_col2:
        decision_class = "decision-approved" if decision == "APROVADO" else "decision-rejected"
        decision_color = "#2ecc71" if decision == "APROVADO" else "#e74c3c"
        risk_css = {"BAIXO": "risk-baixo", "MEDIO": "risk-medio", "ALTO": "risk-alto"}[risk_band]
        risk_label = {"BAIXO": "Risco BAIXO", "MEDIO": "Risco MÉDIO", "ALTO": "Risco ALTO"}[risk_band]

        st.markdown(
            f"""
            <div class="{decision_class}">
                <div class="decision-text" style="color:{decision_color};">{decision}</div>
                <div class="score-line">
                    Score: <strong style="color:#e0e0f0;">{score:.4f}</strong>
                    &nbsp;|&nbsp; Threshold: <strong style="color:#e0e0f0;">{THRESHOLD}</strong>
                </div>
                <span class="risk-badge {risk_css}">{risk_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Dados informados**")
        summary = {
            "Crédito Solicitado": f"R$ {amt_credit:,.0f}",
            "Prazo": f"{credit_term} meses",
            "Idade": f"{age_years} anos",
            "Gênero": code_gender,
            "Educação": name_education[:28] + ("…" if len(name_education) > 28 else ""),
            "EXT_SOURCE_2": f"{ext_source_2:.2f}",
            "EXT_SOURCE_3": f"{ext_source_3:.2f}",
        }
        for k, v in summary.items():
            st.markdown(
                f"<div class='factor-row'><span style='color:#a0a0b8;'>{k}:</span> <strong>{v}</strong></div>",
                unsafe_allow_html=True,
            )

    # Top 5 fatores
    with res_col3:
        st.markdown("**Top 5 Fatores — Importância do Modelo**")
        feat_names = [f["feature"] for f in top_factors]
        feat_imps = [f["importance"] for f in top_factors]
        feat_vals = [f["value"] for f in top_factors]
        max_imp = max(feat_imps) if feat_imps else 1.0
        normalized = [v / max_imp for v in feat_imps]

        fig_factors = go.Figure(go.Bar(
            x=normalized[::-1],
            y=feat_names[::-1],
            orientation="h",
            marker_color=bar_color,
            text=[
                f"{v:.3f}" if isinstance(v, float) and not pd.isna(v)
                else (str(v) if v is not None else "—")
                for v in feat_vals[::-1]
            ],
            textposition="outside",
            hovertext=[f"Gain: {imp:,.0f}" for imp in feat_imps[::-1]],
        ))
        fig_factors.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=90, t=10, b=10),
            xaxis=dict(title="Importância Relativa", range=[0, 1.35]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_factors, use_container_width=True)

    # Legenda
    st.markdown("---")
    st.markdown(
        """
        <div style="display:flex; gap:20px; align-items:center; font-size:0.84rem; flex-wrap:wrap;">
            <span style="color:#a0a0b8;">Zonas de risco:</span>
            <span style="background:#1a3d2e; color:#2ecc71; padding:3px 12px; border-radius:8px; border:1px solid #2ecc71;">
                🟢 BAIXO &lt; 0.20
            </span>
            <span style="background:#3d3000; color:#f39c12; padding:3px 12px; border-radius:8px; border:1px solid #f39c12;">
                🟡 MÉDIO 0.20 – 0.48
            </span>
            <span style="background:#3d1a1a; color:#e74c3c; padding:3px 12px; border-radius:8px; border:1px solid #e74c3c;">
                🔴 ALTO ≥ 0.48
            </span>
            <span style="color:#a0a0b8;">| Linha branca = threshold de decisão (0.48)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
