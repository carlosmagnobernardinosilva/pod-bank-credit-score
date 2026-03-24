# PoD Bank — Credit Score Model

Modelo de score de risco de crédito desenvolvido seguindo o framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

### Dados disponiveis em (https://www.kaggle.com/competitions/pod-academy-analise-de-credito-para-fintech/data)

### Resumo Resultados Finais: (https://pod-bank-credit-score.streamlit.app/simulador)
---

## 1. Entendimento do Negócio

### Contexto

A **PoD Bank** é uma startup financeira focada em concessão de crédito para a população desbancarizada e com histórico de crédito escasso ou inexistente, perfil frequentemente rejeitado pelas instituições tradicionais.

Sem acesso a dados robustos de bureaus de crédito convencionais, a empresa precisa desenvolver sua própria capacidade analítica para distinguir bons e maus pagadores, viabilizando uma operação de crédito sustentável e inclusiva.

### Problema de Negócio

A PoD Bank enfrenta dois riscos simultâneos e opostos:

| Risco | Descrição | Impacto |
|---|---|---|
| **Risco de Crédito** | Conceder crédito a clientes com alta probabilidade de inadimplência | Perda financeira direta |
| **Risco de Exclusão** | Negar crédito a clientes que pagariam corretamente | Perda de receita e missão social comprometida |

A ausência de um modelo quantitativo força decisões subjetivas e inconsistentes, tornando a operação incapaz de escalar com segurança.

### Objetivo do Modelo

Desenvolver um **modelo de credit score** que estime a probabilidade de inadimplência de um solicitante de crédito, permitindo:

- Automatizar e padronizar decisões de concessão de crédito
- Calibrar limites de crédito de acordo com o perfil de risco de cada cliente
- Reduzir perdas por inadimplência sem aumentar a taxa de rejeição de bons pagadores
- Gerar uma pontuação interpretável que possa ser auditada e justificada

### Critérios de Sucesso do Projeto

#### Critérios de Negócio
- Reduzir a taxa de inadimplência em pelo menos **15%** em relação à política atual de concessão
- Manter a taxa de aprovação de bons pagadores acima de **70%** (recall da classe positiva)
- O modelo deve ser explicável o suficiente para cumprir requisitos regulatórios de crédito (ex.: justificativa de recusa)

#### Critérios Técnicos (Modelagem)
- **AUC-ROC** >= 0.75 no conjunto de teste
- **KS Statistic** >= 0.35 (poder discriminante entre bons e maus pagadores)
- **Gini Coefficient** >= 0.50
- Ausência de data leakage confirmada por validação temporal

#### Critérios de Implantação
- Pipeline de scoring reproduzível e versionado
- Tempo de inferência por cliente abaixo de 500ms
- Documentação suficiente para handoff ao time de engenharia

---

## 2. Estrutura do Projeto

```
  ╔══════════════════════════════════════════════════════════════════════════════════════╗                                                                                                                  ║                               POD BANK CREDIT SCORE — ARQUITETURA COMPLETA                     
                            Framework: CRISP-DM End-to-End                                                                                                                                            
  ╚══════════════════════════════════════════════════════════════════════════════════════╝

   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │  FASE 1 & 2 — BUSINESS UNDERSTANDING / DATA UNDERSTANDING                       │
   │                                                                                 │
   │  notebooks/02_data_understanding/01_eda_exploratory.ipynb                       │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
   │  │ TARGET   │  │ Nulos    │  │ Top 10   │  │ Granu-   │                         │
   │  │ 8.07%    │  │ >30%     │  │ Features │  │ laridade │                         │
   │  │ default  │  │ mapeados │  │ (EXT_SRC)│  │ tabelas  │                         │
   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘                         │
   └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
  ╔═════════════════════════════════════════════════════════════════════════════════════╗
  ║  DATA RAW — data/raw/  (9 CSVs, ~2.8 GB, somente leitura)                           ║
  ║                                                                                     ║
  ║  application_train.csv  ──────────────────────────────────────────► TARGET          ║
  ║  application_test.csv                                                               ║
  ║  bureau.csv + bureau_balance.csv                                                    ║
  ║  previous_application.csv                                                           ║
  ║  POS_CASH_balance.csv                                                               ║
  ║  credit_card_balance.csv                                                            ║
  ║  installments_payments.csv                                                          ║
  ╚═════════════════════════════════════════════════════════════════════════════════════╝
           │              │              │              │              │
           ▼              ▼              ▼              ▼              ▼
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │  FASE 3 — DATA PREPARATION   src/data/   +   Multi-Agent Claude Orchestration   │
   │                                                                                 │
   │  ┌─────────────────┐   ┌───────────────┐   ┌──────────────────┐                 │
   │  │application_clean│   │bureau_features│   │prev_app_features │                 │
   │  │      .py        │   │     .py       │   │      .py         │                 │ 
   │  │ • 365243→NaN    │   │ • agg days    │   │ • taxa aprovação │                 │
   │  │ • XNA→Unknown   │   │   overdue     │   │ • amt_req/apprvd │                 │
   │  │ • derivar feats │   │ • cnt crédito │   │ • dias até dec.  │                 │
   │  │ (307,511×178)   │   │ (305,811×10)  │   │ (338,857×11)     │                 │
   │  └─────────────────┘   └───────────────┘   └──────────────────┘                 │
   │                                                                                 │
   │  ┌─────────────────┐   ┌───────────────┐   ┌──────────────────┐                 │
   │  │pos_cash_features│   │credit_card_   │   │installments_     │                 │
   │  │      .py        │   │features.py    │   │features.py       │                 │
   │  │ • status ativo  │   │ • utilização  │   │ • late_rate      │                 │
   │  │ • dpd médio     │   │ • pagto mín   │   │ • dias atraso    │                 │
   │  │ (337,252×9)     │   │ (103,558×11)  │   │ (339,587×10)     │                 │
   │  └─────────────────┘   └───────────────┘   └──────────────────┘                 │
   │                                                                                 │
   │  src/data/feature_fix_pipeline.py   src/features/leakage_check.py               │
   │  └── correções avançadas            └── validação anti-leakage                  │
   │                                                                                 │
   │  src/data/merge_final.py                                                        │
   │  └── JOIN por SK_ID_CURR → train_final.parquet (215,257×223)                    │
   │                              test_final.parquet  ( 92,254×223)                  │
   └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                           data/interim/*.parquet
                           data/processed/*.parquet
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │  FASE 4 — MODELING   src/models/                                                │
   │                                                                                 │
   │  ┌──────────────────────────────────────────────────────────────────────────┐   │
   │  │  Pipeline sklearn (OrdinalEncoder → StandardScaler)                      │   │
   │  │  CV 5-fold Estratificado  |  class_weight='balanced'                     │   │
   │  └──────────────────────────────────────────────────────────────────────────┘   │
   │                                                                                 │
   │  task1_baseline_lr.py          task2_lightgbm.py       task3_xgboost.py         │
   │  ┌──────────────────┐         ┌─────────────────┐     ┌──────────────────┐      │
   │  │ Logistic         │         │ LightGBM v1     │     │ XGBoost v1       │      │
   │  │ Regression       │         │ n_est=221       │     │ n_est=241        │      │
   │  │                  │         │ (early stop)    │     │ (early stop)     │      │
   │  │ AUC  0.7559      │         │ AUC  0.7701     │     │ AUC  0.7698      │      │
   │  │ KS   0.3857      │         │ KS   0.4105     │     │ KS   0.4065      │      │
   │  │ Gini 0.5117      │         │ Gini 0.5401     │     │ Gini 0.5396      │      │
   │  │ ✅ APROVADO      │         │ ✅ APROVADO     │     │ ✅ APROVADO      │     │
   │  └──────────────────┘         └─────────────────┘     └──────────────────┘      │
   │                                        │                                        │
   │                                   CHAMPION ◄─── task4_lightgbm_tuned.py         │
   │                                                                                 │
   │  MLflow Tracking (mlflow.db + mlruns/)                                          │
   │  └── Run IDs, métricas, hiperparâmetros, feature importance, artefatos          │
   └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                   models/*.pkl
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │  FASE 5 — EVALUATION   src/evaluation/                                          │
   │                                                                                 │
   │  evaluate_champion.py                                                           │
   │  ├── AUC-ROC curve          reports/figures/roc_curve.png                       │
   │  ├── KS curve               reports/figures/ks_curve.png                        │
   │  ├── Lift curve             reports/figures/lift_curve.png                      │
   │  ├── Score distribution     reports/figures/score_distribution.png              │
   │  ├── Confusion matrix       reports/figures/confusion_matrix.png                │
   │  ├── Feature importance     reports/figures/feature_importance_final.png        │
   │  └── Relatório              reports/evaluation_report.md                        │
   │                                                                                 │
   │  Targets atingidos:  AUC ≥ 0.72 ✅  KS ≥ 0.32 ✅  Gini ≥ 0.42 ✅              │
   └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │  FASE 6 — DEPLOYMENT   app/                                                     │
   │                                                                                 │
   │  dashboard.py  (Streamlit — entrada principal)                                  │
   │  ├── pages/01_modelo.py      ── desempenho do modelo, métricas, gráficos        │
   │  ├── pages/02_simulador.py   ── simulação individual de crédito (< 500ms)       │
   │  └── pages/03_carteira.py    ── análise de portfólio / scoring em lote          │
   │                                                                                 │
   │  src/models/predict.py                                                          │
   │  └── pipeline de inferência: carrega .pkl → aplica → retorna probabilidade      │
   │                                                                                 │
   │  src/models/build_scoring_pipeline.py                                           │
   │  └── serializa pipeline completo (pré-proc + modelo) para produção              │
   └─────────────────────────────────────────────────────────────────────────────────┘

  ══════════════════════════════════════════════════════════════════════════════════════
   CAMADA TRANSVERSAL — ORQUESTRAÇÃO MULTI-AGENTE (Claude Code)
  ══════════════════════════════════════════════════════════════════════════════════════

   .claude/agents/
   ┌───────────────────┐
   │ orchestrator_agent│ ── coordena todos os agentes abaixo
   └─────────┬─────────┘
             │
       ┌─────┴──────────────────────────────────────────────────────┐
       │                                                             │
       ▼                                                             ▼
   Agentes de Dados                                          Agentes de Modelo
   ├── application_agent    (limpa application_train/test)   ├── baseline_agent   (LR)
   ├── bureau_agent         (agrega bureau + bureau_balance)  ├── lightgbm_agent   (LGBM)
   ├── previous_app_agent   (agrega previous_application)    ├── xgboost_agent    (XGB)
   ├── pos_cash_agent        (agrega POS_CASH_balance)        └── evaluation_agent (métricas)
   ├── credit_card_agent    (agrega credit_card_balance)
   ├── installments_agent   (agrega installments_payments)    Agentes de Suporte
   └── feature_fix_agent    (correções avançadas)             ├── documentation_agent
                                                              ├── pipeline_agent
                                                              └── dashboard_agent

   Hooks (.claude/hooks/):
   ├── pre_tool_use.sh   ── validações antes de cada tool call
   └── post_tool_use.sh  ── logging/auditoria após cada tool call

  ══════════════════════════════════════════════════════════════════════════════════════
   RASTREABILIDADE & DOCUMENTAÇÃO
  ══════════════════════════════════════════════════════════════════════════════════════

   reports/                                  docs/
   ├── evaluation_report.md                  ├── data_dictionary.md
   ├── model_comparison.md                   └── model_documentation.md
   ├── leakage_check_report.md
   ├── feature_fix_report.md
   ├── pipeline_report.md
   └── figures/ (18 gráficos)

  ══════════════════════════════════════════════════════════════════════════════════════
   FLUXO DE DADOS RESUMIDO
  ══════════════════════════════════════════════════════════════════════════════════════

   data/raw/*.csv
        │
        ├──[src/data/*]──► data/interim/*.parquet   (limpeza + agregação por tabela)
        │
        └──[merge_final]──► data/processed/train_final.parquet  (215K × 223 features)
                                           test_final.parquet   ( 92K × 223 features)
                                                 │
                                      [src/models/task*.py]
                                                 │
                                           models/*.pkl
                                                 │
                                ┌────────────────┼────────────────┐
                                │                │                │
                           mlruns/          reports/          app/
                           (tracking)       (avaliação)       (dashboard)

  ---
  Resumo das camadas:

  ┌───────────────┬─────────────────────────────┬────────────────────────────────────┐
  │    Camada     │         Tecnologia          │             Propósito              │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Ingestão      │ pandas / parquet            │ Leitura dos 9 CSVs raw             │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Preparação    │ sklearn Pipeline            │ Limpeza, encoding, agregação       │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Feature Store │ parquet (interim/processed) │ 223 features finais                │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Modelagem     │ LightGBM, XGBoost, LR       │ Predição de default                │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Rastreamento  │ MLflow (SQLite)             │ Experimentos e artefatos           │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Avaliação     │ KS, AUC, Gini, SHAP         │ Validação de performance           │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Serving       │ Streamlit                   │ Dashboard + inferência < 500ms     │
  ├───────────────┼─────────────────────────────┼────────────────────────────────────┤
  │ Orquestração  │ Claude Code (15 agentes)    │ Automação multi-agente do pipeline │
  └───────────────┴─────────────────────────────┴────────────────────────────────────┘
```

---

## 3. Tecnologias

- **Python 3.10+**
- **Pandas / NumPy** — manipulação de dados
- **Scikit-learn** — modelagem e avaliação
- **LightGBM / XGBoost** — modelos de gradient boosting
- **SHAP** — explicabilidade do modelo
- **Matplotlib / Seaborn** — visualização

---

## 4. Como Executar

```bash
# Clone o repositório
git clone <https://github.com/carlosmagnobernardinosilva/pod-bank-credit-score/tree/main>
cd pod-bank-credit-score

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

---

## 5. Framework CRISP-DM

Este projeto segue as 6 fases do CRISP-DM:

```
Business Understanding → Data Understanding → Data Preparation
        ↑                                              ↓
   Deployment          ←      Evaluation      ←   Modeling
```
---
Link Resultados finais StreamLit: https://pod-bank-credit-score.streamlit.app/simulador




