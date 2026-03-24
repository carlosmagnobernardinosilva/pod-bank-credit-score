# PoD Bank — Credit Score Model

Modelo de score de risco de crédito desenvolvido seguindo o framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

---

## 1. Entendimento do Negócio

### Contexto

A **PoD Bank** é uma startup financeira focada em concessão de crédito para a população desbancarizada e com histórico de crédito escasso ou inexistente — perfil frequentemente rejeitado pelas instituições tradicionais.

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
pod-bank-credit-score/
│
├── .claude/                       # Configurações e agentes de IA (Claude)
│   ├── agents                      
│   └── hooks
│
├── .venv/                         # Ambiente virtual (ignorado pelo Git)
│
├── app/                           # Aplicação web e dashboard
│   ├── pages/                     # Páginas secundárias do dashboard
│   ├── dashboard.py               # Arquivo principal do Streamlit
│   └── requirements_dashboard.txt # Dependências exclusivas do dashboard
│
├── data/                          # Dados do projeto (ignorados pelo Git)
│   ├── interim/                   # Dados intermediários (em transformação)
│   ├── processed/                 # Dados finais prontos para modelagem
│   └── raw/                       # Dados brutos originais
│
├── docs/                          # Documentação adicional do projeto
│
├── mlruns/                        # Logs e rastreamento de experimentos de ML
│
├── models/                        # Modelos e pipelines serializados (.pkl)
│   ├── baseline_logistic_regression.pkl
│   ├── imputer_ext_source_1.pkl
│   ├── lightgbm_model.pkl
│   ├── lightgbm_tuned.pkl
│   ├── scoring_pipeline.pkl
│   └── xgboost_model.pkl
│
├── notebooks/                     # Notebooks Jupyter de exploração e modelagem
│   ├── 02_data_understanding/     # EDA e análise de qualidade dos dados
│   └── 04_modeling/               # Treinamento e avaliação
│
├── reports/                       # Relatórios de performance e análises
│   ├── figures/                   # Gráficos e imagens exportadas
│   ├── pipeline_test_results.json # Resultados dos testes de pipeline
│   └── *.md                       # Relatórios detalhados (avaliação, métricas, etc.)
│
├── src/                           # Código-fonte modularizado em Python
│
├── tests/                         # Scripts de testes automatizados
│
├── .gitignore                     # Regras de exclusão do Git
└── activate.sh                    # Script de ativação do ambiente
├── CLAUDE.md                      # Contexto do projeto
├── mlflow.db                      
├── requirements.txt               # Dependências do projeto
└── README.md
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
git clone <url-do-repositorio>
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



