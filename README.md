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
├── data/                          # Dados (não versionados - ver .gitignore)
│   ├── raw/                       # Dados brutos, exatamente como recebidos
│   └── processed/                 # Dados finais prontos para modelagem
│
├── notebooks/                     # Notebooks Jupyter por fase do CRISP-DM
│   ├── 01_business_understanding/ # Análise do problema e hipóteses de negócio
│   ├── 02_data_understanding/     # EDA, qualidade e distribuição dos dados
│   ├── 03_data_preparation/       # Feature engineering e pré-processamento
│   ├── 04_modeling/               # Treinamento e comparação de modelos
│   ├── 05_evaluation/             # Avaliação de desempenho e análise de erros
│   └── 06_deployment/             # Prototipação do pipeline de produção
│
├── src/                           # Código-fonte modularizado
│   ├── data/                      # Scripts de ingestão e transformação de dados
│   ├── features/                  # Construção e seleção de features
│   ├── models/                    # Treinamento, tuning e serialização de modelos
│   └── evaluation/                # Métricas customizadas e relatórios de performance
│
├── models/                        # Modelos serializados (não versionados)
│
├── reports/                       # Relatórios e visualizações geradas
│   └── figures/                   # Gráficos e imagens exportadas
│
├── tests/                         # Testes unitários do código em src/
│
├── .gitignore
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

Cada fase possui notebooks dedicados em `notebooks/` e código reutilizável em `src/`.

---

*Projeto desenvolvido para a PoD Bank — Inclusão financeira com inteligência de dados.*
