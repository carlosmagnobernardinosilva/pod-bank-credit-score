# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Architecture

This project follows the **CRISP-DM** framework end-to-end.

### Data Flow

```
data/raw/ → data/interim/ → data/processed/
```

- `data/raw/` — Home Credit dataset (9 CSV files, ~2.8 GB). Never modify these files.
- `data/interim/` — outputs of cleaning and joining scripts
- `data/processed/` — feature-engineered datasets ready for model training

None of the `data/` contents are versioned (see `.gitignore`).

### Key Raw Files

| File | Description |
|---|---|
| `application_train.csv` | Main training table — one row per loan application, target = `TARGET` (1 = defaulted) |
| `application_test.csv` | Test set (no target) |
| `bureau.csv` | Credit history from other institutions |
| `bureau_balance.csv` | Monthly balance of bureau credits |
| `previous_application.csv` | Prior applications at PoD Bank |
| `POS_CASH_balance.csv` | Monthly balance snapshots of POS/cash loans |
| `credit_card_balance.csv` | Monthly credit card statements |
| `installments_payments.csv` | Repayment history for previous loans |
| `HomeCredit_columns_description.csv` | Column descriptions for all tables |

### Code Organization

```
notebooks/          # Exploration and experimentation — one folder per CRISP-DM phase
  01_business_understanding/
  02_data_understanding/
  03_data_preparation/
  04_modeling/
  05_evaluation/
  06_deployment/

src/                # Production-grade, importable modules
  data/             # Loading and joining raw tables, writing to interim/processed
  features/         # Feature engineering and selection logic
  models/           # Training, hyperparameter tuning, serialization
  evaluation/       # Custom metrics (KS, Gini) and performance reports

tests/              # Unit tests for src/ modules (not notebooks)
reports/figures/    # Charts exported from notebooks or src/evaluation
```

Notebooks are for exploration; stable, reusable logic should be extracted into `src/`.

## Success Metrics

All modeling decisions should be evaluated against these targets:

| Metric | Target |
|---|---|
| AUC-ROC | >= 0.72 |
| KS Statistic | >= 0.32 |
| Gini Coefficient | >= 0.42 |
| Recall (good payers) | >= 60% |

Validate against data leakage using temporal splits.

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/) with the following types:

| Type | When to use |
|---|---|
| `feat` | Nova funcionalidade ou etapa do pipeline (ex.: novo feature de bureau) |
| `fix` | Correção de bug em código existente |
| `docs` | Alterações em README, CLAUDE.md ou docstrings |
| `chore` | Tarefas de manutenção sem impacto no código de produção (ex.: atualizar .gitignore) |
| `refactor` | Reestruturação de código sem mudança de comportamento |
| `test` | Adição ou correção de testes em `tests/` |

Formato: `<type>: <descrição em minúsculas no imperativo>`

```
feat: adicionar features de agregação de bureau_balance
fix: corrigir vazamento de dados no split de validação
refactor: extrair lógica de encoding para src/features
```

## Data Understanding — Achados do EDA

> Fonte: `notebooks/02_data_understanding/01_eda_exploratory.ipynb`

### Distribuição do TARGET

| Classe | Descrição | Contagem | % |
|---|---|---|---|
| `0` | Adimplente | 282,686 | 91.93% |
| `1` | Inadimplente | 24,825 | 8.07% |

Razão de desbalanceamento **~11.4:1**. Implicações para modelagem:
- Usar `class_weight='balanced'` ou equivalente em todos os modelos
- Avaliar com AUC-ROC e KS, não acurácia
- Calibrar threshold de decisão separadamente da otimização do modelo

### Qualidade dos Dados

**application_train — colunas com > 30% de nulos (amostra):**

| Coluna | % Nulos | Grupo |
|---|---|---|
| `COMMONAREA_*` (3 colunas) | ~69.9% | Área comum do imóvel |
| `NONLIVINGAPARTMENTS_*` (3) | ~69.4% | Área não residencial |
| `FONDKAPREMONT_MODE` | ~68.4% | Tipo de fundo do imóvel |
| `LIVINGAPARTMENTS_*` (3) | ~68.4% | Área residencial |
| `YEARS_BUILD_*` (3) | ~66.5% | Ano de construção |
| `OWN_CAR_AGE` | ~65.2% | Idade do veículo |
| `EXT_SOURCE_1` | ~56.4% | Score externo de bureau |
| `OCCUPATION_TYPE` | ~31.3% | Ocupação profissional |

**Valores especiais identificados:**
- `365243` em colunas `DAYS_*` de `previous_application.csv` — código para dado ausente, tratar como `NaN` antes de qualquer cálculo de dias
- `XNA` e `XAP` em variáveis categóricas de `previous_application.csv` — categorias de "não disponível / não aplicável", tratar como categoria explícita `'Unknown'` ou `NaN` dependendo da frequência

### Top 10 Features Correlacionadas com TARGET

| # | Feature | Correlação | Direção |
|---|---|---|---|
| 1 | `EXT_SOURCE_3` | -0.178 | Negativa — score alto = menos risco |
| 2 | `EXT_SOURCE_2` | -0.160 | Negativa |
| 3 | `EXT_SOURCE_1` | -0.155 | Negativa |
| 4 | `DAYS_BIRTH` | -0.078 | Negativa — mais velho = menos risco |
| 5 | `REGION_RATING_CLIENT_W_CITY` | +0.061 | Positiva — região pior = mais risco |
| 6 | `REGION_RATING_CLIENT` | +0.059 | Positiva |
| 7 | `DAYS_LAST_PHONE_CHANGE` | +0.055 | Positiva — troca recente = mais risco |
| 8 | `DAYS_ID_PUBLISH` | +0.051 | Positiva — documento velho = mais risco |
| 9 | `REG_CITY_NOT_WORK_CITY` | +0.045 | Positiva |
| 10 | `FLAG_EMP_PHONE` | +0.045 | Positiva |

`EXT_SOURCE_1/2/3` são os sinais mais fortes — priorizar na feature engineering e investigar interações entre eles.

### Granularidade das Tabelas Secundárias

| Tabela | Linhas | SK únicos | Média linhas/cliente | Chave |
|---|---|---|---|---|
| `bureau` | ~1.72M | ~305K | ~5.6 | `SK_ID_CURR` |
| `bureau_balance` | ~27.3M | ~817K | ~33.4 | `SK_ID_BUREAU` |
| `previous_application` | ~1.67M | ~338K | ~4.9 | `SK_ID_CURR` |
| `POS_CASH_balance` | ~10.0M | ~361K | ~27.7 | `SK_ID_CURR` |
| `credit_card_balance` | ~3.84M | ~103K | ~37.3 | `SK_ID_CURR` |
| `installments_payments` | ~13.6M | ~339K | ~40.1 | `SK_ID_CURR` |

Todas as tabelas secundárias exigem **agregação por `SK_ID_CURR`** antes de fazer join com application_train.

### Decisões para Data Preparation

- [ ] Substituir `365243` por `NaN` em todas as colunas `DAYS_*` das tabelas secundárias
- [ ] Unificar `XNA` e `XAP` como categoria `'Unknown'` nas variáveis categóricas
- [ ] Colunas do bloco `*_AVG / *_MEDI / *_MODE` (características do imóvel) com > 60% de nulos — avaliar drop vs. imputação por mediana agrupada
- [ ] `EXT_SOURCE_1` (56% nulo): imputar pela mediana ou usar modelo de imputação dado seu alto poder preditivo
- [ ] Agregar bureau por `SK_ID_CURR`: dias de atraso (mean, max), contagem de créditos ativos, status atual
- [ ] Agregar installments por `SK_ID_CURR`: taxa de pagamento no prazo, atraso médio em dias, razão pago/devido
- [ ] Agregar previous_application por `SK_ID_CURR`: taxa de aprovação histórica, valor médio aprovado vs. solicitado
- [ ] Validar ausência de data leakage: nenhuma informação posterior à data da aplicação pode ser usada como feature

## Key Constraints

- **Explainability required**: model decisions must be justifiable (use SHAP or similar)
- **Inference latency**: < 500ms per applicant in production
- **No data in git**: all `data/` and `models/` output is gitignored
