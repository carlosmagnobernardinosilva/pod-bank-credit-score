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

## Key Constraints

- **Explainability required**: model decisions must be justifiable (use SHAP or similar)
- **Inference latency**: < 500ms per applicant in production
- **No data in git**: all `data/` and `models/` output is gitignored
