# Pipeline Report — Production Scoring Pipeline

**Data:** 2026-03-24
**Version:** 1.0-tuned
**Model:** LightGBM v2 Tuned (`models/lightgbm_tuned.pkl`)
**Pipeline artifact:** `models/scoring_pipeline.pkl`

---

## Performance Summary

| Metric | Value | SLA |
|---|---|---|
| Average inference time | **7.76 ms** | < 500 ms |
| Max inference time | **8.02 ms** | < 500 ms |
| SLA status | **PASS** | |

Inference latency is ~64x below the 500 ms production requirement, leaving ample headroom for network overhead and pre/post-processing in a real API layer.

---

## Test Results — 5 Real Clients

Clients were selected from `data/processed/train_final.parquet`:
- 2 good payers (TARGET=0)
- 2 defaulters (TARGET=1)
- 1 borderline case (predicted score 0.40–0.55)

| SK_ID_CURR | TARGET_real | Score | Decision | Risk Band | Inference (ms) |
|---|---|---|---|---|---|
| 204040 | 0 | 0.3727 | APROVADO | MEDIO | 7.84 |
| 151722 | 0 | 0.1087 | APROVADO | BAIXO | 8.02 |
| 349833 | 1 | 0.5225 | REPROVADO | ALTO | 7.98 |
| 121179 | 1 | 0.7526 | REPROVADO | ALTO | 7.62 |
| 367944 | 0 | 0.4676 | APROVADO | MEDIO | 7.36 |

All 4 labeled clients (2 good + 2 bad) were correctly classified. The borderline client (SK_ID_CURR=367944) received score 0.4676, just below the 0.48 threshold — resulting in APROVADO/MEDIO, consistent with their actual TARGET=0.

---

## Decision Rules

| Condition | Decision | Risk Band |
|---|---|---|
| score < 0.20 | APROVADO | BAIXO |
| 0.20 <= score < 0.48 | APROVADO | MEDIO |
| score >= 0.48 | REPROVADO | ALTO |

---

## Top Factors (consistent across all clients)

The top 5 features by model importance (LightGBM gain) are identical across all clients, confirming the model relies on stable global signals:

1. `EXT_SOURCE_2` — External bureau score 2 (importance: 240,284)
2. `EXT_SOURCE_3` — External bureau score 3 (importance: 232,392)
3. `ORGANIZATION_TYPE` — Employer organization type (importance: 164,402)
4. `EXT_SOURCE_1` — External bureau score 1 (importance: 152,567)
5. `credit_term` — Loan duration in months (importance: 88,614)

---

## Usage Instructions

### Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Basic usage

```python
from src.models.predict import predict_score

applicant = {
    "EXT_SOURCE_1": 0.45,
    "EXT_SOURCE_2": 0.60,
    "EXT_SOURCE_3": 0.55,
    "AMT_CREDIT": 270000.0,
    "AMT_ANNUITY": 13500.0,
    "DAYS_BIRTH": -14000,
    "DAYS_EMPLOYED": -2000,
    "CODE_GENDER": "F",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    # ... remaining features (missing ones default to NaN)
}

result = predict_score(applicant)
print(result)
# {
#   'score': 0.17,
#   'decision': 'APROVADO',
#   'risk_band': 'BAIXO',
#   'top_factors': [...],
#   'inference_ms': 7.9
# }
```

### Response schema

| Field | Type | Description |
|---|---|---|
| `score` | float | Probability of default — range [0.0, 1.0] |
| `decision` | str | `APROVADO` or `REPROVADO` |
| `risk_band` | str | `BAIXO`, `MEDIO`, or `ALTO` |
| `top_factors` | list[dict] | Top 5 features with `feature`, `value`, `importance` keys |
| `inference_ms` | float | Wall-clock latency in milliseconds |

### Preprocessing applied automatically

- `DAYS_*` columns: value `365243` (sentinel for missing) is replaced with `NaN`
- Categorical strings: `XNA` and `XAP` values are mapped to `"Unknown"`
- Missing features: columns absent from the input dict are filled with `NaN`
- Column alignment: features are ordered exactly as trained

### Pipeline artifact structure

`models/scoring_pipeline.pkl` is a joblib-serialized dict:

```python
{
    'model'          : LGBMClassifier,  # lightgbm_tuned v2
    'feature_columns': list[str],       # 62 features in order
    'threshold'      : 0.48,
    'version'        : '1.0-tuned',
}
```

### Rebuilding the pipeline

If the model is retrained, rebuild the artifact with:

```bash
python src/models/build_scoring_pipeline.py
```

---

## Files Created

| File | Description |
|---|---|
| `src/models/predict.py` | Main scoring function — `predict_score(applicant_dict)` |
| `src/models/build_scoring_pipeline.py` | Script to serialize model + metadata into pipeline artifact |
| `src/models/test_pipeline.py` | Smoke test against 5 real clients |
| `models/scoring_pipeline.pkl` | Production pipeline artifact (joblib) |
| `reports/_pipeline_test_results.json` | Raw test results (JSON) |
