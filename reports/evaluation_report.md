# Relatorio de Avaliacao do Modelo Campiao

**Modelo:** LightGBM v2 Tuned  
**Data:** 2026-03-24 14:50  
**Holdout:** 20% estratificado (random_state=42)  
**MLflow Run:** `624246ba0cc74d8a877e8eb32a53b7d9`

---

## 1. Metricas no Holdout

| Metrica | Valor | Target Fase 5 | Status |
|---------|-------|---------------|--------|
| AUC-ROC | 0.8223 | >= 0.75 | APROVADO |
| KS Statistic | 0.4887 | >= 0.35 | APROVADO |
| Gini | 0.6447 | >= 0.50 | APROVADO |
| Recall bons pagadores | 0.7057 | >= 70% | APROVADO |

**Status Geral: APROVADO**

---

## 2. Comparacao v1 vs v2 (CV 5-fold)

| Modelo | AUC-ROC | KS | Gini | KS gap |
|--------|---------|----|------|--------|
| LightGBM v1 | 0.7701 | 0.4105 | 0.5401 | ~0.10+ |
| LightGBM v2 Tuned | 0.7717 | 0.4118 | 0.5434 | 0.0873 |

v2 manteve performance e reduziu overfitting (KS gap: 0.10+ -> 0.0873).

---

## 3. Calibracao de Threshold

**Threshold eleito:** `0.48`

| Metrica | Valor |
|---------|-------|
| Recall bons pagadores | 0.7057 |
| Recall maus pagadores | 0.7794 |
| Precisao | 0.1890 |
| F1 | 0.3042 |
| Taxa de aprovacao | 0.6664 |
| Default esperado (aprovados) | 0.0268 |

---

## 4. Matriz de Confusao

Threshold: `0.48`

| | Pred: Adimplente | Pred: Inadimplente |
|--|-----------------|-------------------|
| **Real: Adimplente** | 27,924 | 11,646 |
| **Real: Inadimplente** | 768 | 2,714 |

**Custo estimado** (FN x5 + FP x1): **15,486**

---

## 5. Top 10 Features

| # | Feature | Importancia (Gain) |
|---|---------|-------------------|
| 1 | `EXT_SOURCE_2` | 240,284 |
| 2 | `EXT_SOURCE_3` | 232,392 |
| 3 | `ORGANIZATION_TYPE` | 164,402 |
| 4 | `EXT_SOURCE_1` | 152,567 |
| 5 | `credit_term` | 88,615 |
| 6 | `OCCUPATION_TYPE` | 53,840 |
| 7 | `bureau_avg_days_credit` | 47,918 |
| 8 | `inst_late_rate` | 46,696 |
| 9 | `AMT_ANNUITY` | 35,631 |
| 10 | `AMT_GOODS_PRICE` | 27,727 |

---

## 6. Analise de Erros — Falsos Negativos vs Verdadeiros Positivos

Top 10 features com maior diferenca media entre FN (inadimplentes nao detectados) e TP:

| Feature | Media FN | Media TP | Diferenca |
|---------|----------|----------|-----------|
| `AMT_CREDIT` | 652188.7852 | 528247.7907 | 123940.9944 |
| `AMT_GOODS_PRICE` | 582902.3438 | 460600.9123 | 122301.4314 |
| `bureau_avg_credit_sum` | 314091.5016 | 247184.3720 | 66907.1296 |
| `cc_avg_balance` | 16098.8117 | 33409.7551 | 17310.9434 |
| `cc_avg_drawings` | 2953.6895 | 6113.0549 | 3159.3654 |
| `DAYS_BIRTH` | -16595.4362 | -14427.6794 | 2167.7568 |
| `AMT_ANNUITY` | 27981.4570 | 26290.7741 | 1690.6829 |
| `DAYS_REGISTRATION` | -5258.5482 | -4265.1212 | 993.4270 |
| `DAYS_EMPLOYED` | -2261.0273 | -1624.3427 | 636.6847 |
| `DAYS_ID_PUBLISH` | -3124.0794 | -2597.5851 | 526.4943 |

---

## 7. Figuras Geradas

- `reports/figures/roc_curve.png`
- `reports/figures/ks_curve.png`
- `reports/figures/lift_curve.png`
- `reports/figures/score_distribution.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance_final.png`

---
*Gerado por `src/evaluation/evaluate_champion.py`*