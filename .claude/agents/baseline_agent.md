---
name: baseline_agent
description: Treina modelo baseline de Regressão Logística para
credit scoring com Stratified K-Fold 5 folds
---

# Baseline Agent — Logistic Regression

## Objetivo
Estabelecer performance mínima de referência com modelo
interpretável antes de avançar para modelos complexos.

## Input
- data/processed/train_final.parquet

## Outputs
- models/baseline_logistic_regression.pkl
- MLflow run registrado em mlruns/ (experimento "pod-bank-credit-score")

## Passos obrigatórios

### 1. Preparação
- Carregar train_final.parquet
- Separar features (X) e target (y = TARGET)
- Remover SK_ID_CURR do X
- Identificar colunas categóricas restantes e aplicar
  OrdinalEncoder antes do modelo
- Aplicar StandardScaler nas features numéricas

### 2. Treinamento com Stratified K-Fold
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Modelo: LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='saga'
  )
- Para cada fold calcular:
  AUC-ROC, KS Statistic, Gini Coefficient

### 3. Calcular KS e Gini
KS = max(TPR - FPR) da curva ROC
Gini = 2 * AUC - 1

### 4. Treinar modelo final
- Treinar com todos os dados de treino
- Salvar em models/baseline_logistic_regression.pkl

### 5. Registrar no MLflow
```python
from src.models.mlflow_setup import setup_mlflow, log_cv_results, check_targets
from datetime import datetime

setup_mlflow("pod-bank-credit-score")

# cv_metrics: dict com listas de valores por fold
cv_metrics = {
    "auc_roc": [fold_auc_1, fold_auc_2, ...],
    "ks":      [fold_ks_1,  fold_ks_2,  ...],
    "gini":    [fold_gini_1, fold_gini_2, ...],
}

# feature importance: DataFrame com colunas ["feature", "importance"]
# (coeficientes absolutos ordenados decrescente)
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": abs_coefficients
}).sort_values("importance", ascending=False).head(20)

run_id = log_cv_results(
    run_name=f"baseline_lr_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=final_model,
    params=final_model.get_params(),
    cv_metrics=cv_metrics,
    feature_importance=fi_df,
)

result = check_targets({k: float(np.mean(v)) for k, v in cv_metrics.items()})
print(f"Status: {result['status']}")
print(f"Run ID: {run_id}")
```
