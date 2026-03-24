---
name: xgboost_agent
description: Treina modelo XGBoost challenger para credit scoring
com Stratified K-Fold 5 folds
---

# XGBoost Agent — Challenger

## Objetivo
Treinar modelo challenger para comparação com LightGBM e
validação da robustez dos resultados.

## Input
- data/processed/train_final.parquet

## Outputs
- models/xgboost_model.pkl
- reports/figures/xgboost_feature_importance.png
- MLflow run registrado em mlruns/ (experimento "pod-bank-credit-score")

## Passos obrigatórios

### 1. Preparação
- Carregar train_final.parquet
- Separar X e y (TARGET)
- Remover SK_ID_CURR
- Aplicar OrdinalEncoder nas categóricas
  (XGBoost não aceita strings nativamente)

### 2. Hiperparâmetros
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': 11.4,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

### 3. Treinamento com Stratified K-Fold
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Usar early_stopping_rounds=50 em cada fold
- Para cada fold calcular: AUC-ROC, KS, Gini
- Guardar oof_predictions para análise posterior

### 4. Treinar modelo final
- Treinar com todos os dados
- Salvar em models/xgboost_model.pkl

### 5. Feature Importance
- Extrair feature importance por gain
- Gerar barplot horizontal top 30 features
- Salvar em reports/figures/xgboost_feature_importance.png

### 6. Registrar no MLflow
```python
from src.models.mlflow_setup import setup_mlflow, log_cv_results, check_targets
from datetime import datetime

setup_mlflow("pod-bank-credit-score")

cv_metrics = {
    "auc_roc": [fold_auc_1, fold_auc_2, ...],
    "ks":      [fold_ks_1,  fold_ks_2,  ...],
    "gini":    [fold_gini_1, fold_gini_2, ...],
}

# feature importance: gain médio entre os folds
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_gain_importances
}).sort_values("importance", ascending=False).head(30)

run_id = log_cv_results(
    run_name=f"xgboost_v1_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=final_model,
    params=params,
    cv_metrics=cv_metrics,
    feature_importance=fi_df,
)

result = check_targets({k: float(np.mean(v)) for k, v in cv_metrics.items()})
print(f"Status: {result['status']}")
print(f"Run ID: {run_id}")
```
