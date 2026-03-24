---
name: lightgbm_agent
description: Treina modelo LightGBM para credit scoring com
Stratified K-Fold 5 folds e otimização de hiperparâmetros
---

# LightGBM Agent

## Objetivo
Treinar o modelo principal de credit scoring com gradient
boosting otimizado para dados desbalanceados.

## Input
- data/processed/train_final.parquet

## Outputs
- models/lightgbm_model.pkl
- reports/figures/lightgbm_feature_importance.png
- MLflow run registrado em mlruns/ (experimento "pod-bank-credit-score")

## Passos obrigatórios

### 1. Preparação
- Carregar train_final.parquet
- Separar X e y (TARGET)
- Remover SK_ID_CURR
- LightGBM aceita categóricas nativamente — identificar e
  converter para category dtype

### 2. Hiperparâmetros
Usar estes parâmetros otimizados para crédito desbalanceado:
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': 11.4,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

### 3. Treinamento com Stratified K-Fold
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Usar early_stopping_rounds=50 em cada fold
- Para cada fold calcular: AUC-ROC, KS, Gini
- Guardar oof_predictions (out-of-fold) para análise posterior

### 4. Treinar modelo final
- Treinar com todos os dados usando melhor n_estimators
  (média dos folds)
- Salvar em models/lightgbm_model.pkl

### 5. Feature Importance
- Extrair feature importance por gain
- Gerar barplot horizontal top 30 features
- Salvar em reports/figures/lightgbm_feature_importance.png

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
    run_name=f"lightgbm_v1_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=final_model,
    params=params,
    cv_metrics=cv_metrics,
    feature_importance=fi_df,
)

result = check_targets({k: float(np.mean(v)) for k, v in cv_metrics.items()})
print(f"Status: {result['status']}")
print(f"Run ID: {run_id}")
```

Analisar se variáveis var_* aparecem no top 30 e comentar nos logs/print.
