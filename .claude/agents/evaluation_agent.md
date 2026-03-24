---
name: evaluation_agent
description: Avalia o modelo campeão final com curvas, threshold calibration e análise de erros para Fase 5 CRISP-DM
---

# Evaluation Agent — Fase 5

## Objetivo

Avaliação completa do modelo LightGBM campeão

## Inputs

- data/processed/train_final.parquet
- models/lightgbm_tuned.pkl ou models/lightgbm_model.pkl

## Outputs

- reports/evaluation_report.md
- reports/figures/roc_curve.png
- reports/figures/ks_curve.png
- reports/figures/lift_curve.png
- reports/figures/score_distribution.png
- reports/figures/confusion_matrix.png
- reports/figures/feature_importance_final.png

## Passos obrigatórios

### 1. Preparação

- Carregar train_final.parquet
- Separar 20% como holdout final (random_state=42)
  estratificado pelo TARGET
- Carregar modelo campeão

### 2. Curva ROC

- Plotar curva ROC no holdout
- Marcar ponto de operação ideal
- Calcular e exibir AUC final
- Salvar roc_curve.png

### 3. Curva KS

- Plotar distribuição acumulada de bons e maus por score
- Marcar ponto de KS máximo
- Salvar ks_curve.png

### 4. Curva Lift

- Calcular lift por decil
- Plotar lift acumulado e por decil
- Salvar lift_curve.png

### 5. Calibração de Threshold

Testar thresholds de 0.05 a 0.50 (step 0.01):

Para cada threshold calcular:

- Precision, Recall, F1
- Taxa de aprovação
- Taxa de inadimplência esperada

Eleger threshold que:

- Recall bons pagadores >= 70%
- Minimiza inadimplência esperada

### 6. Distribuição do Score

- Plotar histograma separado por TARGET (0 e 1)
- Mostrar sobreposição e separação
- Marcar threshold eleito
- Salvar score_distribution.png

### 7. Matriz de Confusão

- Calcular com threshold eleito
- Plotar heatmap normalizado
- Calcular custo estimado:
  Falso Negativo (aprovar inadimplente) = peso 5
  Falso Positivo (recusar bom pagador) = peso 1
- Salvar confusion_matrix.png

### 8. Análise de Erros

- Top 10 features dos falsos negativos vs verdadeiros positivos
- Perfil do cliente mais difícil de classificar

### 9. Validação final contra targets do negócio

Verificar cada critério do CLAUDE.md:

- AUC-ROC >= 0.75 → APROVADO/REPROVADO
- KS >= 0.35      → APROVADO/REPROVADO
- Gini >= 0.50    → APROVADO/REPROVADO
- Recall >= 70%   → APROVADO/REPROVADO

### 10. Relatório final

Gerar reports/evaluation_report.md com todos os resultados
e invocar documentation_agent para registrar Fase 5
