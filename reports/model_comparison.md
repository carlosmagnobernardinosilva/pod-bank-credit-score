# Comparação de Modelos — Fase 4 (Modelagem)

**Data:** 2026-03-24
**Projeto:** PoD Bank Credit Score
**Fase CRISP-DM:** 4 — Modeling
**Critérios de aprovação (revisados):** AUC-ROC >= 0.75 | KS >= 0.35 | Gini >= 0.50

---

## 1. Tabela Comparativa de Métricas (CV 5-fold)

| Modelo | AUC-ROC | KS | Gini | AUC ✓ | KS ✓ | Gini ✓ | Status | MLflow Run ID |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7559 ± 0.0027 | 0.3857 ± 0.0054 | 0.5117 ± 0.0055 | ✅ | ✅ | ✅ | APROVADO | `53e25c53f91b43708d40876fee99d7b7` |
| **LightGBM v1** | **0.7701 ± 0.0031** | **0.4105 ± 0.0035** | **0.5401 ± 0.0062** | ✅ | ✅ | ✅ | **APROVADO** | `32a8dd5fd7ca45cda61f7dbb8d1531d1` |
| XGBoost v1 | 0.7698 ± 0.0036 | 0.4065 ± 0.0083 | 0.5396 ± 0.0071 | ✅ | ✅ | ✅ | APROVADO | `e6baf972d9334ae1b88db1782fff614f` |

> **Critérios mínimos:** AUC-ROC >= 0.75 · KS >= 0.35 · Gini >= 0.50
> Todos os três modelos aprovados. Targets originais do projeto (AUC >= 0.72, KS >= 0.32, Gini >= 0.42) superados com folga.

---

## 2. Análise de Aprovação por Critério

### 2.1 AUC-ROC >= 0.75

| Modelo | Valor | Margem | Aprovado? |
|---|---|---|---|
| Logistic Regression | 0.7559 | +0.0059 | ✅ |
| LightGBM | 0.7701 | +0.0201 | ✅ |
| XGBoost | 0.7698 | +0.0198 | ✅ |

### 2.2 KS >= 0.35

| Modelo | Valor | Margem | Aprovado? |
|---|---|---|---|
| Logistic Regression | 0.3857 | +0.0357 | ✅ |
| LightGBM | 0.4105 | +0.0605 | ✅ |
| XGBoost | 0.4065 | +0.0565 | ✅ |

### 2.3 Gini >= 0.50

| Modelo | Valor | Margem | Aprovado? |
|---|---|---|---|
| Logistic Regression | 0.5117 | +0.0117 | ✅ |
| LightGBM | 0.5401 | +0.0401 | ✅ |
| XGBoost | 0.5396 | +0.0396 | ✅ |

---

## 3. Ganhos Relativos ao Baseline (Logistic Regression)

| Métrica | LightGBM vs. Baseline | XGBoost vs. Baseline | LightGBM vs. XGBoost |
|---|---|---|---|
| AUC-ROC | +0.0142 (+1.88%) | +0.0139 (+1.84%) | +0.0003 (+0.04%) |
| KS | +0.0248 (+6.43%) | +0.0208 (+5.40%) | +0.0040 (+0.98%) |
| Gini | +0.0284 (+5.55%) | +0.0279 (+5.45%) | +0.0005 (+0.09%) |

> LightGBM e XGBoost apresentam ganho expressivo sobre o baseline linear (~6% no KS).
> A diferença entre LightGBM e XGBoost é mínima (~0.04% no AUC), configurando **empate técnico**.

---

## 4. Modelo Eleito

### **LightGBM v1**

**MLflow Run ID:** `32a8dd5fd7ca45cda61f7dbb8d1531d1`
**Arquivo salvo:** `models/lightgbm_model.pkl`
**n_estimators final:** 221 (melhor via early stopping, média dos 5 folds)

### Justificativa

1. **Aprovação em todos os critérios** — AUC-ROC=0.7701, KS=0.4105, Gini=0.5401; todas as três métricas acima dos limiares revisados.
2. **Melhor desempenho absoluto** — LightGBM lidera em todas as três métricas frente ao XGBoost e ao baseline.
3. **Empate técnico com XGBoost** — diferença de apenas 0.0003 no AUC-ROC (< 0.1%); pela regra de desempate definida no escopo desta comparação, **LightGBM é priorizado**.
4. **Eficiência de treinamento** — LightGBM suporta variáveis categóricas nativamente (sem OrdinalEncoder externo), reduzindo risco de encoding inconsistente entre treino e produção; variância do KS menor (±0.0035 vs. ±0.0083 no XGBoost).
5. **Estabilidade** — menor desvio padrão no KS entre folds (0.0035 vs. 0.0083), indicando predições mais consistentes no conjunto de validação.

### Top Features (consistentes entre modelos)

`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `credit_term`, `inst_late_rate`, `ORGANIZATION_TYPE`, `OCCUPATION_TYPE`

---

## 5. Próximos Passos — Fase 5 (Avaliação)

### 5.1 Análise de Curva ROC e KS Plot
- Gerar curva ROC completa no conjunto holdout (test_final.parquet)
- Plotar distribuição de scores separada por TARGET (0 vs. 1)
- Calcular KS estatístico com tabela de decis

### 5.2 Calibração de Threshold
- Mapear threshold vs. Recall/Precision/F1 para o modelo eleito
- Definir threshold operacional considerando o custo assimétrico (falso negativo > falso positivo)
- Garantir Recall (bons pagadores) >= 60% conforme meta do projeto

### 5.3 Explicabilidade (SHAP)
- Calcular SHAP values globais (feature importance via TreeExplainer)
- Gerar waterfall plot para instâncias de alta/baixa probabilidade de default
- Validar coerência com direção das correlações do EDA (EXT_SOURCE = negativa, REGION_RATING = positiva)

### 5.4 Análise de Erros
- Inspecionar falsos positivos e falsos negativos com maior probabilidade de erro
- Verificar se os erros se concentram em subgrupos específicos (p.ex. clientes jovens, sem histórico de bureau)

### 5.5 Relatório de Avaliação Final
- Consolidar métricas, gráficos e SHAP no `reports/evaluation_report.md`
- Documentar threshold operacional escolhido e impacto no negócio
- Atualizar `docs/model_documentation.md` com resultados da Fase 5

---

## 6. Configuração dos Modelos Treinados

| Modelo | Parâmetros Chave | Estratégia de Balanceamento |
|---|---|---|
| Logistic Regression | solver=saga, max_iter=1000, class_weight='balanced' | class_weight='balanced' |
| LightGBM v1 | lr=0.05, num_leaves=31, n_estimators=221 (early stopping) | scale_pos_weight=11.4 |
| XGBoost v1 | lr=0.05, max_depth=6, n_estimators=241 (early stopping) | scale_pos_weight=11.4 |

---

*Gerado em: 2026-03-24*
*Referência MLflow: experiment `pod-bank-credit-score`*
