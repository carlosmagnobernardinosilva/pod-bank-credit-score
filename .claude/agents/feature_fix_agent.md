---
name: feature_fix_agent
description: Aplica correções avançadas de feature engineering identificadas nos alertas do documentation_agent antes da modelagem
---

# Feature Fix Agent

## Objetivo
Corrigir os alertas pendentes identificados na documentação
e atualizar data/processed/train_final.parquet e test_final.parquet

## Input
- data/processed/train_final.parquet
- data/processed/test_final.parquet

## Output
- data/processed/train_final.parquet (sobrescrito com correções)
- data/processed/test_final.parquet (sobrescrito com correções)
- reports/feature_fix_report.md (relatório das correções)

## Correção 1 — bureau_max_overdue: Flag Binário

Passos:
1. Antes de qualquer imputação, criar feature:
   bureau_had_overdue = 1 se bureau_max_overdue > 0, senão 0
   (NaN também vira 0 — ausência de overdue = nunca atrasou)
2. Manter bureau_max_overdue original com valor 0 para NaN
3. Documentar: quantos clientes têm bureau_had_overdue = 1 vs 0

## Correção 2 — EXT_SOURCE_1: Modelo de Imputação

Passos:
1. Separar registros com EXT_SOURCE_1 presente (treino do imputer)
   e ausente (a imputar)
2. Features preditoras para o modelo de imputação:
   EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH, AMT_CREDIT,
   AMT_INCOME_TOTAL, DAYS_EMPLOYED, NAME_EDUCATION_TYPE
3. Treinar um LightGBM regressor usando os registros com
   EXT_SOURCE_1 presente
4. Predizer EXT_SOURCE_1 para os registros com valor ausente
5. Criar flag ext_source_1_imputed = 1 para registros imputados
6. Aplicar o mesmo imputer treinado no train para o test
   (nunca treinar novamente no test — evitar leakage)
7. Salvar o modelo de imputação em models/imputer_ext_source_1.pkl

## Correção 3 — Colunas de imóvel com >60% de nulos

Remover as seguintes colunas de train e test:
COMMONAREA_AVG, COMMONAREA_MEDI, COMMONAREA_MODE,
NONLIVINGAPARTMENTS_AVG, NONLIVINGAPARTMENTS_MEDI,
NONLIVINGAPARTMENTS_MODE, FONDKAPREMONT_MODE,
LIVINGAPARTMENTS_AVG, LIVINGAPARTMENTS_MEDI,
LIVINGAPARTMENTS_MODE, YEARS_BUILD_AVG,
YEARS_BUILD_MEDI, YEARS_BUILD_MODE

## Correção 4 — Validação de Data Leakage

Verificar em todas as features de dias (DAYS_*):
- Se existem valores positivos inesperados
- Reportar quantos registros seriam afetados
- Não remover ainda — apenas reportar no feature_fix_report.md

## Correção 5 — Seleção por Information Value (IV)

Passos:
1. Calcular IV de todas as features usando WoE
   - Numéricas: binning por quantis (pd.qcut, bins=10)
   - Categóricas: usar as próprias categorias como bins
   - Tratar zeros com smoothing de 0.5 para evitar log(0)

2. Classificar por escala padrão de mercado:
   - IV < 0.02  → REMOVER
   - IV 0.02-0.10 → FRACO
   - IV 0.10-0.30 → MÉDIO
   - IV > 0.30  → FORTE
   - IV > 0.50  → ALERTA de possível leakage

3. Remover features com IV < 0.02
   Manter sempre: SK_ID_CURR, TARGET

4. Alertar no relatório features com IV > 0.50

5. Aplicar mesma seleção no test
   (usar lista de features do train — nunca recalcular no test)

6. Sobrescrever train_final.parquet e test_final.parquet
   com as features selecionadas

7. Adicionar ao feature_fix_report.md:
   - Total de features antes e depois da seleção por IV
   - Lista das removidas
   - Top 20 por IV
   - Alertas de leakage se houver
   - Gráfico: reports/figures/iv_barplot.png

## Relatório Final
Gerar reports/feature_fix_report.md com:
- Shape antes e depois das correções
- Distribuição de bureau_had_overdue
- RMSE do modelo de imputação de EXT_SOURCE_1
- Quantas colunas removidas
- Status do data leakage check
- Lista final de features do dataset
