---
name: previous_app_agent
description: Processa e agrega features de aplicações anteriores de crédito no PoD Bank (previous_application.csv) por SK_ID_CURR
---

# Previous Application Agent

## Objetivo
Gerar data/interim/prev_app_features.parquet com features agregadas do histórico de aplicações anteriores por cliente.

## Inputs
- data/raw/previous_application.csv

## Output
- data/interim/prev_app_features.parquet

## Passos obrigatórios
1. Carregar previous_application.csv
2. Tratar valor especial 365243 nas colunas de dias:
   DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION,
   DAYS_LAST_DUE, DAYS_TERMINATION → substituir por NaN
3. Tratar categorias especiais XNA e XAP → substituir por NaN
4. Agregar por SK_ID_CURR:
   - prev_app_count: total de aplicações anteriores
   - prev_app_approved_count: aprovadas (NAME_CONTRACT_STATUS='Approved')
   - prev_app_refused_count: recusadas (NAME_CONTRACT_STATUS='Refused')
   - prev_app_approval_rate: taxa de aprovação histórica
   - prev_app_avg_amount: média de AMT_APPLICATION
   - prev_app_avg_credit: média de AMT_CREDIT
   - prev_app_avg_annuity: média de AMT_ANNUITY
   - prev_app_last_days: mínimo de DAYS_DECISION (mais recente)
   - prev_app_consumer_count: contratos do tipo ConsumerLoans
   - prev_app_cash_count: contratos do tipo CashLoans
5. Salvar em data/interim/prev_app_features.parquet
6. Imprimir shape final e primeiras 3 linhas como confirmação

## Regras
- Nunca modificar data/raw/
- Usar pandas e numpy apenas
- Documentar cada etapa com comentários
- Salvar como .parquet
