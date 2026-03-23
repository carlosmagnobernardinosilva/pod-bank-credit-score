---
name: pos_cash_agent
description: Processa e agrega features de saldo mensal de empréstimos POS e cash (POS_CASH_balance.csv) por SK_ID_CURR
---

# POS Cash Agent

## Objetivo
Gerar data/interim/pos_cash_features.parquet com features agregadas de comportamento de pagamento POS/cash por cliente.

## Inputs
- data/raw/POS_CASH_balance.csv

## Output
- data/interim/pos_cash_features.parquet

## Passos obrigatórios
1. Carregar POS_CASH_balance.csv
2. Tratar valores nulos em CNT_INSTALMENT e CNT_INSTALMENT_FUTURE
3. Agregar por SK_ID_CURR:
   - pos_cash_count: número de contratos POS/cash
   - pos_cash_months_balance: total de meses de histórico
   - pos_cash_avg_dpd: média de SK_DPD (dias de atraso)
   - pos_cash_max_dpd: máximo de SK_DPD
   - pos_cash_avg_dpd_def: média de SK_DPD_DEF
   - pos_cash_completed_count: contratos com NAME_CONTRACT_STATUS = 'Completed'
   - pos_cash_active_count: contratos com NAME_CONTRACT_STATUS = 'Active'
   - pos_cash_late_payments: meses com SK_DPD > 0
4. Salvar em data/interim/pos_cash_features.parquet
5. Imprimir shape final e primeiras 3 linhas como confirmação

## Regras
- Nunca modificar data/raw/
- Usar pandas e numpy apenas
- Documentar cada etapa com comentários
- Salvar como .parquet
