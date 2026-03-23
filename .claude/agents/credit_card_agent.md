---
name: credit_card_agent
description: Processa e agrega features de comportamento de cartão de crédito (credit_card_balance.csv) por SK_ID_CURR
---

# Credit Card Agent

## Objetivo
Gerar data/interim/credit_card_features.parquet com features agregadas de uso e pagamento de cartão de crédito por cliente.

## Inputs
- data/raw/credit_card_balance.csv

## Output
- data/interim/credit_card_features.parquet

## Passos obrigatórios
1. Carregar credit_card_balance.csv
2. Tratar valores nulos em AMT_BALANCE, AMT_CREDIT_LIMIT_ACTUAL,
   AMT_DRAWINGS_CURRENT, AMT_PAYMENT_CURRENT
3. Calcular feature de utilização do limite:
   cc_utilization = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
   (substituir inf e NaN por 0)
4. Agregar por SK_ID_CURR:
   - cc_count: número de cartões
   - cc_avg_balance: média de AMT_BALANCE
   - cc_max_balance: máximo de AMT_BALANCE
   - cc_avg_utilization: média de utilização do limite
   - cc_max_utilization: máximo de utilização do limite
   - cc_avg_payment: média de AMT_PAYMENT_CURRENT
   - cc_avg_drawings: média de AMT_DRAWINGS_CURRENT
   - cc_avg_dpd: média de SK_DPD
   - cc_max_dpd: máximo de SK_DPD
   - cc_months_balance: total de meses de histórico
5. Salvar em data/interim/credit_card_features.parquet
6. Imprimir shape final e primeiras 3 linhas como confirmação

## Regras
- Nunca modificar data/raw/
- Usar pandas e numpy apenas
- Documentar cada etapa com comentários
- Salvar como .parquet
