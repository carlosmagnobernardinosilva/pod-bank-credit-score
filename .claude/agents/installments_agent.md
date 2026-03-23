---
name: installments_agent
description: Processa e agrega features de histórico de pagamento de parcelas (installments_payments.csv) por SK_ID_CURR
---

# Installments Agent

## Objetivo
Gerar data/interim/installments_features.parquet com features de comportamento de pagamento de parcelas por cliente.

## Inputs
- data/raw/installments_payments.csv

## Output
- data/interim/installments_features.parquet

## Passos obrigatórios
1. Carregar installments_payments.csv
2. Calcular features derivadas:
   - days_past_due: DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
     (positivo = atraso, negativo = adiantado)
   - payment_ratio: AMT_PAYMENT / AMT_INSTALMENT
     (substituir inf e NaN por 0)
   - underpayment: AMT_INSTALMENT - AMT_PAYMENT
     (se positivo, pagou menos que o devido)
3. Agregar por SK_ID_CURR:
   - inst_count: total de parcelas pagas
   - inst_avg_dpd: média de days_past_due
   - inst_max_dpd: máximo de days_past_due
   - inst_late_count: parcelas pagas com atraso (days_past_due > 0)
   - inst_late_rate: taxa de parcelas pagas com atraso
   - inst_avg_payment_ratio: média de payment_ratio
   - inst_min_payment_ratio: mínimo de payment_ratio
   - inst_avg_underpayment: média de underpayment
   - inst_num_instalments: número de contratos distintos (nunique de SK_ID_PREV)
4. Salvar em data/interim/installments_features.parquet
5. Imprimir shape final e primeiras 3 linhas como confirmação

## Regras
- Nunca modificar data/raw/
- Usar pandas e numpy apenas
- Documentar cada etapa com comentários
- Salvar como .parquet
