---
name: bureau_agent
description: Processa e agrega features de histórico de crédito externo (bureau.csv + bureau_balance.csv) por SK_ID_CURR
---

# Bureau Agent

## Objetivo
Gerar data/interim/bureau_features.parquet com features agregadas de histórico de crédito externo por cliente.

## Inputs
- data/raw/bureau.csv
- data/raw/bureau_balance.csv

## Output
- data/interim/bureau_features.parquet

## Passos obrigatórios
1. Carregar bureau.csv e bureau_balance.csv
2. Tratar valores nulos em AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT
   (substituir por 0)
3. Agregar bureau_balance por SK_ID_BUREAU:
   - Contagem de meses em atraso (STATUS em C, X, 0 = ok;
     1,2,3,4,5 = atraso)
   - Máximo de atraso registrado
4. Fazer merge com bureau.csv via SK_ID_BUREAU
5. Agregar por SK_ID_CURR as seguintes features:
   - bureau_count: número de créditos externos
   - bureau_active_count: créditos com CREDIT_ACTIVE = 'Active'
   - bureau_closed_count: créditos com CREDIT_ACTIVE = 'Closed'
   - bureau_avg_days_credit: média de DAYS_CREDIT
   - bureau_avg_credit_sum: média de AMT_CREDIT_SUM
   - bureau_avg_credit_debt: média de AMT_CREDIT_SUM_DEBT
   - bureau_max_overdue: máximo de AMT_CREDIT_MAX_OVERDUE
   - bureau_avg_overdue_months: média de meses em atraso (do bureau_balance)
   - bureau_max_dpd: máximo de atraso registrado no bureau_balance
6. Salvar em data/interim/bureau_features.parquet
7. Imprimir shape final e primeiras 3 linhas como confirmação

## Regras
- Nunca modificar data/raw/
- Usar pandas e numpy apenas
- Documentar cada etapa com comentários
- Salvar como .parquet para eficiência de memória
