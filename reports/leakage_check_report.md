# Relatório de Verificação de Data Leakage

**Convenção:** colunas `DAYS_*` e `MONTHS_BALANCE` devem ser ≤ 0 (passado).
Valores > 0 indicam eventos futuros à data de aplicação — potencial leakage.

---

## bureau.csv

| Coluna | Total Linhas | Registros Suspeitos | % Afetado | Amostra de Valores |
|--------|-------------|---------------------|-----------|-------------------|
| `DAYS_CREDIT` | 1,716,428 | 0 | 0.00% | — |
| `DAYS_CREDIT_ENDDATE` | 1,716,428 | 602,603 | 35.11% | 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 |
| `DAYS_ENDDATE_FACT` | 1,716,428 | 0 | 0.00% | — |
| `DAYS_CREDIT_UPDATE` | 1,716,428 | 17 | 0.00% | 10, 11, 12, 13, 14, 15, 16, 19, 20, 22 |

**Subtotal suspeitos:** 602,620 registros

**Recomendação:** Registros suspeitos em até 35.1% dos dados. Investigar se representam datas futuras reais ou erros de codificação. Recomendar remoção dos registros com valores > 0 antes do treino.

---

## installments_payments.csv

| Coluna | Total Linhas | Registros Suspeitos | % Afetado | Amostra de Valores |
|--------|-------------|---------------------|-----------|-------------------|
| `DAYS_INSTALMENT` | 13,605,401 | 0 | 0.00% | — |
| `DAYS_ENTRY_PAYMENT` | 13,605,401 | 0 | 0.00% | — |

**Subtotal suspeitos:** 0 registros

**Recomendação:** Nenhum leakage detectado. Sem ação necessária.

---

## previous_application.csv

| Coluna | Total Linhas | Registros Suspeitos | % Afetado | Amostra de Valores |
|--------|-------------|---------------------|-----------|-------------------|
| `DAYS_DECISION` | 1,670,214 | 0 | 0.00% | — |
| `DAYS_FIRST_DRAWING` | 1,670,214 | 0 | 0.00% | — |
| `DAYS_FIRST_DUE` | 1,670,214 | 0 | 0.00% | — |
| `DAYS_LAST_DUE_1ST_VERSION` | 1,670,214 | 224,392 | 13.43% | 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 |
| `DAYS_LAST_DUE` | 1,670,214 | 0 | 0.00% | — |
| `DAYS_TERMINATION` | 1,670,214 | 0 | 0.00% | — |

**Subtotal suspeitos:** 224,392 registros

**Recomendação:** Registros suspeitos em até 13.4% dos dados. Investigar se representam datas futuras reais ou erros de codificação. Recomendar remoção dos registros com valores > 0 antes do treino.

---

## POS_CASH_balance.csv

| Coluna | Total Linhas | Registros Suspeitos | % Afetado | Amostra de Valores |
|--------|-------------|---------------------|-----------|-------------------|
| `MONTHS_BALANCE` | 10,001,358 | 0 | 0.00% | — |

**Subtotal suspeitos:** 0 registros

**Recomendação:** Nenhum leakage detectado. Sem ação necessária.

---

## credit_card_balance.csv

| Coluna | Total Linhas | Registros Suspeitos | % Afetado | Amostra de Valores |
|--------|-------------|---------------------|-----------|-------------------|
| `MONTHS_BALANCE` | 3,840,312 | 0 | 0.00% | — |

**Subtotal suspeitos:** 0 registros

**Recomendação:** Nenhum leakage detectado. Sem ação necessária.

---

## Conclusão Geral

Total de **827,012 registros suspeitos** encontrados. Aplicar filtragem antes de re-treinar os modelos.

---
*Gerado por `src/features/leakage_check.py`*