# Dicionário de Dados — PoD Bank Credit Score

> Fonte primária: `data/raw/HomeCredit_columns_description.csv`
> Complementado com achados do EDA em `notebooks/02_data_understanding/01_eda_exploratory.ipynb`

---

## 1. Visão Geral das Tabelas

| Tabela | Linhas | Colunas | Chave primária | Chave para application | Granularidade |
|---|---|---|---|---|---|
| `application_train.csv` | 307,511 | 122 | `SK_ID_CURR` | — | 1 linha por aplicação de crédito (com TARGET) |
| `application_test.csv` | 48,744 | 121 | `SK_ID_CURR` | — | 1 linha por aplicação de crédito (sem TARGET) |
| `bureau.csv` | ~1,716,428 | 17 | `SK_ID_BUREAU` | `SK_ID_CURR` | 1 linha por crédito externo registrado no bureau |
| `bureau_balance.csv` | ~27,299,925 | 3 | — | `SK_ID_BUREAU` → bureau | 1 linha por mês por crédito bureau |
| `previous_application.csv` | ~1,670,214 | 37 | `SK_ID_PREV` | `SK_ID_CURR` | 1 linha por aplicação anterior na PoD Bank |
| `POS_CASH_balance.csv` | ~10,001,358 | 8 | — | `SK_ID_CURR` | 1 linha por mês por contrato POS/cash anterior |
| `credit_card_balance.csv` | ~3,840,312 | 23 | — | `SK_ID_CURR` | 1 linha por mês por cartão de crédito anterior |
| `installments_payments.csv` | ~13,605,401 | 8 | — | `SK_ID_CURR` | 1 linha por parcela paga ou devida de créditos anteriores |

**Diagrama de relacionamento:**
```
application_train / application_test  (SK_ID_CURR)
        │
        ├──► bureau.csv  (SK_ID_CURR → SK_ID_BUREAU)
        │         └──► bureau_balance.csv  (SK_ID_BUREAU)
        │
        ├──► previous_application.csv  (SK_ID_CURR → SK_ID_PREV)
        │
        ├──► POS_CASH_balance.csv  (SK_ID_CURR, SK_ID_PREV)
        │
        ├──► credit_card_balance.csv  (SK_ID_CURR, SK_ID_PREV)
        │
        └──► installments_payments.csv  (SK_ID_CURR, SK_ID_PREV)
```

---

## 2. application_train / application_test

Tabela central do projeto. Cada linha representa **uma solicitação de crédito** feita à PoD Bank.
`application_train` contém a variável alvo `TARGET`. `application_test` não possui `TARGET` (usado para scoring).

### Variável TARGET

| Valor | Significado |
|---|---|
| `0` | Adimplente — cliente não apresentou dificuldades de pagamento |
| `1` | Inadimplente — cliente teve atraso superior a X dias em pelo menos uma das Y primeiras parcelas |

**Distribuição:** 91.93% classe 0 / 8.07% classe 1 (desbalanceamento ~11.4:1).

### Identificação

| Coluna | Tipo | Descrição |
|---|---|---|
| `SK_ID_CURR` | int | Identificador único da aplicação de crédito (chave primária, hash) |
| `TARGET` | int (0/1) | Variável alvo — somente em application_train |

### Dados Pessoais e Demográficos

| Coluna | Tipo | Descrição |
|---|---|---|
| `CODE_GENDER` | cat | Sexo do cliente |
| `DAYS_BIRTH` | int | Idade em dias (negativo) na data da aplicação — dividir por -365 para obter anos |
| `CNT_CHILDREN` | int | Número de filhos |
| `CNT_FAM_MEMBERS` | float | Número de membros da família |
| `NAME_FAMILY_STATUS` | cat | Estado civil (Married, Single/not married, Separated, Widow, Unknown) |
| `NAME_EDUCATION_TYPE` | cat | Nível de escolaridade |
| `OCCUPATION_TYPE` | cat | Tipo de ocupação profissional (~31% nulos) |
| `ORGANIZATION_TYPE` | cat | Tipo de organização onde o cliente trabalha |

### Dados do Contrato Solicitado

| Coluna | Tipo | Descrição |
|---|---|---|
| `NAME_CONTRACT_TYPE` | cat | Tipo do contrato: `Cash loans` ou `Revolving loans` |
| `AMT_CREDIT` | float | Valor do crédito aprovado |
| `AMT_ANNUITY` | float | Valor da anuidade/parcela do empréstimo |
| `AMT_GOODS_PRICE` | float | Preço do bem financiado (para crédito consignado) |
| `AMT_INCOME_TOTAL` | float | Renda total declarada pelo cliente |
| `NAME_INCOME_TYPE` | cat | Tipo de renda (Working, Commercial associate, Pensioner, State servant...) |

### Dados de Emprego e Residência

| Coluna | Tipo | Descrição |
|---|---|---|
| `DAYS_EMPLOYED` | int | Dias de emprego atual (negativo = empregado; `365243` = aposentado/desempregado — tratar como `NaN`) |
| `DAYS_REGISTRATION` | float | Dias desde a última alteração de registro do cliente |
| `DAYS_ID_PUBLISH` | float | Dias desde a última atualização do documento de identidade |
| `DAYS_LAST_PHONE_CHANGE` | float | Dias desde a última troca de telefone |
| `NAME_HOUSING_TYPE` | cat | Tipo de moradia (House/apartment, With parents, Municipal apartment...) |
| `OWN_CAR_AGE` | float | Idade do carro próprio em anos (~65% nulos — ausência indica sem carro) |
| `FLAG_OWN_CAR` | cat | Flag: possui carro (Y/N) |
| `FLAG_OWN_REALTY` | cat | Flag: possui imóvel (Y/N) |

### Localização e Endereços

| Coluna | Tipo | Descrição |
|---|---|---|
| `REGION_POPULATION_RELATIVE` | float | População normalizada da região onde o cliente mora |
| `REGION_RATING_CLIENT` | int | Avaliação interna da região (1, 2, 3 — quanto maior, pior) |
| `REGION_RATING_CLIENT_W_CITY` | int | Avaliação da região com cidade considerada (1, 2, 3) |
| `REG_REGION_NOT_LIVE_REGION` | int (0/1) | Endereço permanente ≠ endereço de contato (nível região) |
| `REG_REGION_NOT_WORK_REGION` | int (0/1) | Endereço permanente ≠ endereço de trabalho (nível região) |
| `REG_CITY_NOT_LIVE_CITY` | int (0/1) | Endereço permanente ≠ endereço de contato (nível cidade) |
| `REG_CITY_NOT_WORK_CITY` | int (0/1) | Endereço permanente ≠ endereço de trabalho (nível cidade) |
| `LIVE_CITY_NOT_WORK_CITY` | int (0/1) | Endereço de contato ≠ endereço de trabalho (nível cidade) |

### Scores Externos (Alta Importância Preditiva)

| Coluna | Tipo | % Nulos | Descrição |
|---|---|---|---|
| `EXT_SOURCE_1` | float | ~56.4% | Score normalizado de fonte externa de bureau (origem não divulgada) |
| `EXT_SOURCE_2` | float | ~0.3% | Score normalizado de fonte externa de bureau |
| `EXT_SOURCE_3` | float | ~19.8% | Score normalizado de fonte externa de bureau |

> Ver seção 10 para interpretação completa.

### Características do Imóvel (Bloco com Alto % de Nulos)

Conjunto de 45 colunas com sufixos `_AVG`, `_MODE`, `_MEDI` descrevendo características normalizadas do imóvel onde o cliente reside. Inclui: tamanho do apartamento, área comum, área de subsolo, número de andares, elevadores, entradas, ano de construção.

| Grupo | % Nulos Típico |
|---|---|
| `COMMONAREA_*` | ~69.9% |
| `NONLIVINGAPARTMENTS_*` | ~69.4% |
| `FONDKAPREMONT_MODE` | ~68.4% |
| `LIVINGAPARTMENTS_*` | ~68.4% |
| `YEARS_BUILD_*` | ~66.5% |
| `OWN_CAR_AGE` | ~65.2% |
| `LANDAREA_*` | ~65.0% |
| `BASEMENTAREA_*` | ~64.4% |
| `FLOORSMIN_*` | ~67.8% |
| `ELEVATORS_*` | ~52.3% |
| `WALLSMATERIAL_MODE` | ~47.7% |
| `EMERGENCYSTATE_MODE` | ~48.8% |

Colunas adicionais: `HOUSETYPE_MODE`, `TOTALAREA_MODE`, `YEARS_BEGINEXPLUATATION_*`.

### Contato e Flags de Comunicação

| Coluna | Tipo | Descrição |
|---|---|---|
| `FLAG_MOBIL` | int (0/1) | Forneceu celular |
| `FLAG_EMP_PHONE` | int (0/1) | Forneceu telefone comercial |
| `FLAG_WORK_PHONE` | int (0/1) | Forneceu telefone fixo no trabalho |
| `FLAG_CONT_MOBILE` | int (0/1) | Celular era acessível |
| `FLAG_PHONE` | int (0/1) | Forneceu telefone residencial |
| `FLAG_EMAIL` | int (0/1) | Forneceu e-mail |

### Documentos Apresentados

| Coluna | Tipo | Descrição |
|---|---|---|
| `FLAG_DOCUMENT_2` a `FLAG_DOCUMENT_21` | int (0/1) | Indicadores de 20 tipos diferentes de documentos fornecidos pelo cliente |

### Consultas ao Bureau de Crédito

| Coluna | Tipo | Descrição |
|---|---|---|
| `AMT_REQ_CREDIT_BUREAU_HOUR` | float | Consultas ao bureau na última hora |
| `AMT_REQ_CREDIT_BUREAU_DAY` | float | Consultas ao bureau no último dia (excluindo 1h) |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | float | Consultas ao bureau na última semana (excluindo 1d) |
| `AMT_REQ_CREDIT_BUREAU_MON` | float | Consultas ao bureau no último mês (excluindo 1 semana) |
| `AMT_REQ_CREDIT_BUREAU_QRT` | float | Consultas ao bureau nos últimos 3 meses (excluindo 1 mês) |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | float | Consultas ao bureau no último ano (excluindo últimos 3 meses) |

### Círculo Social (Inadimplência de Terceiros)

| Coluna | Tipo | Descrição |
|---|---|---|
| `OBS_30_CNT_SOCIAL_CIRCLE` | float | Qtd. observações do círculo social com 30 DPD observável |
| `DEF_30_CNT_SOCIAL_CIRCLE` | float | Qtd. de pessoas do círculo social inadimplentes a 30 DPD |
| `OBS_60_CNT_SOCIAL_CIRCLE` | float | Qtd. observações do círculo social com 60 DPD observável |
| `DEF_60_CNT_SOCIAL_CIRCLE` | float | Qtd. de pessoas do círculo social inadimplentes a 60 DPD |

### Processo de Aplicação

| Coluna | Tipo | Descrição |
|---|---|---|
| `WEEKDAY_APPR_PROCESS_START` | cat | Dia da semana em que a aplicação foi iniciada |
| `HOUR_APPR_PROCESS_START` | int | Hora aproximada de início da aplicação (arredondada) |
| `NAME_TYPE_SUITE` | cat | Quem acompanhou o cliente na aplicação |

---

## 3. bureau.csv

Histórico de créditos que o cliente possui em **outras instituições financeiras**, fornecidos pelo bureau de crédito externo. Um cliente pode ter 0, 1 ou vários registros nesta tabela.

**Chave de relacionamento:** `SK_ID_CURR` (join com application_train) | `SK_ID_BUREAU` (chave primária desta tabela, usada no join com bureau_balance)

| Coluna | Tipo | Descrição |
|---|---|---|
| `SK_ID_CURR` | int | FK para application_train/test |
| `SK_ID_BUREAU` | int | Identificador único do crédito no bureau (hash) — PK desta tabela |
| `CREDIT_ACTIVE` | cat | Status atual do crédito: `Active`, `Closed`, `Sold`, `Bad debt` |
| `CREDIT_CURRENCY` | cat | Moeda do crédito (recodificada como currency1–4) |
| `DAYS_CREDIT` | int | Dias antes da aplicação atual em que o cliente solicitou este crédito (negativo) |
| `CREDIT_DAY_OVERDUE` | int | Dias em atraso no momento da aplicação atual |
| `DAYS_CREDIT_ENDDATE` | float | Duração restante do crédito em dias na data de aplicação |
| `DAYS_ENDDATE_FACT` | float | Dias desde o encerramento do crédito (somente para créditos fechados) |
| `AMT_CREDIT_MAX_OVERDUE` | float | Valor máximo em atraso registrado no crédito |
| `CNT_CREDIT_PROLONG` | float | Quantas vezes o crédito foi prorrogado |
| `AMT_CREDIT_SUM` | float | Valor total atual do crédito bureau |
| `AMT_CREDIT_SUM_DEBT` | float | Saldo devedor atual do crédito bureau |
| `AMT_CREDIT_SUM_LIMIT` | float | Limite atual do cartão de crédito (se aplicável) |
| `AMT_CREDIT_SUM_OVERDUE` | float | Valor atualmente em atraso no crédito bureau |
| `CREDIT_TYPE` | cat | Tipo de crédito (Car loan, Consumer credit, Credit card, Mortgage, Microloan...) |
| `DAYS_CREDIT_UPDATE` | int | Dias antes da aplicação em que a última informação do bureau foi atualizada |
| `AMT_ANNUITY` | float | Anuidade do crédito bureau |

---

## 4. bureau_balance.csv

Saldo **mensal** de cada crédito registrado no bureau. Tabela com a maior granularidade do dataset (~27M linhas). Conecta-se ao restante do projeto via `bureau.csv`.

**Chave de relacionamento:** `SK_ID_BUREAU` → join com `bureau.csv`, que por sua vez conecta a `SK_ID_CURR` em application.

| Coluna | Tipo | Descrição |
|---|---|---|
| `SK_ID_BUREAU` | int | FK para bureau.csv |
| `MONTHS_BALANCE` | int | Mês relativo à data de aplicação (−1 = mais recente; valores negativos crescentes = mais antigo) |
| `STATUS` | cat | Status do crédito bureau durante aquele mês (ver tabela abaixo) |

### Categorias da Coluna STATUS

| Valor | Significado |
|---|---|
| `C` | Crédito encerrado (*Closed*) naquele mês |
| `X` | Status desconhecido (*Unknown*) |
| `0` | Sem atraso (0 DPD — *Days Past Due*) |
| `1` | Atraso de 1 a 30 dias (DPD 1–30) |
| `2` | Atraso de 31 a 60 dias (DPD 31–60) |
| `3` | Atraso de 61 a 90 dias (DPD 61–90) |
| `4` | Atraso de 91 a 120 dias (DPD 91–120) |
| `5` | Atraso superior a 120 dias, ou crédito vendido/baixado (*written off*) |

> Para feature engineering: contar meses com STATUS >= '2' como indicador de histórico de inadimplência grave.

---

## 5. previous_application.csv

Aplicações de crédito **anteriores** feitas pelo mesmo cliente na PoD Bank. Uma aplicação anterior pode ou não ter resultado em crédito concedido.

**Chave de relacionamento:** `SK_ID_CURR` (join com application) | `SK_ID_PREV` (chave desta tabela, usada em POS_CASH_balance, credit_card_balance e installments_payments)

| Coluna | Tipo | Descrição |
|---|---|---|
| `SK_ID_PREV` | int | Identificador único da aplicação anterior (hash) |
| `SK_ID_CURR` | int | FK para application_train/test |
| `NAME_CONTRACT_TYPE` | cat | Tipo do produto (Cash loan, Consumer loan, Revolving loan, XNA) |
| `AMT_ANNUITY` | float | Anuidade da aplicação anterior |
| `AMT_APPLICATION` | float | Valor que o cliente solicitou na aplicação anterior |
| `AMT_CREDIT` | float | Valor final aprovado (pode diferir de AMT_APPLICATION) |
| `AMT_DOWN_PAYMENT` | float | Valor de entrada pago na aplicação anterior |
| `AMT_GOODS_PRICE` | float | Preço do bem solicitado (se aplicável) |
| `NAME_CONTRACT_STATUS` | cat | Status final da aplicação: `Approved`, `Cancelled`, `Refused`, `Unused offer` |
| `DAYS_DECISION` | int | Dias antes da aplicação atual em que a decisão foi tomada |
| `CODE_REJECT_REASON` | cat | Motivo de recusa, se aplicável (XAP = Approved — ver seção 9) |
| `NAME_PAYMENT_TYPE` | cat | Forma de pagamento escolhida |
| `NAME_CLIENT_TYPE` | cat | Novo ou antigo cliente na época da aplicação anterior |
| `NAME_CASH_LOAN_PURPOSE` | cat | Finalidade declarada do empréstimo (XNA/XAP para não informado) |
| `NAME_GOODS_CATEGORY` | cat | Categoria do bem financiado |
| `NAME_PORTFOLIO` | cat | Portfólio: Cash, POS, Cards, Car |
| `CHANNEL_TYPE` | cat | Canal de aquisição do cliente |
| `CNT_PAYMENT` | float | Número de parcelas do crédito anterior |
| `NAME_YIELD_GROUP` | cat | Faixa da taxa de juros (low, middle, high, XNA) |
| `RATE_DOWN_PAYMENT` | float | Taxa de entrada normalizada |
| `DAYS_FIRST_DRAWING` | float | Dias até o primeiro desembolso (ver nota sobre 365243) |
| `DAYS_FIRST_DUE` | float | Dias até o primeiro vencimento |
| `DAYS_LAST_DUE_1ST_VERSION` | float | Dias até o primeiro vencimento (versão original) |
| `DAYS_LAST_DUE` | float | Dias até o último vencimento |
| `DAYS_TERMINATION` | float | Dias até o encerramento esperado do contrato |
| `NFLAG_INSURED_ON_APPROVAL` | float | Cliente solicitou seguro na aprovação (1=Sim, 0=Não) |

> **Valor especial 365243:** Presente nas colunas `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`, `DAYS_LAST_DUE` e `DAYS_TERMINATION`. Representa dado ausente ou não aplicável — deve ser substituído por `NaN` antes de qualquer operação com datas.

---

## 6. POS_CASH_balance.csv

Saldo mensal de **empréstimos POS (ponto de venda) e empréstimos em dinheiro anteriores** concedidos pela PoD Bank. Registra a evolução mensal de cada contrato.

**Chave de relacionamento:** `SK_ID_CURR` → application | `SK_ID_PREV` → previous_application

| Coluna | Tipo | Descrição |
|---|---|---|
| `SK_ID_PREV` | int | FK para previous_application |
| `SK_ID_CURR` | int | FK para application_train/test |
| `MONTHS_BALANCE` | int | Mês relativo à data de aplicação (−1 = mais recente) |
| `CNT_INSTALMENT` | float | Número total de parcelas do contrato (pode mudar ao longo do tempo) |
| `CNT_INSTALMENT_FUTURE` | float | Parcelas restantes a pagar no contrato |
| `NAME_CONTRACT_STATUS` | cat | Status do contrato naquele mês (Active, Completed, Signed, Demand, XNA...) |
| `SK_DPD` | int | Dias de atraso (DPD) naquele mês |
| `SK_DPD_DEF` | int | DPD com tolerância (dívidas de baixo valor são ignoradas) |

---

## 7. credit_card_balance.csv

Extratos mensais de **cartões de crédito anteriores** emitidos pela PoD Bank. Contém dados de utilização, pagamento e limite de cada cartão.

**Chave de relacionamento:** `SK_ID_CURR` → application | `SK_ID_PREV` → previous_application

| Coluna | Tipo | Descrição |
|---|---|---|
| `SK_ID_PREV` | int | FK para previous_application |
| `SK_ID_CURR` | int | FK para application_train/test |
| `MONTHS_BALANCE` | int | Mês relativo à data de aplicação (−1 = mais recente) |
| `AMT_BALANCE` | float | Saldo devedor no mês |
| `AMT_CREDIT_LIMIT_ACTUAL` | float | Limite do cartão naquele mês |
| `AMT_DRAWINGS_ATM_CURRENT` | float | Saques em ATM no mês |
| `AMT_DRAWINGS_CURRENT` | float | Total de gastos no mês |
| `AMT_DRAWINGS_OTHER_CURRENT` | float | Outros tipos de saque/uso no mês |
| `AMT_DRAWINGS_POS_CURRENT` | float | Gastos em compras no mês |
| `AMT_INST_MIN_REGULARITY` | float | Parcela mínima exigida no mês |
| `AMT_PAYMENT_CURRENT` | float | Valor pago no mês |
| `AMT_PAYMENT_TOTAL_CURRENT` | float | Total pago no mês (incluindo encargos) |
| `AMT_RECEIVABLE_PRINCIPAL` | float | Principal a receber |
| `AMT_RECIVABLE` | float | Total a receber |
| `AMT_TOTAL_RECEIVABLE` | float | Total a receber (incluindo juros) |
| `CNT_DRAWINGS_ATM_CURRENT` | float | Número de saques ATM no mês |
| `CNT_DRAWINGS_CURRENT` | float | Número de transações no mês |
| `CNT_DRAWINGS_OTHER_CURRENT` | float | Número de outros saques no mês |
| `CNT_DRAWINGS_POS_CURRENT` | float | Número de compras no mês |
| `CNT_INSTALMENT_MATURE_CUM` | float | Total de parcelas pagas acumuladas |
| `NAME_CONTRACT_STATUS` | cat | Status do contrato (Active, Completed, Demand, Signed...) |
| `SK_DPD` | int | Dias de atraso (DPD) no mês |
| `SK_DPD_DEF` | int | DPD com tolerância |

> Feature sugerida: taxa de utilização do limite (`AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`) — proxy de estresse financeiro.

---

## 8. installments_payments.csv

Histórico detalhado de **pagamento de parcelas** de todos os créditos anteriores do cliente na PoD Bank. Cada linha representa uma parcela — com valor devido, valor pago e datas prevista e efetiva.

**Chave de relacionamento:** `SK_ID_CURR` → application | `SK_ID_PREV` → previous_application

| Coluna | Tipo | Descrição |
|---|---|---|
| `SK_ID_PREV` | int | FK para previous_application |
| `SK_ID_CURR` | int | FK para application_train/test |
| `NUM_INSTALMENT_VERSION` | float | Versão do calendário de parcelas (0 = cartão de crédito; mudança de versão indica renegociação) |
| `NUM_INSTALMENT_NUMBER` | float | Número sequencial da parcela observada |
| `DAYS_INSTALMENT` | float | Quando a parcela deveria ser paga (relativo à aplicação atual) |
| `DAYS_ENTRY_PAYMENT` | float | Quando a parcela foi efetivamente paga |
| `AMT_INSTALMENT` | float | Valor da parcela prescrito |
| `AMT_PAYMENT` | float | Valor efetivamente pago pelo cliente |

> Features derivadas de alta importância:
> - `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT` → atraso em dias por parcela (positivo = pagou com atraso)
> - `AMT_PAYMENT / AMT_INSTALMENT` → razão de pagamento (< 1 = pagou menos do que devia)

---

## 9. Valores Especiais e Anomalias

| Valor | Coluna(s) | Tabela | Significado Real | Tratamento Recomendado |
|---|---|---|---|---|
| `365243` | `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`, `DAYS_LAST_DUE`, `DAYS_TERMINATION` | `previous_application` | Dado ausente ou não aplicável (ex.: contrato cancelado antes de desembolso) | Substituir por `NaN` antes de qualquer cálculo |
| `365243` | `DAYS_EMPLOYED` | `application_train/test` | Cliente não possui emprego formal (aposentado, desempregado, autônomo sem vínculo) | Substituir por `NaN`; criar flag binária `IS_NOT_EMPLOYED` |
| `XNA` | `NAME_CASH_LOAN_PURPOSE`, `CODE_GENDER`, `NAME_CONTRACT_TYPE`, `NAME_YIELD_GROUP` | `previous_application`, `application` | Não disponível (*Not Available*) | Tratar como categoria explícita `'Unknown'` ou `NaN` dependendo da frequência |
| `XAP` | `CODE_REJECT_REASON`, `NAME_CASH_LOAN_PURPOSE` | `previous_application` | Não aplicável (*Not Applicable*) — geralmente indica que o campo não era pertinente (ex.: motivo de recusa quando o crédito foi aprovado) | Tratar como categoria explícita `'Approved'` para `CODE_REJECT_REASON`; `NaN` para os demais |
| `currency1–4` | `CREDIT_CURRENCY` | `bureau` | Moeda recodificada para anonimização | Manter como feature categórica; `currency1` é dominante |

---

## 10. Features Prioritárias para Modelagem

### EXT_SOURCE — Scores Externos de Bureau

Os três scores externos são as features com maior poder preditivo individual neste dataset.

| Feature | Correlação com TARGET | % Nulos | Interpretação |
|---|---|---|---|
| `EXT_SOURCE_3` | **−0.178** | ~19.8% | Score normalizado de fonte externa não identificada. Quanto mais alto, menor o risco de inadimplência. |
| `EXT_SOURCE_2` | **−0.160** | ~0.3% | Idem. Disponível para quase todos os clientes — mais confiável para usar diretamente. |
| `EXT_SOURCE_1` | **−0.155** | ~56.4% | Idem. Alta taxa de ausência — provavelmente um score de bureau que não cobre toda a base. |

**Estratégia recomendada:**
- Criar feature combinada: `EXT_SOURCE_MEAN = mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)`
- Criar feature combinada: `EXT_SOURCE_PROD = EXT_SOURCE_2 * EXT_SOURCE_3` (contorna os nulos de EXT_SOURCE_1)
- Explorar interações entre os três scores

### Top 10 Correlações com TARGET (application_train)

| Rank | Feature | Correlação | Direção | Interpretação |
|---|---|---|---|---|
| 1 | `EXT_SOURCE_3` | −0.178 | Negativa | Score externo alto = menor risco |
| 2 | `EXT_SOURCE_2` | −0.160 | Negativa | Score externo alto = menor risco |
| 3 | `EXT_SOURCE_1` | −0.155 | Negativa | Score externo alto = menor risco |
| 4 | `DAYS_BIRTH` | −0.078 | Negativa | Clientes mais velhos tendem a ser mais adimplentes |
| 5 | `REGION_RATING_CLIENT_W_CITY` | +0.061 | Positiva | Regiões com pior avaliação têm mais inadimplência |
| 6 | `REGION_RATING_CLIENT` | +0.059 | Positiva | Idem (sem considerar cidade) |
| 7 | `DAYS_LAST_PHONE_CHANGE` | +0.055 | Positiva | Troca recente de telefone associada a mais risco |
| 8 | `DAYS_ID_PUBLISH` | +0.051 | Positiva | Documento de identidade mais antigo (menor valor absoluto) = menos risco |
| 9 | `REG_CITY_NOT_WORK_CITY` | +0.045 | Positiva | Morar em cidade diferente do trabalho = mais risco |
| 10 | `FLAG_EMP_PHONE` | +0.045 | Positiva | Não fornecer telefone comercial = mais risco |

> **Nota:** Colunas `DAYS_*` têm valores negativos no dataset (dias antes da aplicação). Correlações positivas com TARGET para essas colunas indicam que valores menos negativos (evento mais recente ou menor histórico) estão associados a maior risco.
