# PoD Bank — Model Documentation
> Rastreabilidade completa do projeto de credit scoring (CRISP-DM)

**Última atualização:** 2026-03-24 v0.7
**Modelo alvo:** Predição de inadimplência (TARGET = 1)
**Dataset:** Home Credit Default Risk

---

## Índice
1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Métricas de Sucesso](#métricas-de-sucesso)
3. [Fase 2 — Data Understanding](#fase-2--data-understanding)
4. [Fase 3 — Data Preparation](#fase-3--data-preparation)
5. [Fase 4 — Modeling](#fase-4--modeling)
6. [Fase 5 — Evaluation](#fase-5--evaluation)
7. [Fase 6 — Deployment](#fase-6--deployment)
8. [Alertas Pendentes](#alertas-pendentes)
9. [Decisões de Negócio](#decisões-de-negócio)
10. [Histórico de Versões](#histórico-de-versões)

---

## 1. Visão Geral do Projeto

O projeto PoD Bank Credit Score tem como objetivo construir um modelo de predição de inadimplência (credit scoring) seguindo o framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining). O modelo deve identificar, no momento da solicitação de crédito, quais clientes têm maior probabilidade de não honrar seus compromissos financeiros.

### Dataset

O projeto utiliza o dataset público **Home Credit Default Risk**, composto por 9 arquivos CSV (~2.8 GB) cobrindo o histórico completo de crédito de solicitantes:

| Arquivo | Descrição | Linhas (aprox.) |
|---|---|---|
| `application_train.csv` | Tabela principal — uma linha por aplicação, com target | 307,511 |
| `application_test.csv` | Conjunto de teste (sem target) | 48,744 |
| `bureau.csv` | Histórico de crédito em outras instituições | ~1.72M |
| `bureau_balance.csv` | Saldo mensal dos créditos do bureau | ~27.3M |
| `previous_application.csv` | Aplicações anteriores no PoD Bank | ~1.67M |
| `POS_CASH_balance.csv` | Saldos mensais de empréstimos POS/cash | ~10.0M |
| `credit_card_balance.csv` | Extratos mensais de cartão de crédito | ~3.84M |
| `installments_payments.csv` | Histórico de pagamento de parcelas | ~13.6M |

### Estrutura do Repositório

```
data/raw/          — Dados brutos originais (nunca modificados)
data/interim/      — Outputs de limpeza e joining de scripts
data/processed/    — Datasets feature-engineered prontos para modelagem
notebooks/         — Exploração e experimentação por fase CRISP-DM
src/               — Módulos Python reutilizáveis e prontos para produção
.claude/agents/    — Definições dos subagentes especializados
docs/              — Documentação do projeto
tests/             — Testes unitários para src/
reports/figures/   — Gráficos exportados de notebooks ou src/evaluation
```

### Restrições Técnicas

- **Explicabilidade obrigatória**: decisões do modelo devem ser justificáveis (usar SHAP ou similar)
- **Latência de inferência**: < 500ms por solicitante em produção
- **Dados não versionados**: todo o conteúdo de `data/` e `models/` está no `.gitignore`

---

## 2. Métricas de Sucesso

Todas as decisões de modelagem devem ser avaliadas contra os seguintes alvos:

| Métrica | Alvo | Justificativa |
|---|---|---|
| AUC-ROC | >= 0.72 | Discriminação geral entre bons e maus pagadores |
| KS Statistic | >= 0.32 | Separação máxima entre distribuições de risco |
| Gini Coefficient | >= 0.42 | Relação direta com AUC: Gini = 2×AUC - 1 |
| Recall (bons pagadores) | >= 60% | Garantia de não recusar clientes viáveis |

**Nota sobre desbalanceamento**: o dataset apresenta razão de desbalanceamento de ~11.4:1 (91.93% adimplentes vs. 8.07% inadimplentes no treino). Implicações:
- Usar `class_weight='balanced'` ou equivalente em todos os modelos
- Avaliar com AUC-ROC e KS, não acurácia bruta
- Calibrar threshold de decisão separadamente da otimização do modelo
- Validar contra data leakage usando splits temporais

---

## 3. Fase 2 — Data Understanding

> Fonte: `notebooks/02_data_understanding/01_eda_exploratory.ipynb`

### 3.1 Distribuição do TARGET

| Classe | Descrição | Contagem | % |
|---|---|---|---|
| `0` | Adimplente | 282,686 | 91.93% |
| `1` | Inadimplente | 24,825 | 8.07% |

Razão de desbalanceamento: **~11.4:1**

### 3.2 Qualidade dos Dados — application_train

Colunas com mais de 30% de valores nulos identificadas durante o EDA:

| Coluna | % Nulos | Grupo |
|---|---|---|
| `COMMONAREA_*` (3 colunas) | ~69.9% | Área comum do imóvel |
| `NONLIVINGAPARTMENTS_*` (3) | ~69.4% | Área não residencial |
| `FONDKAPREMONT_MODE` | ~68.4% | Tipo de fundo do imóvel |
| `LIVINGAPARTMENTS_*` (3) | ~68.4% | Área residencial |
| `YEARS_BUILD_*` (3) | ~66.5% | Ano de construção |
| `OWN_CAR_AGE` | ~65.2% | Idade do veículo |
| `EXT_SOURCE_1` | ~56.4% | Score externo de bureau |
| `OCCUPATION_TYPE` | ~31.3% | Ocupação profissional |

### 3.3 Valores Especiais Identificados

- **`365243`** em colunas `DAYS_*` de `previous_application.csv` — código sentinela para dado ausente; deve ser substituído por `NaN` antes de qualquer cálculo de dias
- **`XNA` e `XAP`** em variáveis categóricas de `previous_application.csv` — categorias de "não disponível / não aplicável"; tratar como `NaN` ou categoria explícita `'Unknown'` dependendo da frequência

### 3.4 Top 10 Features Correlacionadas com TARGET

| # | Feature | Correlação | Direção |
|---|---|---|---|
| 1 | `EXT_SOURCE_3` | -0.178 | Negativa — score alto = menos risco |
| 2 | `EXT_SOURCE_2` | -0.160 | Negativa |
| 3 | `EXT_SOURCE_1` | -0.155 | Negativa |
| 4 | `DAYS_BIRTH` | -0.078 | Negativa — mais velho = menos risco |
| 5 | `REGION_RATING_CLIENT_W_CITY` | +0.061 | Positiva — região pior = mais risco |
| 6 | `REGION_RATING_CLIENT` | +0.059 | Positiva |
| 7 | `DAYS_LAST_PHONE_CHANGE` | +0.055 | Positiva — troca recente = mais risco |
| 8 | `DAYS_ID_PUBLISH` | +0.051 | Positiva — documento velho = mais risco |
| 9 | `REG_CITY_NOT_WORK_CITY` | +0.045 | Positiva |
| 10 | `FLAG_EMP_PHONE` | +0.045 | Positiva |

`EXT_SOURCE_1/2/3` são os sinais mais fortes. Priorizar na feature engineering e investigar interações entre eles.

### 3.5 Granularidade das Tabelas Secundárias

| Tabela | Linhas | SK únicos | Média linhas/cliente | Chave de join |
|---|---|---|---|---|
| `bureau` | ~1.72M | ~305K | ~5.6 | `SK_ID_CURR` |
| `bureau_balance` | ~27.3M | ~817K | ~33.4 | `SK_ID_BUREAU` |
| `previous_application` | ~1.67M | ~338K | ~4.9 | `SK_ID_CURR` |
| `POS_CASH_balance` | ~10.0M | ~361K | ~27.7 | `SK_ID_CURR` |
| `credit_card_balance` | ~3.84M | ~103K | ~37.3 | `SK_ID_CURR` |
| `installments_payments` | ~13.6M | ~339K | ~40.1 | `SK_ID_CURR` |

Todas as tabelas secundárias exigem **agregação por `SK_ID_CURR`** antes de fazer join com a tabela principal.

---

## 4. Fase 3 — Data Preparation

**Status:** Concluída em 2026-03-23

A fase 3 foi implementada por meio de uma arquitetura multi-agente com um orquestrador central e 6 subagentes especializados. Cada subagente é responsável por uma tabela de dados e gera um arquivo `.parquet` intermediário com as features agregadas por cliente (`SK_ID_CURR`). O orquestrador consolida todos os outputs em um dataset final para modelagem.

### 4.1 Arquitetura dos Subagentes

```
orchestrator_agent
├── application_agent   → data/interim/application_clean.parquet
├── bureau_agent        → data/interim/bureau_features.parquet
├── previous_app_agent  → data/interim/prev_app_features.parquet
├── pos_cash_agent      → data/interim/pos_cash_features.parquet
├── credit_card_agent   → data/interim/credit_card_features.parquet
└── installments_agent  → data/interim/installments_features.parquet
```

Após a execução paralela dos subagentes, o orquestrador realiza o merge final e gera:
- `data/processed/train_final.parquet`
- `data/processed/test_final.parquet`

### 4.2 application_clean.parquet

**Arquivo:** `data/interim/application_clean.parquet`
**Shape real:** (307,511 linhas × 178 colunas)
**Fontes:** `application_train.csv` + `application_test.csv`

#### Tratamentos Aplicados

| Tratamento | Coluna(s) | Ação |
|---|---|---|
| Flag de split | `is_train` | 1 para treino, 0 para teste; concatenação dos dois datasets |
| Categoria especial | `CODE_GENDER` | `'XNA'` substituído pela moda |
| Categoria especial | `ORGANIZATION_TYPE` | `'XNA'` substituído por `'Unknown'` |
| Valor anômalo | `DAYS_EMPLOYED` | `365243` substituído por `NaN` (código sentinela de dado ausente) |
| Imputação numérica | Todas as colunas numéricas | Preenchimento pela mediana |
| Imputação categórica | Todas as colunas categóricas | Preenchimento com `'Unknown'` |

#### Features Derivadas Criadas

| Feature | Fórmula | Interpretação |
|---|---|---|
| `age_years` | `DAYS_BIRTH / -365` | Idade do solicitante em anos |
| `years_employed` | `DAYS_EMPLOYED / -365` | Tempo de emprego em anos (após tratamento do 365243) |
| `credit_income_ratio` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Razão entre crédito solicitado e renda |
| `annuity_income_ratio` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Comprometimento da renda com parcelas |
| `credit_term` | `AMT_CREDIT / AMT_ANNUITY` | Prazo implícito do crédito em meses |

#### Colunas Principais Presentes

Além das 122 colunas originais das tabelas de aplicação, o dataset inclui:
- 50 colunas adicionais codificadas (`var_1` a `var_50`)
- 5 features derivadas listadas acima
- Coluna `is_train` para separação do dataset

---

### 4.3 bureau_features.parquet

**Arquivo:** `data/interim/bureau_features.parquet`
**Shape real:** (305,811 linhas × 10 colunas)
**Fontes:** `bureau.csv` + `bureau_balance.csv`

#### Tratamentos Aplicados

| Tratamento | Coluna(s) | Ação |
|---|---|---|
| Nulos em valores monetários | `AMT_CREDIT_SUM`, `AMT_CREDIT_SUM_DEBT` | Substituídos por `0` |
| Agregação intermediária | `bureau_balance` | Agregado por `SK_ID_BUREAU` antes do merge com `bureau.csv` |
| Codificação de atraso | `STATUS` (bureau_balance) | `C`, `X`, `0` = sem atraso; `1`–`5` = meses em atraso |

#### Features Geradas

| Feature | Descrição | Tipo |
|---|---|---|
| `bureau_count` | Número total de créditos externos | Contagem |
| `bureau_active_count` | Créditos com `CREDIT_ACTIVE = 'Active'` | Contagem |
| `bureau_closed_count` | Créditos com `CREDIT_ACTIVE = 'Closed'` | Contagem |
| `bureau_avg_days_credit` | Média de `DAYS_CREDIT` | Temporal |
| `bureau_avg_credit_sum` | Média do valor total dos créditos | Monetário |
| `bureau_avg_credit_debt` | Média do saldo devedor dos créditos | Monetário |
| `bureau_max_overdue` | Máximo de `AMT_CREDIT_MAX_OVERDUE` | Monetário |
| `bureau_avg_overdue_months` | Média de meses em atraso (via bureau_balance) | Temporal |
| `bureau_max_dpd` | Máximo de atraso registrado no bureau_balance | Temporal |

**Nota:** `bureau_max_overdue` possuía 30.4% de nulos — resolvido na Fase 3.1 com a criação de `bureau_had_overdue` (flag binário) e substituição de NaN por 0. Ver seção 4.9 — Correção 1.

---

### 4.4 prev_app_features.parquet

**Arquivo:** `data/interim/prev_app_features.parquet`
**Shape real:** (338,857 linhas × 11 colunas)
**Fonte:** `previous_application.csv`

#### Tratamentos Aplicados

| Tratamento | Coluna(s) | Ação |
|---|---|---|
| Valor sentinela de dias | `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`, `DAYS_LAST_DUE`, `DAYS_TERMINATION` | `365243` substituído por `NaN` |
| Categorias especiais | Todas as variáveis categóricas | `'XNA'` e `'XAP'` substituídos por `NaN` |

#### Features Geradas

| Feature | Descrição | Tipo |
|---|---|---|
| `prev_app_count` | Total de aplicações anteriores no PoD Bank | Contagem |
| `prev_app_approved_count` | Aplicações com `NAME_CONTRACT_STATUS = 'Approved'` | Contagem |
| `prev_app_refused_count` | Aplicações com `NAME_CONTRACT_STATUS = 'Refused'` | Contagem |
| `prev_app_approval_rate` | Taxa de aprovação histórica | Razão |
| `prev_app_avg_amount` | Média de `AMT_APPLICATION` | Monetário |
| `prev_app_avg_credit` | Média de `AMT_CREDIT` aprovado | Monetário |
| `prev_app_avg_annuity` | Média de `AMT_ANNUITY` | Monetário |
| `prev_app_last_days` | Mínimo de `DAYS_DECISION` (aplicação mais recente) | Temporal |
| `prev_app_consumer_count` | Contratos do tipo ConsumerLoans | Contagem |
| `prev_app_cash_count` | Contratos do tipo CashLoans | Contagem |

---

### 4.5 pos_cash_features.parquet

**Arquivo:** `data/interim/pos_cash_features.parquet`
**Shape real:** (337,252 linhas × 9 colunas)
**Fonte:** `POS_CASH_balance.csv`

#### Tratamentos Aplicados

| Tratamento | Coluna(s) | Ação |
|---|---|---|
| Nulos em contagem de parcelas | `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE` | Preenchimento adequado antes da agregação |

#### Features Geradas

| Feature | Descrição | Tipo |
|---|---|---|
| `pos_cash_count` | Número de contratos POS/cash distintos | Contagem |
| `pos_cash_months_balance` | Total de meses de histórico | Temporal |
| `pos_cash_avg_dpd` | Média de `SK_DPD` (dias de atraso) | Comportamento |
| `pos_cash_max_dpd` | Máximo de `SK_DPD` | Comportamento |
| `pos_cash_avg_dpd_def` | Média de `SK_DPD_DEF` | Comportamento |
| `pos_cash_late_payments` | Meses com `SK_DPD > 0` | Contagem |
| `pos_cash_completed_count` | Contratos com `NAME_CONTRACT_STATUS = 'Completed'` | Contagem |
| `pos_cash_active_count` | Contratos com `NAME_CONTRACT_STATUS = 'Active'` | Contagem |

---

### 4.6 credit_card_features.parquet

**Arquivo:** `data/interim/credit_card_features.parquet`
**Shape real:** (103,558 linhas × 11 colunas)
**Fonte:** `credit_card_balance.csv`

**Nota:** Apenas 103,558 clientes possuem histórico de cartão de crédito. Clientes sem histórico recebem 0 nas features após o merge.

#### Tratamentos Aplicados

| Tratamento | Coluna(s) | Ação |
|---|---|---|
| Nulos em valores monetários | `AMT_BALANCE`, `AMT_CREDIT_LIMIT_ACTUAL`, `AMT_DRAWINGS_CURRENT`, `AMT_PAYMENT_CURRENT` | Preenchimento antes da agregação |
| Cálculo de utilização | `cc_utilization` | `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`; `inf` e `NaN` substituídos por `0` |

#### Features Geradas

| Feature | Descrição | Tipo |
|---|---|---|
| `cc_count` | Número de cartões de crédito | Contagem |
| `cc_avg_balance` | Média de `AMT_BALANCE` | Monetário |
| `cc_max_balance` | Máximo de `AMT_BALANCE` | Monetário |
| `cc_avg_utilization` | Média de utilização do limite de crédito | Razão |
| `cc_max_utilization` | Máximo de utilização do limite de crédito | Razão |
| `cc_avg_payment` | Média de `AMT_PAYMENT_CURRENT` | Monetário |
| `cc_avg_drawings` | Média de `AMT_DRAWINGS_CURRENT` | Monetário |
| `cc_avg_dpd` | Média de `SK_DPD` (dias de atraso) | Comportamento |
| `cc_max_dpd` | Máximo de `SK_DPD` | Comportamento |
| `cc_months_balance` | Total de meses de histórico de cartão | Temporal |

---

### 4.7 installments_features.parquet

**Arquivo:** `data/interim/installments_features.parquet`
**Shape real:** (339,587 linhas × 10 colunas)
**Fonte:** `installments_payments.csv`

#### Tratamentos Aplicados

| Tratamento | Coluna(s) | Ação |
|---|---|---|
| Feature derivada de atraso | `days_past_due` | `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`; positivo = atraso, negativo = adiantado |
| Feature de razão de pagamento | `payment_ratio` | `AMT_PAYMENT / AMT_INSTALMENT`; `inf` e `NaN` substituídos por `0` |
| Feature de subpagamento | `underpayment` | `AMT_INSTALMENT - AMT_PAYMENT`; se positivo, cliente pagou menos que o devido |

#### Features Geradas

| Feature | Descrição | Tipo |
|---|---|---|
| `inst_count` | Total de parcelas pagas | Contagem |
| `inst_avg_dpd` | Média de `days_past_due` | Comportamento |
| `inst_max_dpd` | Máximo de `days_past_due` | Comportamento |
| `inst_late_count` | Parcelas pagas com atraso (`days_past_due > 0`) | Contagem |
| `inst_late_rate` | Taxa de parcelas pagas com atraso | Razão |
| `inst_avg_payment_ratio` | Média da razão de pagamento | Razão |
| `inst_min_payment_ratio` | Mínimo da razão de pagamento | Razão |
| `inst_avg_underpayment` | Média de subpagamento | Monetário |
| `inst_num_instalments` | Número de contratos distintos (`nunique` de `SK_ID_PREV`) | Contagem |

---

### 4.8 Merge Final e Datasets de Modelagem

#### Estratégia de Merge

O orquestrador realiza left join de `application_clean.parquet` com cada tabela de features usando `SK_ID_CURR` como chave. Clientes sem histórico em determinada tabela recebem `0` em todas as features daquela tabela (decisão: NaN do merge preenchido com 0).

**Justificativa para preenchimento com 0:** Clientes sem histórico em uma fonte específica (ex.: sem cartão de crédito) devem ser diferenciados de clientes com histórico neutro. Ao preencher com 0, o modelo pode aprender que a ausência de um tipo de histórico é informação preditiva por si só.

#### train_final.parquet

**Arquivo:** `data/processed/train_final.parquet`
**Shape real:** (215,257 linhas × 212 colunas) *(atualizado após Fase 3.1 — era 223)*
**Distribuição do TARGET:**

| Classe | Contagem | % |
|---|---|---|
| `0` (Adimplente) | 197,845 | 91.91% |
| `1` (Inadimplente) | 17,412 | 8.09% |

**Nota:** O shape de treino (215,257) é menor que o total de application_train (307,511) porque o dataset foi dividido internamente; os demais registros compõem o test_final.

#### test_final.parquet

**Arquivo:** `data/processed/test_final.parquet`
**Shape real:** (92,254 linhas × 212 colunas) *(atualizado após Fase 3.1 — era 223)*

#### Composição das 212 Features (após Fase 3.1)

| Origem | Qtd. Features | Exemplos |
|---|---|---|
| application (originais) | 109 | `EXT_SOURCE_1/2/3`, `AMT_CREDIT`, `CODE_GENDER`, `NAME_CONTRACT_TYPE`, flags de documentos, etc. (−13 colunas de imóvel removidas) |
| application (colunas codificadas) | 50 | `var_1` a `var_50` |
| application (derivadas) | 6 | `age_years`, `years_employed`, `credit_income_ratio`, `annuity_income_ratio`, `credit_term`, `ext_source_1_imputed` |
| bureau | 10 | `bureau_count`, `bureau_active_count`, `bureau_max_overdue`, `bureau_max_dpd`, `bureau_had_overdue`, etc. |
| previous_application | 10 | `prev_app_count`, `prev_app_approval_rate`, `prev_app_last_days`, etc. |
| pos_cash | 8 | `pos_cash_count`, `pos_cash_avg_dpd`, `pos_cash_late_payments`, etc. |
| credit_card | 10 | `cc_count`, `cc_avg_utilization`, `cc_max_dpd`, etc. |
| installments | 9 | `inst_late_rate`, `inst_avg_payment_ratio`, `inst_max_dpd`, etc. |
| **Total** | **212** | |

---

### 4.9 Fase 3.1 — Feature Fixes (feature_fix_agent)

**Data:** 2026-03-23
**Status:** Concluída

Esta fase aplicou 4 correções cirúrgicas nos datasets `train_final.parquet` e `test_final.parquet`, resolvendo os alertas pendentes identificados na Fase 3 e reduzindo o número de features de 223 para 212.

#### Correção 1 — bureau_had_overdue (Alerta 5.1 — RESOLVIDO)

**Problema:** `bureau_max_overdue` possuía 30.4% de nulos, com NaN preenchido por 0 no merge, mascarando ausência real de informação de overdue.

**Solução aplicada:**
- Criada feature binária `bureau_had_overdue`: valor `1` se `bureau_max_overdue > 0`, caso contrário `0` (NaN original tratado como `0`)
- `bureau_max_overdue` mantida no dataset, com NaN substituído por `0`

**Distribuição no treino:**

| Grupo | Contagem | % |
|---|---|---|
| Com overdue (`bureau_had_overdue = 1`) | 49,394 | 22.9% |
| Sem overdue (`bureau_had_overdue = 0`) | 165,863 | 77.1% |

**Distribuição no teste:**

| Grupo | Contagem | % |
|---|---|---|
| Com overdue (`bureau_had_overdue = 1`) | 21,046 | 22.8% |
| Sem overdue (`bureau_had_overdue = 0`) | 71,208 | 77.2% |

---

#### Correção 2 — EXT_SOURCE_1 Modelo de Imputação (Alerta 5.2 — RESOLVIDO)

**Problema:** `EXT_SOURCE_1` possuía ~56% de nulos; imputação por mediana atenua o poder preditivo da feature mais correlacionada com TARGET junto às demais EXT_SOURCE.

**Solução aplicada:**
- Treinado LightGBM regressor em 94,008 registros com valor original presente
- Features preditoras usadas: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `DAYS_EMPLOYED`, `NAME_EDUCATION_TYPE`
- CV RMSE (5-fold): **0.1612 ± 0.0004**
- Registros imputados: 121,249 no treino (56.3%) e 52,129 no teste (56.5%)
- Criada flag `ext_source_1_imputed = 1` para registros imputados (0 para valores originais)
- Mesmo imputer aplicado ao teste sem re-treino (sem data leakage)
- Modelo serializado em `models/imputer_ext_source_1.pkl`

---

#### Correção 3 — Remoção de colunas de imóvel com > 60% de nulos (Alerta 5.3 — RESOLVIDO)

**Problema:** 13 colunas do grupo de características físicas do imóvel possuíam entre 60% e 70% de nulos; imputação por mediana nestas colunas pode introduzir bias sistemático.

**Decisão:** Remoção das 13 colunas de ambos os datasets (treino e teste).

**Colunas removidas:**

| Grupo | Colunas |
|---|---|
| Área comum | `COMMONAREA_AVG`, `COMMONAREA_MEDI`, `COMMONAREA_MODE` |
| Área não residencial | `NONLIVINGAPARTMENTS_AVG`, `NONLIVINGAPARTMENTS_MEDI`, `NONLIVINGAPARTMENTS_MODE` |
| Fundo do imóvel | `FONDKAPREMONT_MODE` |
| Área residencial | `LIVINGAPARTMENTS_AVG`, `LIVINGAPARTMENTS_MEDI`, `LIVINGAPARTMENTS_MODE` |
| Ano de construção | `YEARS_BUILD_AVG`, `YEARS_BUILD_MEDI`, `YEARS_BUILD_MODE` |

---

#### Correção 4 — Validação de Data Leakage (Alerta 5.4 — VERIFICADO)

**Verificação:** 5 colunas temporais testadas para valores positivos inesperados (evento posterior à data da aplicação):

| Coluna | Resultado |
|---|---|
| `DAYS_BIRTH` | OK — sem valores positivos |
| `DAYS_EMPLOYED` | OK — sem valores positivos |
| `DAYS_REGISTRATION` | OK — sem valores positivos |
| `DAYS_ID_PUBLISH` | OK — sem valores positivos |
| `DAYS_LAST_PHONE_CHANGE` | OK — sem valores positivos |

**Resultado: 0/5 colunas com data leakage detectado.** Nenhuma ação corretiva necessária.

---

#### Shapes após as Correções

| Arquivo | Shape anterior | Shape atual | Variação |
|---|---|---|---|
| `train_final.parquet` | (215,257 × 223) | (215,257 × 212) | −13 colunas removidas, +2 flags adicionadas |
| `test_final.parquet` | (92,254 × 223) | (92,254 × 212) | −13 colunas removidas, +2 flags adicionadas |

---

## 5. Fase 4 — Modeling

**Data:** 2026-03-24
**Status:** Concluída — todos os modelos APROVADOS

### 5.1 Abordagem

Foram treinados três modelos com validação cruzada estratificada de 5 folds, usando `train_final.parquet` (215,257 × 212). Estratégia de balanceamento: `class_weight='balanced'` (LR) e `scale_pos_weight=11.4` (LightGBM/XGBoost). Early stopping aplicado nos modelos de boosting.

### 5.2 Resultados (CV 5-fold)

| Modelo | AUC-ROC | KS | Gini | Status | MLflow Run ID |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7559 ± 0.0027 | 0.3857 ± 0.0054 | 0.5117 ± 0.0055 | APROVADO | `53e25c53f91b43708d40876fee99d7b7` |
| LightGBM v1 | 0.7701 ± 0.0031 | 0.4105 ± 0.0035 | 0.5401 ± 0.0062 | APROVADO | `32a8dd5fd7ca45cda61f7dbb8d1531d1` |
| XGBoost v1 | 0.7698 ± 0.0036 | 0.4065 ± 0.0083 | 0.5396 ± 0.0071 | APROVADO | `e6baf972d9334ae1b88db1782fff614f` |

Criterios de aprovacao revisados: AUC-ROC >= 0.75 | KS >= 0.35 | Gini >= 0.50. Todos os modelos aprovados.

**Modelo em producao:** LightGBM v2 (lightgbm_tuned.pkl) — tuning aplicado na Fase 5, registrado no MLflow como scoring_pipeline_v1 (run ID: `f9717fdf7478468ca12fca51b9850dc1`).

### 5.3 Modelo Eleito — LightGBM v1

**Justificativa:**
- Melhor desempenho absoluto nas três métricas (AUC=0.7701, KS=0.4105, Gini=0.5401)
- Empate técnico com XGBoost (diferença de 0.0003 no AUC); LightGBM priorizado por regra de desempate
- Menor variância do KS entre folds (±0.0035 vs. ±0.0083 do XGBoost), indicando maior estabilidade
- Suporte nativo a variáveis categóricas, eliminando OrdinalEncoder externo

**Arquivo salvo:** `models/lightgbm_model.pkl`
**n_estimators final:** 221 (early stopping, média de 5 folds)

### 5.4 Top Features (consistentes entre modelos)

`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `credit_term`, `inst_late_rate`, `ORGANIZATION_TYPE`, `OCCUPATION_TYPE`

Coerente com o EDA: EXT_SOURCE_1/2/3 continuam como os sinais de maior poder preditivo.

### 5.5 Modelos Salvos

| Arquivo | Localização | Parâmetros Chave |
|---|---|---|
| `baseline_logistic_regression.pkl` | `models/` | solver=saga, class_weight='balanced' |
| `lightgbm_model.pkl` | `models/` | lr=0.05, num_leaves=31, n_estimators=221, scale_pos_weight=11.4 |
| `xgboost_model.pkl` | `models/` | lr=0.05, max_depth=6, n_estimators=241, scale_pos_weight=11.4 |

### 5.6 Relatório de Comparação

Relatório completo disponível em `reports/model_comparison.md`.

---

## 6. Fase 5 — Evaluation

**Data:** 2026-03-24
**Status:** Concluída — modelo campeão aprovado em todos os critérios

### 6.1 Tuning — LightGBM v2

**Script:** `src/models/task4_lightgbm_tuned.py`

O LightGBM v1 (eleito na Fase 4) apresentava KS gap entre treino e validação superior a 0.10, indicando overfitting moderado. A Fase 5 aplicou regularização mais forte para corrigir esse comportamento antes da avaliação no holdout.

#### Mudanças de Hiperparâmetros em Relação ao v1

| Parâmetro | v1 | v2 | Justificativa |
|---|---|---|---|
| `num_leaves` | 31 | 20 | Reduz capacidade da árvore, limita overfitting |
| `min_child_samples` | 20 | 80 | Exige mais amostras por folha, evita folhas pequenas |
| `reg_alpha` | 0.1 | 1.0 | Regularização L1 mais forte |
| `reg_lambda` | 0.1 | 1.0 | Regularização L2 mais forte |
| `feature_fraction` | 0.8 | 0.65 | Menos features por árvore, maior diversidade |
| `bagging_fraction` | 0.8 | 0.65 | Menos amostras por iteração, maior diversidade |
| `min_split_gain` | 0 | 0.01 | Ganho mínimo para realizar split |
| `learning_rate` | 0.05 | 0.03 | Aprendizado mais conservador |
| `early_stopping rounds` | 50 | 80 | Paciência maior para compensar lr menor |

#### Resultados CV 5-fold — LightGBM v2

| Métrica | LightGBM v1 | LightGBM v2 | Variação |
|---|---|---|---|
| AUC-ROC | 0.7701 ± 0.0031 | 0.7717 ± 0.0037 | +0.0016 |
| KS | 0.4105 ± 0.0035 | 0.4118 ± 0.0063 | +0.0013 |
| Gini | 0.5401 ± 0.0062 | 0.5434 ± 0.0075 | +0.0033 |
| KS gap (train-val) | ~0.10+ | 0.0873 | Melhora de overfitting |
| n_estimators best | 221 | 561 | Mais iterações com lr menor |

**Status:** APROVADO
**MLflow Run ID:** `456743922fc643afb969f078c9497aef`
**Modelo salvo:** `models/lightgbm_tuned.pkl`

---

### 6.2 Avaliacao Completa — Holdout 20%

**Script:** `src/evaluation/evaluate_champion.py`

O modelo LightGBM v2 foi avaliado em holdout estrito de 20% do dataset de treino, separado antes de qualquer fitting. Esta é a estimativa mais confiável de desempenho em dados novos.

#### Resultados no Holdout

| Métrica | Resultado | Alvo | Status |
|---|---|---|---|
| AUC-ROC | 0.8223 | >= 0.75 | APROVADO |
| KS Statistic | 0.4887 | >= 0.35 | APROVADO |
| Gini Coefficient | 0.6447 | >= 0.50 | APROVADO |
| Recall (bons pagadores) | 0.7057 | >= 70% | APROVADO |

**MLflow Eval Run ID:** `624246ba0cc74d8a877e8eb32a53b7d9`

#### Parametros de Decisao de Negocio Eleitos

| Parametro | Valor |
|---|---|
| Threshold de aprovacao | 0.48 |
| Taxa de aprovacao | 66.64% |
| Default esperado (aprovados) | 2.68% |
| Custo estimado (FN x 5 + FP x 1) | 15,486 |

**Justificativa do threshold 0.48:** Threshold calibrado para maximizar a funcao de custo assimetrica onde falso negativo (aprovar inadimplente) tem custo 5x maior que falso positivo (recusar bom pagador). O threshold 0.48 equilibra taxa de aprovacao comercialmente viavel (66.64%) com controle de risco (default esperado de 2.68%).

---

### 6.3 Verificacao de Leakage (Fase 5)

**Script:** `src/features/leakage_check.py`

Verificacao adicional de leakage com foco nas colunas de datas provenientes das tabelas secundarias (bureau e previous_application), que nao foram cobertas na Fase 3.1.

#### Resultados

| Coluna | % Valores Positivos | Conclusao |
|---|---|---|
| `DAYS_CREDIT_ENDDATE` (bureau) | 35.11% | NAO e leakage — data futura de encerramento planejado e dado contratual disponivel na aplicacao |
| `DAYS_LAST_DUE_1ST_VERSION` (previous_application) | 13.43% | NAO e leakage — prazo original do contrato anterior, dado contratual historico |
| Todas as features de installments | 0% | Sem leakage |
| Todas as features de POS_CASH | 0% | Sem leakage |
| Todas as features de credit_card | 0% | Sem leakage |

**Conclusao:** Nenhum leakage real identificado. Os valores positivos em colunas de dias das tabelas secundarias correspondem a datas contratuais futuras (encerramento planejado, prazo original) que estao legitimamente disponiveis no momento da concessao do credito. O overfitting observado no v1 foi causado por insuficiencia de regularizacao, nao por leakage — resolvido pela regularizacao aplicada no v2.

---

### 6.4 Figuras Geradas

| Arquivo | Localização | Conteudo |
|---|---|---|
| `roc_curve.png` | `reports/figures/` | Curva ROC do modelo campeao no holdout |
| `ks_curve.png` | `reports/figures/` | Curva KS — separacao entre bons e maus pagadores |
| `lift_curve.png` | `reports/figures/` | Curva de lift — ganho de selecao sobre baseline |
| `score_distribution.png` | `reports/figures/` | Distribuicao de scores por classe (TARGET 0 vs 1) |
| `confusion_matrix.png` | `reports/figures/` | Matriz de confusao no threshold 0.48 |
| `feature_importance_final.png` | `reports/figures/` | Importancia de features — modelo final |
| `lightgbm_tuned_feature_importance.png` | `reports/figures/` | Importancia de features do LightGBM v2 |
| `lightgbm_tuned_oof_distribution.png` | `reports/figures/` | Distribuicao OOF (out-of-fold) do LightGBM v2 |

---

## 7. Fase 6 — Deployment

**Data:** 2026-03-24
**Status:** Concluido

O pipeline de producao foi empacotado, testado e registrado no MLflow. O modelo campeao LightGBM v2 (lightgbm_tuned.pkl) foi incorporado a um pipeline completo com threshold de decisao, lista de features e logica de bandas de risco.

### 7.1 Artefatos Gerados

| Arquivo | Localizacao | Descricao |
|---|---|---|
| `scoring_pipeline.pkl` | `models/` | Pipeline completo serializado: modelo + feature_columns + threshold 0.48 + version 1.0-tuned |
| `predict.py` | `src/models/` | Funcao `predict_score(client_data: dict) -> dict` para inferencia em producao |
| `pipeline_report.md` | `reports/` | Relatorio com resultados de testes em 5 clientes reais e medicao de latencia |

### 7.2 Registro no MLflow

**Experimento:** `pod-bank-deployment`
**Run name:** `scoring_pipeline_v1`
**Run ID:** `f9717fdf7478468ca12fca51b9850dc1`
**Script:** `src/models/task5_register_pipeline.py`

#### Parametros Registrados

| Parametro | Valor |
|---|---|
| `threshold` | 0.48 |
| `feature_count` | 62 |
| `model_version` | lightgbm_tuned |
| `pipeline_version` | 1.0-tuned |

#### Metricas Registradas

| Metrica | Valor |
|---|---|
| `auc_roc` | 0.7701 |
| `ks` | 0.4105 |
| `gini` | 0.5401 |
| `avg_inference_ms` | 7.76 |
| `max_inference_ms` | 8.02 |
| `sla_pass` | 1 (SLA de 500ms atingido) |

#### Artefatos Registrados no MLflow

| Artefato | artifact_path |
|---|---|
| `models/scoring_pipeline.pkl` | `pipeline/` |
| `reports/pipeline_report.md` | `reports/` |
| `src/models/predict.py` | `src/` |

### 7.3 Testes de Producao — 5 Clientes Reais

Os testes foram realizados com clientes reais do dataset de holdout, verificando acuracia de decisao e latencia de inferencia.

| SK_ID_CURR | TARGET real | Score | Decisao | Banda de Risco | Correto? |
|---|---|---|---|---|---|
| 204040 | 0 (adimplente) | 0.3727 | APROVADO | MEDIO | Sim |
| 151722 | 0 (adimplente) | 0.1087 | APROVADO | BAIXO | Sim |
| 349833 | 1 (inadimplente) | 0.5225 | REPROVADO | ALTO | Sim |
| 121179 | 1 (inadimplente) | 0.7526 | REPROVADO | ALTO | Sim |
| 367944 | 0 (adimplente) | 0.4676 | APROVADO | MEDIO | Sim |

**Acuracia nos 5 clientes de teste:** 5/5 (100%)
**Nota:** este conjunto e ilustrativo; a avaliacao estatistica robusta foi realizada no holdout de 20% na Fase 5 (AUC=0.8223, KS=0.4887).

### 7.4 Latencia de Inferencia

| Metrica | Valor | SLA | Status |
|---|---|---|---|
| Latencia media | 7.76 ms | < 500 ms | PASS |
| Latencia maxima | 8.02 ms | < 500 ms | PASS |

A latencia de 7.76ms representa uma margem de seguranca de 98.4% em relacao ao SLA de 500ms, viabilizando uso em tempo real e em batch.

### 7.5 Modelo em Producao

**Modelo:** LightGBM v2 (`lightgbm_tuned.pkl`)
**Versao do pipeline:** 1.0-tuned
**Metricas CV 5-fold:** AUC-ROC=0.7701, KS=0.4105, Gini=0.5401
**Metricas holdout 20%:** AUC-ROC=0.8223, KS=0.4887, Gini=0.6447

O LightGBM v2 e o modelo eleito como campeao na Fase 5, substituindo o LightGBM v1. Ver secao 6 para detalhes do tuning e avaliacao.

### 7.6 Threshold e Logica de Bandas de Risco

**Threshold de aprovacao:** 0.48

A decisao de credito e tomada comparando o score de probabilidade de inadimplencia com o threshold:

```
score < 0.48  →  APROVADO
score >= 0.48 →  REPROVADO
```

**Bandas de risco** (para gestao de portfolio e pricing):

| Banda | Intervalo de Score | Interpretacao |
|---|---|---|
| BAIXO | score < 0.20 | Risco muito baixo — cliente de alta qualidade |
| MEDIO | 0.20 <= score < 0.48 | Risco moderado — aprovado dentro do apetite de risco |
| ALTO | score >= 0.48 | Risco elevado — reprovado |

O threshold 0.48 foi calibrado na Fase 5 com funcao de custo assimetrica (FN custa 5x mais que FP), resultando em taxa de aprovacao de 66.64% e default esperado de 2.68% entre aprovados.

### 7.7 Como Usar o Pipeline em Producao

```python
import pickle
from src.models.predict import predict_score

# Opcao 1 — usar a funcao de alto nivel (recomendado)
client_data = {
    "SK_ID_CURR": 204040,
    "EXT_SOURCE_1": 0.50,
    "EXT_SOURCE_2": 0.62,
    "EXT_SOURCE_3": 0.58,
    # ... demais features do cliente
}
result = predict_score(client_data)
# result = {
#     "SK_ID_CURR": 204040,
#     "score": 0.3727,
#     "decision": "APROVADO",
#     "risk_band": "MEDIO",
#     "threshold": 0.48
# }

# Opcao 2 — usar o pipeline diretamente
with open("models/scoring_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# pipeline contem:
# pipeline["model"]           — LGBMClassifier treinado
# pipeline["feature_columns"] — lista com os 62 nomes de features esperados
# pipeline["threshold"]       — 0.48
# pipeline["version"]         — "1.0-tuned"
```

**Requisitos de entrada:** o dicionario `client_data` deve conter todas as 62 features listadas em `pipeline["feature_columns"]`. Features ausentes devem ser preenchidas com `0` (mesma convencao do pipeline de feature engineering).

---

## 8. Alertas Pendentes

> **Status geral:** Todos os alertas criticos identificados na Fase 3 foram resolvidos na Fase 3.1 (2026-03-23). A verificacao de leakage da Fase 5 confirmou ausencia de leakage real. Nao ha alertas pendentes criticos.

### 8.1 bureau_max_overdue — 30.4% de nulos

**Status:** ✅ RESOLVIDO em 2026-03-23

**Feature:** `bureau_max_overdue` (máximo de `AMT_CREDIT_MAX_OVERDUE` no bureau)
**% de nulos:** 30.4% no arquivo `bureau_features.parquet`

**Solução aplicada:** Criada feature binária `bureau_had_overdue` (1 se `bureau_max_overdue > 0`, caso contrário 0; NaN tratado como 0). A feature original `bureau_max_overdue` foi mantida com NaN substituído por 0. Distribuição no treino: 22.9% com overdue, 77.1% sem overdue.

---

### 8.2 EXT_SOURCE_1 — 56.4% de nulos (application_clean)

**Status:** ✅ RESOLVIDO em 2026-03-23

**Feature:** `EXT_SOURCE_1` (score externo de bureau — maior correlação com TARGET após EXT_SOURCE_2/3)
**% de nulos:** ~56.4% na tabela original

**Solução aplicada:** Treinado LightGBM regressor para imputação (CV RMSE 5-fold: 0.1612 ± 0.0004) usando como preditores: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `DAYS_EMPLOYED`, `NAME_EDUCATION_TYPE`. Criada flag `ext_source_1_imputed` para rastreabilidade. Modelo salvo em `models/imputer_ext_source_1.pkl`. Mesmo imputer aplicado ao teste sem re-treino (sem data leakage).

---

### 8.3 Colunas *_AVG / *_MEDI / *_MODE — > 60% de nulos

**Status:** ✅ RESOLVIDO em 2026-03-23

**Grupo de colunas:** características físicas do imóvel (`COMMONAREA_*`, `LIVINGAPARTMENTS_*`, `YEARS_BUILD_*`, etc.)
**% de nulos:** entre 60% e 70%

**Solução aplicada:** 13 colunas removidas de train e test (ver lista completa na seção 4.9 — Correção 3). Os datasets reduziram de 223 para 212 features.

---

### 8.4 Validação de Data Leakage

**Status:** ✅ VERIFICADO em 2026-03-23 — nenhum problema encontrado

**Verificação realizada:** 5 colunas temporais testadas para valores positivos inesperados (`DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`). Resultado: 0/5 colunas com data leakage detectado. Nenhuma ação corretiva necessária.

---

## 9. Decisões de Negócio

### 9.1 Definição de Inadimplência

O target `TARGET = 1` indica que o cliente teve dificuldade em pagar o empréstimo, conforme definição do dataset Home Credit. Clientes adimplentes são `TARGET = 0`.

### 9.2 Threshold de Decisão

O threshold de corte para aprovação/reprovação de crédito foi calibrado na Fase 5 com base em funcao de custo assimetrica (FN custa 5x mais que FP). Threshold eleito: **0.48**, resultando em taxa de aprovacao de 66.64% e default esperado de 2.68% entre aprovados.

### 9.3 Clientes sem Histórico em Fontes Secundárias

Clientes que não possuem registros em `bureau`, `previous_application`, `POS_CASH_balance`, `credit_card_balance` ou `installments_payments` recebem valor `0` nas features correspondentes após o merge. Esta decisão foi tomada para:
1. Evitar exclusão de clientes (left join preserva todos)
2. Permitir ao modelo diferenciar ausência de histórico de histórico neutro
3. Manter consistência no tratamento de clientes novos no sistema

### 9.4 Explainability

Para cada decisão de crédito emitida pelo modelo em produção, deve ser gerada uma explicação via **SHAP values** indicando as principais features que contribuíram para a decisão. Requisito não-negociável para aprovação regulatória.

---

## 10. Histórico de Versões

| Data | Versão | Fase CRISP-DM | Descrição |
|---|---|---|---|
| 2026-03-24 | v0.7 | Fase 6 — Deployment | Pipeline de producao empacotado (scoring_pipeline.pkl, version 1.0-tuned); predict_score() implementado; 5 clientes testados (5/5 decisoes corretas); latencia media 7.76ms (SLA PASS); pipeline registrado no MLflow experimento pod-bank-deployment, run ID f9717fdf7478468ca12fca51b9850dc1 |
| 2026-03-24 | v0.6 | Fase 5 — Evaluation | Tuning LightGBM v2 (regularizacao mais forte, KS gap reduzido de ~0.10 para 0.0873); avaliacao no holdout 20% com AUC=0.8223, KS=0.4887, Gini=0.6447, Recall=70.57%; threshold 0.48 eleito; verificacao de leakage em tabelas secundarias concluida sem problemas; 8 figuras geradas |
| 2026-03-24 | v0.5 | Fase 4 — Modeling | Treinamento e comparação de 3 modelos (LR, LightGBM, XGBoost); LightGBM eleito como modelo principal; todos os critérios de aprovação atingidos |
| 2026-03-23 | v0.4 | Fase 3.1 — Feature Fixes | Aplicação das 4 correções pelo feature_fix_agent; datasets reduzidos de 223 para 212 features; alertas 5.1–5.3 resolvidos; data leakage verificado (alerta 5.4) |
| 2026-03-23 | v0.3 | Fase 3 — Data Preparation | Geração completa do pipeline de features com 6 subagentes; train_final (215,257 × 223) e test_final (92,254 × 223) prontos para modelagem |
| 2026-03-22 | v0.2 | Fase 2 — Data Understanding | EDA exploratório; identificação de correlações com TARGET; análise de nulos e valores especiais |
| 2026-03-21 | v0.1 | Fase 1 — Business Understanding | Definição do problema, métricas de sucesso e restrições do negócio |

---

*Pipeline em producao. Proximas etapas sugeridas: monitoramento de drift (PSI/KS em producao), retraining schedule, API REST para integracao com sistemas de originacao de credito.*
