---
name: orchestrator_agent
description: Orquestra toda a Fase 3 do CRISP-DM (Data Preparation)
invocando os subagentes especializados e fazendo o merge final
---

# Orchestrator Agent — Data Preparation

## Objetivo
Coordenar a execução de todos os subagentes de feature engineering
e consolidar o dataset final para modelagem.

## Outputs esperados
Ao final da execução, os seguintes arquivos devem existir:

data/interim/
├── application_clean.parquet
├── bureau_features.parquet
├── prev_app_features.parquet
├── pos_cash_features.parquet
├── credit_card_features.parquet
└── installments_features.parquet

data/processed/
├── train_final.parquet
└── test_final.parquet

## Passo 1 — Invocar subagentes em paralelo
Invocar os seguintes agentes usando Task():
- application_agent
- bureau_agent
- previous_app_agent
- pos_cash_agent
- credit_card_agent
- installments_agent

Aguardar todos finalizarem antes de avançar.

## Passo 2 — Validar outputs
Verificar se todos os 6 arquivos .parquet foram criados
em data/interim/. Se algum falhar, reportar qual falhou
e o motivo antes de continuar.

## Passo 3 — Merge final
Carregar application_clean.parquet como base e fazer
left join com cada feature table via SK_ID_CURR:
- Merge com bureau_features
- Merge com prev_app_features
- Merge com pos_cash_features
- Merge com credit_card_features
- Merge com installments_features

Preencher NaN do merge com 0 (cliente sem histórico = 0)

## Passo 4 — Separar train e test
Usar coluna is_train para separar:
- is_train == 1 → data/processed/train_final.parquet
- is_train == 0 → data/processed/test_final.parquet
Remover coluna is_train antes de salvar.

## Passo 5 — Relatório final
Imprimir:
- Shape de train_final e test_final
- Total de features geradas
- % de nulos remanescentes
- Distribuição do TARGET no train_final
- Confirmação de que está pronto para Fase 4 — Modeling
