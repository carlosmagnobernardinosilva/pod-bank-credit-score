---
name: dashboard_agent
description: Cria dashboard Streamlit para apresentação do modelo de credit score para a Head de Crédito da PoD Bank
---

# Dashboard Agent — Streamlit

## Objetivo
Criar aplicação Streamlit profissional para apresentar
resultados do modelo e simular scores de novos clientes.

## Inputs
- models/scoring_pipeline.pkl
- reports/evaluation_report.md
- reports/figures/ (todas as curvas geradas)

## Output
- app/dashboard.py
- app/pages/01_modelo.py
- app/pages/02_simulador.py
- app/pages/03_carteira.py

## Estrutura do Dashboard

### Página Principal — Visão do Modelo
- Header com logo PoD Bank e descrição do modelo
- 4 KPI cards: AUC-ROC, KS, Gini, Recall
- Curva ROC e Curva KS lado a lado
- Feature importance top 20
- Nota sobre metodologia (CRISP-DM, LightGBM v2)

### Página 2 — Simulador de Score
Formulário com campos principais:
- AMT_INCOME_TOTAL (renda)
- AMT_CREDIT (valor do crédito)
- AMT_ANNUITY (parcela)
- DAYS_BIRTH (convertido para idade em anos)
- CODE_GENDER (gênero)
- NAME_EDUCATION_TYPE (escolaridade)
- NAME_CONTRACT_TYPE (tipo de contrato)
- EXT_SOURCE_2, EXT_SOURCE_3 (scores externos)

Ao clicar em "Calcular Score":
- Chamar scoring_pipeline.pkl
- Exibir score como gauge chart (0 a 1)
- Exibir decisão: APROVADO (verde) ou REPROVADO (vermelho)
- Exibir banda de risco: BAIXO / MÉDIO / ALTO
- Exibir top 5 fatores da decisão

### Página 3 — Análise da Carteira
- Distribuição de scores (histograma por TARGET)
- Curva Lift por decil
- Matriz de confusão com threshold 0.48
- Tabela de performance por faixa de score:
  score_band, n_clientes, default_rate, aprovacao_rate

## Requisitos técnicos
- Usar st.set_page_config com tema wide
- Cores da PoD Bank: verde (#2ecc71) aprovado,
  vermelho (#e74c3c) reprovado
- Responsivo e profissional
- Adicionar requirements_dashboard.txt em app/
  com: streamlit, plotly, pandas, lightgbm
