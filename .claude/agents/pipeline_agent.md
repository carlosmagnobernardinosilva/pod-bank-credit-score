---
name: pipeline_agent
description: Serializa o pipeline completo de scoring para produção incluindo preprocessamento e modelo
---

# Pipeline Agent — Deployment

## Objetivo
Criar um pipeline reproduzível que recebe dados brutos
de um novo cliente e retorna o score de risco.

## Inputs
- models/lightgbm_tuned.pkl
- models/imputer_ext_source_1.pkl
- data/processed/train_final.parquet (para referência)

## Outputs
- models/scoring_pipeline.pkl
- src/models/predict.py
- reports/pipeline_report.md

## Passos obrigatórios

### 1. Criar src/models/predict.py
Função principal:

def predict_score(client_data: dict) -> dict:
    """
    Recebe dados brutos de um cliente e retorna score.

    Args:
        client_data: dicionário com features do cliente

    Returns:
        score: probabilidade de inadimplência (0 a 1)
        decision: APROVADO se score < 0.48, REPROVADO se >= 0.48
        risk_band: BAIXO (<0.20), MÉDIO (0.20-0.48), ALTO (>0.48)
        top_factors: top 5 features que mais impactaram a decisão
    """

### 2. Pipeline de preprocessamento
Incluir na ordem correta:
- Tratar valores especiais (365243, XNA, XAP)
- Aplicar imputer_ext_source_1 se EXT_SOURCE_1 ausente
- Criar features derivadas (age_years, credit_income_ratio etc.)
- Aplicar mesma seleção de features do IV
- Garantir mesma ordem de colunas do modelo

### 3. Banda de risco
Definir 3 faixas:
- BAIXO:  score < 0.20  → aprovado com limite máximo
- MÉDIO:  score 0.20-0.48 → aprovado com limite reduzido
- ALTO:   score >= 0.48 → reprovado

### 4. Teste do pipeline
Testar com 5 clientes reais do holdout:
- 2 adimplentes conhecidos
- 2 inadimplentes conhecidos
- 1 caso borderline (score próximo de 0.48)
Verificar se decisões fazem sentido.

### 5. Serializar pipeline completo
Salvar models/scoring_pipeline.pkl com:
- preprocessador
- imputer
- modelo
- threshold (0.48)
- lista de features esperadas

### 6. Relatório
Gerar reports/pipeline_report.md com:
- Tempo médio de inferência por cliente
- Confirmação que está abaixo de 500ms
- Resultado dos 5 testes
- Instruções de uso do pipeline
