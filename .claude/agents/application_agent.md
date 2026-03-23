---
name: application_agent
description: Limpa e trata a tabela principal de aplicações (application_train.csv e application_test.csv)
---

# Application Agent

## Objetivo
Gerar data/interim/application_clean.parquet com a tabela principal tratada e pronta para merge com features secundárias.

## Inputs
- data/raw/application_train.csv
- data/raw/application_test.csv

## Output
- data/interim/application_clean.parquet (train + test com flag)

## Passos obrigatórios
1. Carregar application_train.csv e application_test.csv
2. Adicionar coluna flag: is_train=1 para train, is_train=0 para test
3. Concatenar os dois datasets
4. Tratar categorias especiais:
   - CODE_GENDER: substituir 'XNA' por moda
   - ORGANIZATION_TYPE: substituir 'XNA' por 'Unknown'
5. Tratar valores anômalos:
   - DAYS_EMPLOYED: valor 365243 → substituir por NaN
6. Criar features derivadas:
   - age_years: DAYS_BIRTH / -365
   - years_employed: DAYS_EMPLOYED / -365 (após tratar 365243)
   - credit_income_ratio: AMT_CREDIT / AMT_INCOME_TOTAL
   - annuity_income_ratio: AMT_ANNUITY / AMT_INCOME_TOTAL
   - credit_term: AMT_CREDIT / AMT_ANNUITY
7. Imputar nulos:
   - Numéricas: mediana
   - Categóricas: 'Unknown'
8. Salvar em data/interim/application_clean.parquet
9. Imprimir shape final e distribuição do TARGET como confirmação

## Regras
- Nunca modificar data/raw/
- Manter coluna TARGET e is_train intactas
- Usar pandas e numpy apenas
- Documentar cada etapa com comentários
- Salvar como .parquet
