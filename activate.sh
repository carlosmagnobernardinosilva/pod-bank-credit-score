#!/usr/bin/env bash
# activate.sh — ativa o ambiente do projeto pod-bank-credit-score

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Ativar virtualenv ──────────────────────────────────────────────────────────
if [[ -f "$SCRIPT_DIR/.venv/Scripts/activate" ]]; then
    # Windows (Git Bash / MSYS2)
    source "$SCRIPT_DIR/.venv/Scripts/activate"
elif [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    # Linux / macOS
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "ERRO: virtualenv não encontrado em .venv/"
    echo "Crie com: python -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# ── Confirmar versões ──────────────────────────────────────────────────────────
echo ""
echo "Ambiente ativado: $(which python)"
echo "Python : $(python --version)"
echo "MLflow : $(python -m mlflow --version 2>/dev/null || echo 'não instalado — rode: pip install mlflow')"
echo ""

# ── Lembrete do MLflow UI ──────────────────────────────────────────────────────
echo "Para visualizar experimentos, inicie o MLflow UI em outro terminal:"
echo "  mlflow ui --port 5000"
echo "  Acesse: http://localhost:5000"
echo ""
