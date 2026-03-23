#!/usr/bin/env bash
# post_tool_use.sh — Hook executado após cada uso de ferramenta pelo Claude Code.
# Faz commit automático de arquivos criados em src/ ou data/processed/.

# Lê o JSON completo do stdin
input=$(cat)

# Extrai o nome da ferramenta
tool_name=$(echo "$input" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('tool_name', ''))
")

# Só age quando a ferramenta for write_file
if [[ "$tool_name" != "write_file" ]]; then
    exit 0
fi

# Extrai o caminho do arquivo criado
file_path=$(echo "$input" | python3 -c "
import sys, json
data = json.load(sys.stdin)
params = data.get('tool_input', {})
print(params.get('file_path', params.get('path', '')))
")

# Verifica se o caminho contém src/ ou data/processed/
if ! echo "$file_path" | grep -qE "(src/|data/processed/)"; then
    exit 0
fi

# Extrai apenas o nome do arquivo para a mensagem de commit
file_name=$(basename "$file_path")

# Navega para a raiz do repositório (onde o .git está)
repo_root=$(git -C "$(dirname "$file_path")" rev-parse --show-toplevel 2>/dev/null)
if [[ -z "$repo_root" ]]; then
    echo "AVISO: Não foi possível encontrar o repositório git para '$file_path'" >&2
    exit 0
fi

# Adiciona o arquivo ao staging area
if ! git -C "$repo_root" add "$file_path"; then
    echo "AVISO: git add falhou para '$file_path'" >&2
    exit 0
fi

# Realiza o commit automático
commit_msg="auto: ${file_name} criado por agente"
if git -C "$repo_root" commit -m "$commit_msg"; then
    echo "✔ Commit automático realizado: '${commit_msg}'"
else
    echo "AVISO: git commit falhou para '${file_name}' (arquivo pode já estar commitado ou sem mudanças)" >&2
fi

# Post hooks nunca devem bloquear — sempre sair com 0
exit 0
