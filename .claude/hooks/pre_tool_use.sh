#!/usr/bin/env bash
# pre_tool_use.sh — Hook executado antes de cada uso de ferramenta pelo Claude Code.
# Bloqueia qualquer escrita ou edição em data/raw/ para proteger os dados originais.

# Lê o JSON completo do stdin (Claude Code passa os dados da ferramenta assim)
input=$(cat)

# Extrai o nome da ferramenta e o caminho do arquivo usando python3
tool_name=$(echo "$input" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('tool_name', ''))
")

file_path=$(echo "$input" | python3 -c "
import sys, json
data = json.load(sys.stdin)
# O caminho pode estar em campos diferentes dependendo da ferramenta
params = data.get('tool_input', {})
print(params.get('file_path', params.get('path', '')))
")

# Verifica se a ferramenta é de escrita ou edição
if [[ "$tool_name" == "write_file" || "$tool_name" == "edit_file" || \
      "$tool_name" == "str_replace_editor" || "$tool_name" == "str_replace_based_edit_tool" || \
      "$tool_name" == "create_file" ]]; then
    # Verifica se o caminho contém data/raw/
    if echo "$file_path" | grep -q "data/raw/"; then
        echo "BLOQUEADO: data/raw/ é somente leitura. Use data/interim/ ou data/processed/" >&2
        exit 1
    fi
fi

# Permite a ação para todos os outros casos
exit 0
