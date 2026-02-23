#!/bin/bash

SCRIPT_PY=${1:-"./src/executors/executor_dib.py"}
PARAMS_JSON=${2:-"./src/parameters/parameters_distil_bert_semi_hard_flip.json"}

if [ ! -f "$SCRIPT_PY" ]; then
    echo "Erro: Arquivo Python '$SCRIPT_PY' não encontrado."
    exit 1
fi

if [ ! -f "$PARAMS_JSON" ]; then
    echo "Erro: Arquivo JSON '$PARAMS_JSON' não encontrado."
    exit 1
fi

echo "Executando: $SCRIPT_PY"
echo "Com parâmetros: $PARAMS_JSON"
echo "--------------------------------"

python "$SCRIPT_PY" "$PARAMS_JSON"

echo "--------------------------------"
echo "Execução concluída!"