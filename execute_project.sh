#!/bin/bash

# Ativa o ambiente conda (opcional, mas recomendado)
# Substitua 'nome_do_env' pelo nome do seu ambiente
# source activate nome_do_env

echo "Iniciando o executor DIB..."

python ./src/executors/executor_dib.py ./src/parameters/parameters_distil_bert_semi_hard.json

echo "Execução finalizada!"