# Detecting Ideological Bias (DIB)

Este projeto tem como objetivo a detecção de viés ideológico em textos utilizando modelos de Deep Learning e Processamento de Linguagem Natural (NLP), especificamente arquiteturas baseadas em Transformers como o **DistilBERT**.

## 🚀 Estrutura do Projeto

* `src/executors/`: Scripts principais para treinamento e inferência.
* `src/parameters/`: Arquivos JSON de configuração de hiperparâmetros.
* `src/dataset/`: Diretório destinado aos dados (ex: `abp_train.csv`).
* `rodar_modelo.sh`: Script de automação para execução do pipeline.

## 🛠️ Pré-requisitos

Certifique-se de ter o [Conda](https://docs.conda.io/) instalado.

### Configuração do Ambiente

1. Crie o ambiente a partir do arquivo `environment.yml`:
```bash
conda env create -f environment.yml

```


2. Ative o ambiente:
```bash
conda activate [nome-do-ambiente]

```



## 📦 Dataset

Devido às restrições de tamanho de arquivo do GitHub, datasets maiores que 100MB (como o `abp_train.csv`) não são rastreados diretamente no repositório. Certifique-se de baixar o dataset necessário e posicioná-lo em:
`src/dataset/abp_train.csv`

## 🏃 Como Executar

O projeto utiliza um script shell para facilitar a execução com diferentes configurações.

### Usando o Script de Automação

Dê permissão de execução (apenas na primeira vez):

```bash
chmod +x rodar_modelo.sh

```

Execute com os parâmetros padrão:

```bash
./rodar_modelo.sh

```

Ou passe caminhos personalizados via linha de comando:

```bash
./rodar_modelo.sh ./src/executors/meu_script.py ./src/parameters/config.json

```

## 🧠 Modelos Utilizados

O foco principal deste repositório é a implementação de:

* **DistilBERT** (com abordagens de Semi-Hard triplet loss ou Fine-tuning padrão).
* Processamento de enviesamento ideológico.

---
