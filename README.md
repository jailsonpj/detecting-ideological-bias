# Detecting Ideological Bias (DIB)

Este projeto tem como objetivo a detecção de viés ideológico em textos utilizando modelos de Deep Learning e Processamento de Linguagem Natural (NLP), especificamente arquiteturas baseadas em Transformers como o **DistilBERT**.

## 📚 Publicação e Citação

Este trabalho foi publicado no **XV Brazilian Workshop on Social Network Analysis and Mining (BrasNAM 2026)**. 

O artigo completo pode ser acessado através do link: [Artigo no SOL SBC](https://sol.sbc.org.br/index.php/brasnam/article/view/43242)

Para referenciar este projeto ou artigo em trabalhos acadêmicos, utilize o seguinte formato BibTeX:

```bibtex
@inproceedings{brasnam,
 author = {Jailson Januário and André Luiz Carvalho},
 title = { Detecção de Viés Ideológico em Artigos de Notícias Utilizando Aprendizagem Métrica Profunda e Representações Contextuais},
 booktitle = {Anais do XV Brazilian Workshop on Social Network Analysis and Mining},
 location = {Gramado/RS},
 year = {2026},
 keywords = {},
 issn = {2595-6094},
 pages = {109--121},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/brasnam.2026.21950},
 url = {[https://sol.sbc.org.br/index.php/brasnam/article/view/43242](https://sol.sbc.org.br/index.php/brasnam/article/view/43242)}
}

```

## 🚀 Estrutura do Projeto

* `src/executors/`: Scripts principais para treinamento e inferência.
* `src/topics/`: Scrips de treinamento e geração dos tópicos dos artigos de texto.
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
chmod +x execute_project.sh

```

Execute com os parâmetros padrão:

```bash
./execute_project.sh

```

Ou passe caminhos personalizados via linha de comando:

```bash
./execute_project.sh ./src/executors/meu_script.py ./src/parameters/config.json

```

## 🧠 Modelos Utilizados

Transformers:

* **DistilBERT**
* **DistilRoberta**

Modelo de Tópicos:

* **LDA**

Funções métricas:

* **Contrastive Loss**
* **Triple Loss**

Modelos de Classificação

* **KNN**
* **Kmeans**
* **MLP**

---