import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from src_graph_rag.llms.llms import LLMShortLearning

# Configurações de Ambiente (Substitua pelos seus dados)
os.environ["NEO4J_URI"] = "https://9be0da18.databases.neo4j.io/db/9be0da18/query/v2"
os.environ["NEO4J_USERNAME"] = "9be0da18"
os.environ["NEO4J_PASSWORD"] = "1aqjgA72jbFc6Re0dYyc9c5xwykED4Yt_dY4JTPJYtQ"
#os.environ["OPENAI_API_KEY"] = "sk-..."

def get_graph():
    return Neo4jGraph()

def get_llm():
    #model, _ = LLMShortLearning().get_model_tokenizer()
    model = ChatOllama(
        model="llama3.3",
        temperature=0.7
    )
    return model