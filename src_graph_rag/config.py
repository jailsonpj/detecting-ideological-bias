import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

# Configurações de Ambiente (Substitua pelos seus dados)
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "sua_senha"
os.environ["OPENAI_API_KEY"] = "sk-..."

def get_graph():
    return Neo4jGraph()

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)