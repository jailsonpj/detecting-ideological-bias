from config import get_graph, get_llm
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate

class ListaTopicos(BaseModel):
    topicos: List[str] = Field(description="Lista de tópicos políticos ou econômicos mencionados")

def extrair_topicos(texto, llm):
    """Identifica os assuntos principais da notícia."""
    extrator = llm.with_structured_output(ListaTopicos)
    prompt = ChatPromptTemplate.from_template(
        "Extraia os 3 tópicos políticos/econômicos fundamentais deste texto: {texto}"
    )
    chain = prompt | extrator
    return chain.invoke({"texto": texto}).topicos

def classificar_noticia(texto_noticia):
    graph = get_graph()
    llm = get_llm()

    # 1. Extração de Tópicos
    topicos_encontrados = extrair_topicos(texto_noticia, llm)
    print(f"Tópicos detectados: {topicos_encontrados}")

    # 2. Query Cypher para buscar o contexto ideológico no Grafo
    cypher_query = """
    UNWIND $topicos AS nome_topico
    MATCH (t:Tópico) 
    WHERE t.id CONTAINS nome_topico OR nome_topico CONTAINS t.id
    MATCH (t)-[:ESTÁ_ASSOCIADO_A|CARACTERIZA|OPÕE_SE_A*1..2]-(espectro:EspectroPolítico)
    RETURN t.id AS Topico, espectro.id AS Vies, count(*) AS Peso
    """
    
    contexto_grafo = graph.query(cypher_query, params={"topicos": topicos_encontrados})

    # 3. Prompt de decisão final
    prompt_final = ChatPromptTemplate.from_template("""
    Você é um analista político. Use as informações do Grafo de Conhecimento para classificar a notícia.
    
    NOTÍCIA: {noticia}
    
    CONEXÕES DO GRAFO: {contexto}
    
    Com base apenas nos tópicos e suas conexões conhecidas, classifique em DIREITA, CENTRO ou ESQUERDA.
    Justifique sua resposta mencionando os tópicos encontrados.
    """)
    
    chain_final = prompt_final | llm
    return chain_final.invoke({
        "noticia": texto_noticia, 
        "contexto": contexto_grafo
    }).content

if __name__ == "__main__":
    # Exemplo de Teste
    noticia_teste = "O governo defendeu hoje a reforma agrária como prioridade para este semestre."
    
    print("\nIniciando Classificação GraphRAG...")
    resultado = classificar_noticia(noticia_teste)
    print("\n=== RESULTADO ===")
    print(resultado)