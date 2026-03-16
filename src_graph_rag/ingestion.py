from config import get_graph, get_llm
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

def pipeline_treino():
    graph = get_graph()
    llm = get_llm()

    # Definimos a ontologia focada em Tópicos e Assuntos
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Tópico", "EspectroPolítico", "Valor"],
        allowed_relationships=["ESTÁ_ASSOCIADO_A", "OPÕE_SE_A", "CARACTERIZA"]
    )

    # Dados de treino: Conhecimento teórico sobre política
    conhecimento_base = [
        Document(page_content="Privatização, Teto de Gastos e Livre Mercado são Tópicos associados à Direita."),
        Document(page_content="Reforma Agrária, Estatização e Justiça Social são Tópicos associados à Esquerda."),
        Document(page_content="Equilíbrio Fiscal e Programas Sociais Focados são Tópicos associados ao Centro."),
        Document(page_content="O Tópico 'Privatização' opõe-se ao Tópico 'Estatização'."),
        Document(page_content="Liberdade Individual e Propriedade Privada caracterizam a Direita."),
        Document(page_content="Coletivismo e Intervenção Estatal caracterizam a Esquerda.")
    ]

    print("Extraindo entidades e relações (Treino)...")
    graph_documents = transformer.convert_to_graph_documents(conhecimento_base)
    
    graph.add_graph_documents(graph_documents, baseEntityLabel=True)
    print("✓ Grafo de conhecimento populado com sucesso.")

if __name__ == "__main__":
    pipeline_treino()