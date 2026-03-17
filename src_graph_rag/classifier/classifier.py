from src_graph_rag.config.config import get_graph, get_llm
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate

class TopicList(BaseModel):
    topics: List[str] = Field(description="List of mentioned political or economic topics")

def extract_topics(text, llm):
    """Identifies the main subjects of the news article."""
    extractor = llm.with_structured_output(TopicList)
    prompt = ChatPromptTemplate.from_template(
        "Extract the 3 fundamental political/economic topics from this text: {text}"
    )
    chain = prompt | extractor
    return chain.invoke({"text": text}).topics

def classify_news(news_text):
    graph = get_graph()
    llm = get_llm()

    # 1. Topic Extraction
    detected_topics = extract_topics(news_text, llm)
    print(f"Detected Topics: {detected_topics}")

    # 2. Cypher Query to fetch ideological context from the Graph
    # Note: Kept the relationship types matching the previous translated ingestion pipeline
    cypher_query = """
    UNWIND $topics AS topic_name
    MATCH (t:Topic) 
    WHERE t.id CONTAINS topic_name OR topic_name CONTAINS t.id
    MATCH (t)-[:ASSOCIATED_WITH|CHARACTERIZES|OPPOSES*1..2]-(spectrum:PoliticalSpectrum)
    RETURN t.id AS Topic, spectrum.id AS Bias, count(*) AS Weight
    """
    
    graph_context = graph.query(cypher_query, params={"topics": detected_topics})

    # 3. Final decision prompt
    final_prompt = ChatPromptTemplate.from_template("""
    You are a political analyst. Use the information from the Knowledge Graph to classify the news.
    
    NEWS: {news}
    
    GRAPH CONNECTIONS: {context}
    
    Based only on the topics and their known connections, classify as RIGHT, CENTER, or LEFT.
    Justify your answer by mentioning the topics found.
    """)
    
    final_chain = final_prompt | llm
    return final_chain.invoke({
        "news": news_text, 
        "context": graph_context
    }).content