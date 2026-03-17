from src_graph_rag.config.config import get_graph, get_llm
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

import os
import pandas as pd
from typing import List, Dict, Tuple

class DocumentIngestionPipeline:
    def __init__(self, path: str, filename: str):
        self.llm = get_llm()
        self.graph = get_graph()
        self.path = path
        self.filename = filename

    def read_file(self, filename: str) -> pd.DataFrame:
        full_path = os.path.join(self.path, filename)
        if filename.endswith('.csv'):
            return pd.read_csv(full_path, on_bad_lines='skip', encoding_errors='ignore')
        return pd.read_excel(full_path)
    
    def get_corpus_flipbias(self, df):
        pattern = r'(?i)(right|center|left)'
        map_bias = {'left': 0, 'center': 1, 'right': 2}

        df['labels'] = df['labels'].str.extract(pattern, expand=False).str.lower()
        df = df[['labels', 'text']]
        df['labels'] = df['labels'].map(map_bias)
        
        return df
    
    def get_dataset_custom(self, dict_rename: Dict = {}, corpus = 'abp') -> Tuple[pd.DataFrame, None]:
        df = self.read_file(self.filename)
        
        if dict_rename:
            df = df.rename(columns=dict_rename)

        if corpus == 'flip':
            df = self.get_corpus_flipbias(df=df)
            self.df_dataset = df
            return df, None
           
        self.df_dataset = df
        return df, None
    
    def generate_documents(self, dict_rename, corpus):
        df, _ = self.get_dataset_custom(dict_rename=dict_rename, corpus=corpus)
        knowledge_base = [
            Document(page_content=row['text'])
            for _, row in df.iterrows()
        ]
        return knowledge_base

    def ingest_documents(self, parameters):
        # We define the ontology focused on Topics and Subjects
        transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Topic", "PoliticalSpectrum", "Value"],
            allowed_relationships=["ASSOCIATED_WITH", "OPPOSES", "CHARACTERIZES"]
        )

        knowledge_base = self.generate_documents(
            dict_rename=parameters['dict_rename'],
            corpus=parameters['corpus']
        )

        print("Extracting entities and relationships (Training)...")
        graph_documents = transformer.convert_to_graph_documents(knowledge_base)
        
        self.graph.add_graph_documents(graph_documents, baseEntityLabel=True)
        print("✓ Knowledge graph successfully populated.")
