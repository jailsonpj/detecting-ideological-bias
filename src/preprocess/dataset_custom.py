import pandas as pd
import torch
from torch.utils.data import Dataset
import os, sys
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

class ClfDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # Long é padrão para classificação (CrossEntropy)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
    
import os
from typing import List, Dict, Tuple

class NewsPaperData:
    def __init__(self, path: str, filename: str):
        self.path = path
        self.filename = filename
        self.df_dataset = pd.DataFrame()

    def read_file(self, filename: str) -> pd.DataFrame:
        full_path = os.path.join(self.path, filename)
        if filename.endswith('.csv'):
            return pd.read_csv(full_path)
        return pd.read_excel(full_path)

    def union_title_corpus(self, df: pd.DataFrame, col_title: str, col_corpus: str) -> pd.DataFrame:
        df["text"] = df[col_title].astype(str) + " " + df[col_corpus].astype(str)
        return df
    
    def get_dataset_custom(self, dict_rename: Dict = {}) -> Tuple[pd.DataFrame, None]:
        df = self.read_file(self.filename)
        
        if dict_rename:
            df = df.rename(columns=dict_rename)
            
        self.df_dataset = df
        return df[:10], None
    
class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df['text'].astype(str).tolist()
        self.targets = df['labels'].tolist()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = " ".join(self.texts[index].split())
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            return_token_type_ids=True
        )

        data_input = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long)
        }
        return data_input, torch.tensor(self.targets[index], dtype=torch.long)