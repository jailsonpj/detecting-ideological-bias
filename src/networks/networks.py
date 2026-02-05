import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import os, sys
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
sys.path.insert(0, project_root)
class Classifier(nn.Module):
    def __init__(self, input_size, num_labels):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels),
        )
    
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x
    
class TransformerNet(nn.Module):
    def __init__(self, name_model):
        super(TransformerNet, self).__init__()
        self.name_model = name_model
        self.transformer = AutoModel.from_pretrained(name_model)
        hidden_size = self.transformer.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)

    def mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, ids, mask, token_type_ids=None):
        outputs = self.transformer(
            input_ids=ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # 1. Pooling
        sentence_embeddings = self.mean_pooling(last_hidden_state, mask)
        
        # 2. Layer Normalization
        sentence_embeddings = self.layer_norm(sentence_embeddings)
        
        # 3. Normalização L2
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
    
    @torch.no_grad()
    def get_embeddings(self, ids, mask, token_type_ids=None):
        """
        Método utilitário para inferência pura.
        """
        self.eval()
        return self.forward(ids, mask, token_type_ids)
    
class TopicsNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TopicsNet, self).__init__()
        self.linear_relu_stack = nn.Sequencial(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
    def get_embeddings(self, x):
        return self.forward(x)

class ModalText(nn.Module):
    def __init__(self, fusion_type: str, text_hidden_size: int, topics_hidden_size: int):
        super(ModalText, self).__init__()
        self.fusion_type = fusion_type
        self.combined_dim = text_hidden_size + topics_hidden_size
        if self.fusion_type == "late_blending":
            self.fc1 = nn.Sequential(
                nn.Linear(self.combined_dim, text_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1) # Boa prática para evitar overfitting na fusão
            )

    def early_fuse(self, x1: torch.Tensor, x2: torch.Tensor, x3=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x1: torch.Tensor
            Tensor de features de texto para concatenação
        x2: torch.Tensor
            Tensor de features de tópicos para concatenação
        x3: None ou torch.Tensor
            Tensor com features para concatenação

        Returns
        ----------
        torch.Tensor
            Vetor com as representações concatenadas
        """
        if x3 == None:
            return torch.cat((x1, x2), dim=1)
        return torch.cat((x1, x2, x3), dim=1)
    
    def late_blending_fuse(self, x1: torch.Tensor, x2: torch.Tensor, x3=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x1: torch.Tensor
            Tensor de features de texto para concatenação
        x2: torch.Tensor
            Tensor de features de tópicos para concatenação
        x3: None ou torch.Tensor
            Tensor com features para concatenação

        Returns
        ----------
        torch.Tensor
            Vetor com as representações concatenadas
        """
        fused = torch.cat((x1, x2), dim=1) if x3 is None else torch.cat((x1, x2, x3), dim=1)
        return self.fc1(fused)
    
    def fuse_features(self, x1: torch.Tensor, x2: torch.Tensor, x3=None) -> torch.Tensor:
        if self.fusion_type == "early":
            return self.early_fuse(x1, x2, x3)
        return self.late_blending_fuse(x1, x2, x3)
       
class TextTopicsNet(ModalText):
    def __init__(self, dict_param: dict, name_model: str, fusion_type: str):
        text_dim = dict_param["text"]["hidden_size"]
        topics_in = dict_param["topics"]["input_dim"]
        topics_out = dict_param["topics"]["output_dim"]
        
        super(TextTopicsNet, self).__init__(fusion_type, text_dim, topics_out)
        
        self.text_net = TransformerNet(name_model)
        self.topics_net = TopicsNet(topics_in, topics_out)

    def forward(self, ids, mask, topics):
        x1 = self.text_net(ids, mask)
        x2 = self.topics_net(topics)
        output = self.fuse_features(x1, x2)
        return output
        
    def get_embedding(self, ids, mask, topics):
        return self.forward(ids, mask, topics)
    

"""params = {
    "text": {"hidden_size": 768},
    "topics": {
        "input_dim": 20,  # O número de tópicos do seu LDA
        "output_dim": 32  # Para quanto você quer projetar os tópicos antes da fusão
    }
}

model = TextTopicsNet(params, "bert-base-uncased", "late_blending")"""