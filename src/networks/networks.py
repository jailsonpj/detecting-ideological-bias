import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

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
    def __init__(self):
        super(TopicsNet, self).__init__()
        self.linear_relu_stack = nn.Linear(7, 10)

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
    
    def get_embeddings(self, x):
        return self.forward(x)

class MultimodalNN(nn.Module):
    def __init__(self, fusion_type: str):
        super(MultimodalNN, self).__init__()
        self.fusion_type = fusion_type

        if self.fusion_type == "late_blending":
            self.fc1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(768 + 10, 768)
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
        if x3 == None:
            return self.fc1(torch.cat((x1, x2), 1))
        return self.fc1(torch.cat((x1, x2, x3), 1))
    
    def fuse_features(self, x1: torch.Tensor, x2: torch.Tensor, x3=None) -> torch.Tensor:
        if self.fusion_type == "early":
            return self.early_fuse(x1, x2, x3)
        return self.late_blending_fuse(x1, x2, x3)
       
class TextTopicsNet(MultimodalNN):
    def __init__(self, dict_param: dict, name_model: str, fusion_type: str):
        super(TextTopicsNet, self).__init__(fusion_type)
        self.text_params = dict_param["text"]
        self.topics_params = dict_param["topics"]
        self.name_model = name_model
        self.text_net = TransformerNet(self.name_model)
        self.topics_net = TopicsNet()
        self.fusion_type = fusion_type

    def forward(self, ids, mask, topics):
        x1 = self.text_net(ids, mask)
        x2 = self.topics_net(topics)
        output = self.fuse_features(x1, x2)
        return output
        
    def get_embedding(self, ids, mask, topics):
        return self.forward(ids, mask, topics)