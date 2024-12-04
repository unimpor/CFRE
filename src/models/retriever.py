"""
https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/src/model/retriever.py
"""
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from src.models.dde import DDE

class Retriever(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        emb_size = config['hidden_size']
        model_type = config['model_type']
        model_kwargs = config[model_type]

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = config["topic_pe"]
        self.dde = DDE(**model_kwargs)
        
        pred_in_size = 4 * emb_size
        if self.topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (model_kwargs['num_rounds'] + model_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def forward(
        self,
        graph,
        q_emb,
    ):
        device = self.device
        x, edge_index, edge_attr, topic_entity_one_hot = graph.x, graph.edge_index, graph.edge_attr, graph.topic_signal
        # replace all-zero row with nn.embeddings
        x[(x==0).all(dim=1)] = self.non_text_entity_emb(torch.LongTensor([0]).to(device))
        
        h_e_list = [x]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)
        
        dde_list = self.dde(topic_entity_one_hot, edge_index, edge_index.flip(0))
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        # Potentially memory-wise problematic
        
        h_id_tensor, t_id_tensor = edge_index
        h_triple = torch.cat([
            q_emb,
            h_e[h_id_tensor],
            edge_attr,
            h_e[t_id_tensor]
        ], dim=1)
        
        return self.pred(h_triple).squeeze()