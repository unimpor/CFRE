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
        emb_size,
        topic_pe,
        kwargs
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**kwargs)
        
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (kwargs['num_rounds'] + kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot
    ):
        device = entity_embs.device
        
        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        
        return self.pred(h_triple)