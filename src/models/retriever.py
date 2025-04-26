"""
https://github.com/Graph-COM/SubgraphRAG/blob/main/retrieve/src/model/retriever.py
"""
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from src.models.dde import DDE
from src.models.gnn import PNA

class Retriever(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        emb_size = config['hidden_size']
        self.model_type = config['model_type']
        model_kwargs = config[self.model_type]

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = config["topic_pe"]
        pred_in_size = 4 * emb_size
        if self.model_type == "DDE":
            self.DDE = DDE(**model_kwargs)
            if self.topic_pe:
                pred_in_size += 2 * 2
            pred_in_size += 2 * 2 * (model_kwargs['num_rounds'] + model_kwargs['num_reverse_rounds'])
        elif self.model_type == "PNA":
            model_kwargs['deg'] = kwargs.get("deg")
            self.PNA = PNA(**model_kwargs)
            pred_in_size = 2 * emb_size + 2
        else:
            raise NotImplementedError
        
        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, config['output_size'])
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
        if self.model_type == 'DDE':
            if self.topic_pe:
                h_e_list.append(topic_entity_one_hot)
            ent_rprs = self.DDE(topic_entity_one_hot, edge_index, edge_index.flip(0))
        elif self.model_type == 'PNA':
            ent_rprs = self.PNA(x, edge_attr, edge_index)
        h_e_list.extend(ent_rprs)
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
    
    def trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

class BaselineRetriever(Retriever):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
    
    def forward(self, graph, q_emb,):
        device = self.device
        x, edge_index, edge_attr, topic_entity_one_hot = graph.x, graph.edge_index, graph.edge_attr, graph.topic_signal
        # assert edge_index.max() < x.size(0), f"edge_index max {edge_index.max()} >= x.size(0) {x.size(0)}"
        # assert edge_index.min() >= 0, f"edge_index has negative value: {edge_index.min()}"
        # if edge_index.numel() == 0:
        #     print("Warning: empty edge_index")

        x[(x==0).all(dim=1)] = self.non_text_entity_emb(torch.LongTensor([0]).to(device))

        h_e_list = [topic_entity_one_hot]

        if self.model_type == 'PNA':
            ent_rprs = self.PNA(x, edge_attr, edge_index)
        h_e_list.extend(ent_rprs)
        h_e = torch.cat(h_e_list, dim=1)

        h_triple = torch.cat([
            q_emb,
            h_e,
        ], dim=1)
        # get the score of each entity. triple score = head score + tail score.
        return self.pred(h_triple).squeeze()
    