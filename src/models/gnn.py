"""
GNN Backbone. Now we only implement GraphSAGE
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SAGEConv(MessagePassing):
    def __init__(self,
                 emb_size,
                 aggr):
        super().__init__(aggr=aggr)

        self.mlp = nn.Sequential(
            nn.Linear(3 * emb_size, emb_size),
            nn.ReLU()
        )

    def forward(self, edge_index, h_e, h_r):
        h_e_nbr = self.propagate(edge_index, x=h_e, h_r=h_r)
        return self.mlp(torch.cat([
            h_e, h_e_nbr
        ], dim=1))

    def message(self, x_i, x_j, h_r):  # i: tgt, j:src
        msg_list = [x_j, h_r]
        msg = torch.cat(msg_list, dim=1)
        return msg


class SAGE(nn.Module):
    def __init__(self,
                 emb_size,
                 topic_pe,
                 num_gnn_layers,
                 aggr):
        super().__init__()

        self.gnn_layer_list = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layer_list.append(SAGEConv(emb_size, aggr))
        self.topic_pe = topic_pe

        # self.proj_reverse = nn.Sequential(
        #     nn.Linear(emb_size, emb_size),
        #     nn.ReLU(),
        #     nn.Linear(emb_size, emb_size)
        # )

        self.out_size = emb_size
        if self.topic_pe:
            self.out_size += 2

    def forward(self,
                edge_index,
                topic_entity_one_hot,
                h_e,
                h_r,
                **kwargs):
        # TODO: werid...
        if self.topic_pe:
            h_e_list = [topic_entity_one_hot]
        else:
            h_e_list = []

        # num_edges = edge_index.shape[1]
        # h_q = h_q.expand(num_edges, -1)

        # h_r_reverse = self.proj_reverse(h_r)
        # h_r = torch.cat([h_r, h_r_reverse], dim=0)
        # edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)

        for gnn_layer in self.gnn_layer_list:
            h_e = gnn_layer(edge_index, h_e, h_r)
        h_e_list.append(h_e)

        return h_e_list

