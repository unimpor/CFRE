"""
This file is temporarily not useful.
"""
import torch
import torch.nn as nn
from gnn import SAGE, SAGEConv
from src.utils import gumbel_topk


class FineGrainedRetriever(nn.Module):
    def __init__(self,
                 config,
                 filtering_strategy,
                 filtering_num_or_ratio,
                 **kwargs):
        super().__init__()

        self.strategy = filtering_strategy  # strategy of filtering irrelevant info
        self.filter_num_or_ratio = filtering_num_or_ratio

        emb_size = config['hidden_size']
        model_type = config['model_type']

        # This is deprecated. We just use trivial non-text entity embedding.
        # if config['learn_non_text']:
        #     self.non_text_entity_emb = nn.Embedding(1, emb_size)
        # else:
        #     self.non_text_entity_emb = None
        # TODO: You may try different types of fg-retriever. But now only GraphSage implemented.
        if model_type == 'graphsage':
            self.backbone = SAGE(
                emb_size,
                config['topic_pe'],
                config['num_layers'],
                config['aggr']
            )
        else:
            raise NotImplementedError(f'GNN type {model_type} not implemented.')

        pred_in_size = 2 * emb_size + 2 * self.backbone.out_size

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
        self.proj_reverse = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

    def forward(self, entity_embd, edge_index, edge_attr, q_embd):
        # TODO: Currently deprecate two Functions, `non_text_entity_embd` and `topic_entity_onehot`
        # if self.non_text_entity_emb is None:
        #     h_e = torch.cat([
        #         entity_embs,
        #         torch.zeros(num_non_text_entities, entity_embs.size(1)).to(device)
        #     ], dim=0)
        # else:
        #     h_e = torch.cat([
        #         entity_embs,
        #         self.non_text_entity_emb(torch.LongTensor([0]).to(device)).expand(
        #             num_non_text_entities, -1)
        #     ], dim=0)

        h_id_tensor, t_id_tensor = edge_index
        # add reverse for performance gain
        edge_index_reverse = edge_index.flip(0)
        edge_index = torch.cat([edge_index, edge_index_reverse], dim=1)

        edge_attr_reverse = self.proj_reverse(edge_attr)
        h_r = torch.cat([edge_attr, edge_attr_reverse], dim=0)

        h_e_list = self.backbone(
            edge_index,
            # topic_entity_one_hot,
            entity_embd,
            h_r,
        )
        h_e = torch.cat(h_e_list, dim=1)  # updated node embd

        # triplet embedding. Note that relation embeddings not updated.
        h_triple = torch.cat([
            q_embd.expand(len(edge_attr), -1),
            h_e[h_id_tensor],
            edge_attr,
            h_e[t_id_tensor]
        ], dim=1)
        # attention logits for each triplet.
        attn_logtis = self.pred(h_triple)
        return attn_logtis, self.sampling(attn_logtis)

    def sampling(self, att_log_logit, temp=1, training=True):
        """
        strategy = "idp-bern" or "topk"
        K only applies when `strategy` set to "topk"
        """
        if not training:
            return att_log_logit.sigmoid()

        # training -- introduce stochastic
        if self.strategy == "idp-bern":
            # TODO: add straight-through gumbel.
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            return ((att_log_logit + random_noise) / temp).sigmoid()
        elif self.strategy == "topk":
            # TODO: Note the `dim`. Consider batch_size.
            return gumbel_topk(att_log_logit, K=self.filter_num_or_ratio, hard=True, dim=0)
        else:
            raise NotImplementedError
