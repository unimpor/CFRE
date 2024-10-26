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

        if config['learn_non_text']:
            self.non_text_entity_emb = nn.Embedding(1, emb_size)
        else:
            self.non_text_entity_emb = None
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

    def forward(self, batch):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, _ = batch

        device = entity_embs.device

        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)

        h_q = q_emb

        if self.non_text_entity_emb is None:
            h_e = torch.cat([
                entity_embs,
                torch.zeros(num_non_text_entities, entity_embs.size(1)).to(device)
            ], dim=0)
        else:
            h_e = torch.cat([
                entity_embs,
                self.non_text_entity_emb(torch.LongTensor([0]).to(device)).expand(
                    num_non_text_entities, -1)
            ], dim=0)
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]
        # add reverse for performance gain
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)

        # TODO: adjust put sth from self.gnn.forward to here !
        h_r_reverse = self.proj_reverse(h_r)
        h_r = torch.cat([h_r, h_r_reverse], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)

        # TODO: simplify its input
        h_e_list = self.backbone(
            edge_index,
            topic_entity_one_hot,
            h_e,
            h_r,
        )
        h_e = torch.cat(h_e_list, dim=1)

        # triplet embedding. Note that relation embeddings not updated.
        h_triple = torch.cat([
            q_emb.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        # attention logits for each triplet.
        attn_logtis = self.pred(h_triple)
        return self.sampling(attn_logtis)

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
