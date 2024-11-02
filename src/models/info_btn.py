"""
This file is temporarily not useful.
"""
import torch
import torch.nn as nn
from src.models.gnn import SAGE
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
        if model_type == 'graphsage':
            self.backbone = SAGE(
                emb_size,
                config['topic_pe'],
                config['num_layers'],
                config['aggr']
            )
        elif model_type == "PNA":
            pass
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

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def forward(self, batch, batch_q_embds):
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
        h_id_tensor, t_id_tensor = batch.edge_index
        # h_id_tensor, t_id_tensor = edge_index
        # add reverse for performance gain
        edge_index_reverse = batch.edge_index.flip(0)
        edge_index_ = torch.cat([batch.edge_index, edge_index_reverse], dim=1)

        edge_attr_reverse = self.proj_reverse(batch.edge_attr)
        edge_attr_ = torch.cat([batch.edge_attr, edge_attr_reverse], dim=0)

        h_e_list = self.backbone(
            edge_index_,
            # topic_entity_one_hot,
            batch.x,
            edge_attr_,
        )
        h_e = torch.cat(h_e_list, dim=1)  # updated node embd

        # triplet embedding. Note that relation embeddings not updated.

        h_triple = torch.cat([
            batch_q_embds,
            h_e[h_id_tensor],
            batch.edge_attr,
            h_e[t_id_tensor]
        ], dim=1)
        # attention logits for each triplet.
        attn_logtis = self.pred(h_triple).squeeze()
        return attn_logtis, self.sampling(attn_logtis)

    def sampling(self, att_log_logit, temp=1, training=True):
        """
        strategy = "idp-bern" or "topk"
        K only applies when `strategy` set to "topk"
        """


        # training -- introduce stochastic
        if self.strategy == "idp-bern":
            # TODO: add straight-through gumbel.
            if not training:
                return att_log_logit.sigmoid()
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            return ((att_log_logit + random_noise) / temp).sigmoid()
        elif self.strategy == "topk":
            if not training:
                _, topk_indices = att_log_logit.topk(self.filter_num_or_ratio, dim=0, largest=True, sorted=False)
                return torch.zeros_like(att_log_logit, memory_format=torch.legacy_contiguous_format).scatter_(0, topk_indices, 1.0)
            # TODO: Note the `dim`. Consider batch_size.
            return gumbel_topk(att_log_logit, K=self.filter_num_or_ratio, hard=True, dim=0)
        else:
            raise NotImplementedError
