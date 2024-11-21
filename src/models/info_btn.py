"""
This file is temporarily not useful.
"""
import math
import torch
import torch.nn as nn
from src.models.gnn import SAGE, PNA
from src.utils import gumbel_topk


class FineGrainedRetriever(nn.Module):
    def __init__(self,
                 config,
                 algo_config,
                 **kwargs):
        super().__init__()

        self.strategy = algo_config["filtering"]  # strategy of filtering irrelevant info
        self.filter_num_or_ratio = algo_config["filtering_num_or_ratio"]
        self.training = True
        self.add_gumbel = algo_config["gumbel"]
        self.current_epoch = None
        self.tau = float(algo_config["tau"])
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
                config['num_layers'],
                config['aggr']
            )
        elif model_type == "PNA":
            self.backbone = PNA(
                emb_size,
                config['num_layers'],
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

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def set_eval(self):
        self.training = False
    
    def set_train(self):
        self.training = True
    
    def forward(self, batch, triplet_batch_idx, batch_q_embds, epoch):
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
        self.current_epoch = epoch
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
        prob_batch, mask_batch, sorted_idx_batch, logits_batch = [], [], [], []
        # 0/1 attention for each triplet. Note that topk strategy should be done per sample, rather than batch.
        if self.strategy == "idp-bern":
            # attns = self.sampling(attn_logtis)
            # deprecated feature
            pass
        elif self.strategy == "topk":
            
            for i in range(batch.num_graphs):
                attn_logit = attn_logtis[triplet_batch_idx == i]
                logits_batch.append(attn_logit)
                attn, sorted_idx = self.sampling(attn_logit)  # get each sample's gumbel-perturbed attention and 1's index
                mask_batch.append(attn)
                sorted_idx_batch.append(sorted_idx)
                prob_batch.append((attn_logit / self.tau).softmax(dim=0)[sorted_idx])
            # attns = torch.concat(attns)
        else:
            raise NotImplementedError
        # return attn_logtis, attns_batch, sorted_idx_batch
        return prob_batch, mask_batch, sorted_idx_batch, logits_batch

    def get_r(self, decay_interval=3, decay_r=0.1, init_r=0.9, final_r=0.3):
        r = init_r - self.current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
    
    def sampling(self, att_log_logit, temp=1):
        """
        strategy = "idp-bern" or "topk"
        K only applies when `strategy` set to "topk"
        """

        # training -- introduce stochastic
        if self.strategy == "idp-bern":
            # TODO: add straight-through gumbel.
            if not self.training:
                return att_log_logit.sigmoid()
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            return ((att_log_logit + random_noise) / temp).sigmoid()
        elif self.strategy == "topk":
            if not self.training:
                # in val or test process.
                if self.filter_num_or_ratio is not None:
                    K = math.ceil(len(att_log_logit) * self.filter_num_or_ratio)
                else: 
                    K = math.ceil(len(att_log_logit) * self.get_r())
                # K = self.filter_num_or_ratio if type(self.filter_num_or_ratio) is int \
                # else math.ceil(len(att_log_logit) * self.filter_num_or_ratio)
                _, topk_indices = att_log_logit.topk(K, dim=0, largest=True, sorted=True)
                y_hard = torch.zeros_like(att_log_logit, memory_format=torch.legacy_contiguous_format).scatter_(0, topk_indices, 1.0)
                return y_hard, topk_indices
            # TODO: Note the `dim`. Consider batch_size.
            else:
                K = math.ceil(len(att_log_logit) * self.get_r())
                return gumbel_topk(att_log_logit, K=K, tau=self.tau, mode="hard", dim=0, add_grumbel=self.add_gumbel)
        else:
            raise NotImplementedError
