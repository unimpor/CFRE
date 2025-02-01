import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from src.utils import RewardMetrics, gumbel_topk, print_log
import copy
import random
import math
from collections import Counter
from copy import deepcopy


class RLRE(nn.Module):
    """
    RL driven Retriver, better aligned with LLM reasoning capability.
    """
    def __init__(self, retriever, llm_model, config, **kwargs):
        super().__init__()
        self.retriever = retriever
        self.llms = llm_model
        self.coeff1 = float(config['coeff1'])  # trade-off coeff between RL and regularization
        self.coeff2 = float(config['coeff2'])
        self.algo = config["algo"]
        self.metrics_name = config["reward_metrics"]
        self.reward_metrics = RewardMetrics(self.metrics_name)
        self.perturb_per_sample = config["perturb_per_sample"]  # Use 1. Not used in this method.
        self.regularize = config["regularize"]
        self.K = config["ret_num"]
        self.evaluation = {}
        print(self.K)
        self.set_moving_baseline = config["set_moving_baseline"]
        self.eps = 1e-6  # for numerical stability
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.strategy = config["filtering"]
        self.filter_num_or_ratio = config["filtering_num_or_ratio"]
        self.gumbel_strength = float(config["gumbel_strength"])
        self.tau = float(config["tau"])
        self.baseline_order_invariant = config["baseline_order_invariant"]
        self.update_num = 0
        self.noise_generator = torch.Generator(device=self.device)
        
    @property
    def device(self):
        return self.retriever.device

    def cal_reward_single(self, t, idx, q, a):
        masked_triplets_batch, _ = self.mask_triplet([t], [idx])
        g = self.llms([q], masked_triplets_batch)
        print_log(g[0])
        return self.reward_metrics.calc_r(g[0],a,q), g[0]

    def cal_reward(self, g_batch, q_batch, a_batch, id_batch, epoch, training):
        """
        Given query and retrieved triplets, return the reward rerlative to the ``baseline''.
        """
        reward_batch = {
            "F1": [],
            "precision": [],
            "recall": [],
        }
        for g,q,a,d in zip(g_batch, q_batch, a_batch, id_batch):
            (f1, prec, recall), _ = self.reward_metrics.calc_r(g,a,q)
            print(f1, prec, recall)
            self.evaluation[d]["F1"] = f1
            self.evaluation[d]["precision"] = prec
            self.evaluation[d]["recall"] = recall
            self.evaluation[d]["gen"] = g

            reward_batch["F1"].append(f1)
            reward_batch["precision"].append(prec)
            reward_batch["recall"].append(recall)
        return reward_batch   
    
    def cal_loss_down_sampling(self, attn_logtis, relevant_idx, shortest_path_idx, N=2):
        # delete unsure negative;
        # sample N * |\wt| negative samples, if N is none, sample all neg.
        logit_list, target_list = [], []
        for logit, pos, boundry in zip(attn_logtis, relevant_idx, shortest_path_idx):
            if len(pos) == 0:
                print("Not identifying positive.")
                continue
            target = torch.zeros_like(logit, device=logit.device)
            target[pos] = 1.
            pos = torch.tensor(pos, device=logit.device)
            neg_candidates = torch.ones_like(logit, dtype=torch.bool, device=logit.device)
            neg_candidates[pos] = 0
            neg_candidates[boundry] = 0
            
            neg = torch.nonzero(neg_candidates).squeeze()
            if len(neg) > N * len(pos):
                neg = neg[torch.randperm(len(neg))[:N * len(pos)]]
            else:
                neg = neg[torch.randperm(len(neg))]
            
            logit_list.append(logit[torch.cat([pos, neg])])
            target_list.append(target[torch.cat([pos, neg])])

        if not logit_list or not target_list:
            return torch.tensor(0.0, device=attn_logtis.device), {"warmup": 0.0}
        
        logit_list = torch.cat(logit_list)
        target_list = torch.cat(target_list)
        
        warmup_loss = F.binary_cross_entropy_with_logits(logit_list, target_list)
        
        return warmup_loss, {"warmup": warmup_loss.item()}

    def cal_loss(self, attn_logtis, relevant_idx):
        # batch version
        # target = torch.zeros_like(attn_logtis, device=attn_logtis.device)
        # target[relevant_idx] = 1.
        # warmup_loss = F.binary_cross_entropy_with_logits(attn_logtis, target)
        logit_list, target_list = [], []
        for logit, pos in zip(attn_logtis, relevant_idx):
            if len(pos) == 0:
                print("Not identifying positive.")
                continue
            target = torch.zeros_like(logit, device=logit.device)
            target[pos] = 1.
            
            logit_list.append(logit)
            target_list.append(target)
            # weighted version
            # weight = torch.ones_like(logit, device=logit.device)
            # weight[pos] = 5
            # weight = weight / weight.sum()
            # warmup_loss.append(F.binary_cross_entropy_with_logits(logit, target).unsqueeze(0))
        logit_list = torch.cat(logit_list)
        target_list = torch.cat(target_list)

        warmup_loss = F.binary_cross_entropy_with_logits(logit_list, target_list)
        return warmup_loss, {"warmup": warmup_loss.item()}

    def inference(self, batch): 
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch, shortest_path_idx_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"], batch["shortest_path_idx"]

        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)
        select_batch = []
        
        for i, dat_id in enumerate(id_batch):
            attn_logit = attn_logits_batch[triplet_batch_idx == i]
            _, select_idx = self.sampling_v3(attn_logit, K=self.K)
            
            self.evaluation[dat_id] = {"select": select_idx}
            select_batch.append(select_idx)

        masked_triplets_batch, _ = self.mask_triplet(triplet_batch, select_batch)
        generation_batch = self.llms(question_batch, answer_batch, masked_triplets_batch)
        reward_loggings = self.cal_reward(generation_batch, question_batch, answer_batch, id_batch, 0, training=False)
        return 0, reward_loggings

    @torch.no_grad()
    def oracle_detection(self, batch, epoch, update_reference=False):
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch, paths_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"], batch["relevant_paths"]

        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        for i, (d, q, a, p, allt) in enumerate(zip(id_batch, question_batch, answer_batch, paths_batch, triplet_batch)):
            # g = self.llms([q], [a], [p])
            selected = self.llms.oracle_detection(q, a, p, allt)
            self.evaluation[d] = selected

    def forward_pass(self, batch, epoch, training=True): 
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch, shortest_path_idx_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"], batch["shortest_path_idx"]

        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)
        logits_batch = []
        reward_loggings = {"recall": [], "step": []}
        
        for i, dat_id in enumerate(id_batch):
            attn_logit = attn_logits_batch[triplet_batch_idx == i]
            logits_batch.append(attn_logit)
            if not training:
                _, select_idx = self.sampling_v3(attn_logit, K=self.K)

                all_triplets, ans_list = triplet_batch[i], answer_batch[i]
                selected_triplets = [all_triplets[idx] for idx in select_idx]
                selected_entities = [t[0] for t in selected_triplets] + [t[-1] for t in selected_triplets]
                
                recall = [ans for ans in ans_list if ans in selected_entities]
                step = [t for t in selected_triplets if (t[0] in ans_list or t[-1] in ans_list)]
                
                reward_loggings["recall"].append(float(len(recall) / len(ans_list)))
                reward_loggings["step"].append(float(len(step) / len(selected_triplets)))
                
        if not training:
            reward_loggings = {k:np.mean(v).item() for k,v in reward_loggings.items()}
            return 0, reward_loggings
        
        # loss, reward_loggings = self.cal_loss_down_sampling(logits_batch, relevant_idx_batch, shortest_path_idx_batch)
        loss, reward_loggings = self.cal_loss(logits_batch, relevant_idx_batch)
        return loss, reward_loggings

    def sampling_v3(self, tensor, training=False, K=6):
        sorted_indices = torch.argsort(tensor, descending=True)
        positive_indices = sorted_indices[:K]
        return None, positive_indices.tolist()

    def sampling_v2(self, tensor, training=False, K=6):
        indices = torch.arange(len(tensor), device=tensor.device)
    
        positive_indices = indices[tensor > 0]
        positive_values = tensor[tensor > 0]

        if len(positive_indices) > K:
            sorted_indices = torch.argsort(positive_values, descending=True)
            positive_indices = positive_indices[sorted_indices]
        else:
            sorted_indices = torch.argsort(tensor, descending=True)
            positive_indices = sorted_indices[:K]
        return None, positive_indices.tolist()

    def mask_triplet(self, triplets_batch, select_idx_batch, to_str=True):
        """
        `strategy` can be set as "drop" or "mask", "drop" as default.
        Mask tokenized triplets using attn
        # Note this method is achieved based on sample-wise not batch-wise.
        # we prefer "drop" strategy with ordered K-sampling w.o. replacement.
        """
        # Example: triplets converted into strings

        # Load the tokenizer
        def triplet_to_str(triplet):
            return f"({triplet[0]},{triplet[1]},{triplet[2]})"
        # Tokenize each triplet string
        masked_attns_batch, masked_triplets_batch = [], []
        for triplets, keep_idx in zip(triplets_batch, select_idx_batch):
        # In this strategy, just drop the unselected triplets.
            # keep_idx = [idx for idx, score in enumerate(attns) if score.item() == 1]
            select_triplets = [triplet_to_str(triplets[idx]) for idx in keep_idx]
            masked_triplets_batch.append(select_triplets)
            # assert attns.shape == triplets_token_ids.shape
            # masked_token_ids = attns * triplets_token_ids
        return masked_triplets_batch, masked_attns_batch
        # return masked_token_ids

    def case_study(self, id_batch, question_batch, answer_batch, triplet_batch, select_idx, gt_idx):
        
        print_log(id_batch[0])
        print_log(question_batch[0])
        print_log(answer_batch[0])
        
        print_log("select cases:")
        print_log(select_idx[0])
        masked_triplets_batch, _ = self.mask_triplet(triplet_batch, select_idx)
        generation_batch = self.llms(question_batch, masked_triplets_batch)
        print_log(generation_batch[0])

        print_log("weak signal cases:")
        print_log(gt_idx[0])
        masked_triplets_batch, _ = self.mask_triplet(triplet_batch, gt_idx)
        generation_batch = self.llms(question_batch, masked_triplets_batch)
        print_log(generation_batch[0])

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result