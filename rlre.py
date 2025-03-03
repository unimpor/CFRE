import asyncio
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from src.utils import RewardMetrics, gumbel_topk, print_log, reorder
import copy
import random
import math
from collections import Counter
from copy import deepcopy
import networkx as nx


class RLRE(nn.Module):
    """
    RL driven Retriver, better aligned with LLM reasoning capability.
    """
    def __init__(self, retriever, llm_model, config, **kwargs):
        super().__init__()
        self.retriever = retriever
        self.llms = llm_model
        self.coeff1 = float(config['coeff1'])
        self.coeff2 = float(config['coeff2'])
        self.metrics_name = config["reward_metrics"]
        self.reward_metrics = RewardMetrics(self.metrics_name)
        self.K = config["ret_num"]
        self.evaluation = {}
        # hard negatives
        self.add_hard = kwargs.get("add_hard", True)
        self.weight = kwargs.get("weight", 1.0)
        # triplets arrangement
        self.reorder = kwargs.get("reorder", False)
        self.retrieval_clip = kwargs.get("clipping", False)
        self.budget = kwargs.get("budget", 100)
        self.level = kwargs.get("level", "triplet")
        print(self.budget, self.retrieval_clip, self.reorder, self.level)
        
    @property
    def device(self):
        return self.retriever.device

    # def cal_reward_single(self, t, idx, q, a):
    #     masked_triplets_batch, _ = self.mask_triplet([t], [idx])
    #     g = self.llms([q], masked_triplets_batch)
    #     print_log(g[0])
    #     return self.reward_metrics.calc_r(g[0],a,q), g[0]

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
            (f1, prec, recall), (matched, not_prec, missing) = self.reward_metrics.calc_r(g,a,q)
            self.evaluation[d].update({
                "F1": f1,
                "precision": prec,
                "recall": recall,
                "gen": g,
                "matched": matched,
                "not_prec": not_prec,
                "missing": missing
            })
            reward_batch["F1"].append(f1)
            reward_batch["precision"].append(prec)
            reward_batch["recall"].append(recall)
        return reward_batch   
    
    def cal_loss_multi_cat(self, attn_logits, scores):
        warmup_loss = F.cross_entropy(attn_logits, scores)
        return warmup_loss, {"warmup": warmup_loss.item()}

    def cal_loss_with_weights(self, attn_logtis, relevant_idx, hard_idx):
        logit_list, target_list, weight_list = [], [], []
        for logit, pos, hard in zip(attn_logtis, relevant_idx, hard_idx):
            if len(pos) == 0:
                continue
            target = torch.zeros_like(logit, device=logit.device)
            target[pos] = 1.

            weight = torch.ones_like(logit, device=logit.device)
            weight[hard] = self.weight
            
            logit_list.append(logit)
            target_list.append(target)
            weight_list.append(weight)
        
        logit_list = torch.cat(logit_list)
        target_list = torch.cat(target_list)
        weight_list = torch.cat(weight_list)
        if self.add_hard:
            warmup_loss = F.binary_cross_entropy_with_logits(logit_list, target_list, weight_list)
        else:
            warmup_loss = F.binary_cross_entropy_with_logits(logit_list, target_list)
        lengths = [len(sublist) for sublist in hard_idx]
        return warmup_loss, {"warmup": warmup_loss.item(), "hard_len": sum(lengths) / len(lengths)}

    def cal_loss(self, attn_logtis, relevant_idx, relevant_idx_in_path, epoch):
        # batch version
        # target = torch.zeros_like(attn_logtis, device=attn_logtis.device)
        # target[relevant_idx] = 1.
        # warmup_loss = F.binary_cross_entropy_with_logits(attn_logtis, target)
        logit_list, target_list, cons_loss_list = [], [], []

        for logit, pos, paths in zip(attn_logtis, relevant_idx, relevant_idx_in_path):
            if len(pos) == 0:
                continue
            target = torch.zeros_like(logit, device=logit.device)
            target[pos] = 1.
            
            logit_list.append(logit)
            target_list.append(target)
            for p in paths:
                if len(p) > 1:
                    assert all(x in pos for x in p)
                    # cons_loss_list.append((logit[p] - logit[p].mean().detach()) ** 2)
                    # closer to mean and larger than 0
                    # cons_loss_list.append((logit[p] - torch.maximum(logit[p].mean(), torch.tensor(0.0).to(logit.device)).detach()) ** 2)
                    
                    # # TODO var-like loss
                    vmax = torch.maximum(logit[p].max(), torch.tensor(0.0).to(logit.device))
                    # target = torch.where(
                    #     vmax - logit[p] <= self.coeff2,
                    #     vmax,
                    #     logit[p] + self.coeff2
                    # ).to(logit.device)
                    # cons = (logit[p] - target.detach()) ** 2
                    cons = (logit[p] - vmax.detach()) ** 2
                    cons_loss_list.append(cons)

                    # TODO series loss
                    # for i in range(len(p)):
                    #     for j in range(i + 1, len(p)):
                    #         logit_i = logit[p[i]]
                    #         logit_j = logit[p[j]]
                    #         cons_loss_list.append(((logit_i - logit_j) ** 2).unsqueeze(0))
                     
        logit_list = torch.cat(logit_list)
        target_list = torch.cat(target_list)
        cons_loss_list = torch.cat(cons_loss_list)
        # loss clamp
        # cons_loss_list = torch.clip(cons_loss_list, max=self.coeff2)

        warmup_loss = F.binary_cross_entropy_with_logits(logit_list, target_list)
        cons_loss = torch.mean(cons_loss_list)
        
        total_loss = warmup_loss + self.coeff1 * cons_loss if epoch > 2 else warmup_loss
        return total_loss, {"warmup": warmup_loss.item(), "consistency": cons_loss.item()}

    def preparing(self, batch):
        # batch size = 1
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch, shortest_path_idx_batch, hints_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"], batch["shortest_path_idx"], batch["q_entity"]
        dat_id, all_triplets, q_entities = id_batch[0], triplet_batch[0], hints_batch[0]
        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)
        select_idx = self.sampling_v3(attn_logits_batch, K=self.K)

        select_paths = None
        if self.reorder:
            select_idx, select_paths = self.reorganize(all_triplets, q_entities, select_idx, attn_logits_batch)
        select_triplets = [triplet_to_str(all_triplets[i]) for i in select_idx]
        
        self.evaluation[dat_id] = {"select": select_idx, 
                                   "select_paths": select_paths,
                                   "select_triplets": select_triplets}

    async def inference(self, batch, logging=None, retrieval=True): 
        answer_batch, question_batch, id_batch, hints_batch = batch["y"], batch["q"], batch["id"], batch["q_entity"]
        cache = self.evaluation[id_batch[0]]
        select_triplets, select_paths = cache["select_triplets"], cache["select_paths"]
        if self.level == "triplet":
            generation_batch = await self.llms.forward_pass(question_batch, hints_batch, [select_triplets])
        elif self.level == "path":
            generation_batch = await self.llms.forward_pass(question_batch, hints_batch, [select_paths])
        reward_loggings = self.cal_reward([generation_batch], question_batch, answer_batch, id_batch, 0, training=False)
        return reward_loggings

    @torch.no_grad()
    def oracle_detection(self, batch, logging):
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, question_batch, q_embd_batch, id_batch, paths_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["q"], batch["q_embd"], batch["id"], batch["relevant_paths"]

        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)
        
        selected_batch = self.llms.oracle_detection(id_batch, question_batch, answer_batch, paths_batch, logging=logging)

        for d, s in zip(id_batch, selected_batch):
            d = d.split("+")[0]
            if d in self.evaluation:
                self.evaluation[d].extend(s)
            else:
                self.evaluation[d] = s
            # self.evaluation[d] = {"selected": s}
            
    def forward_pass(self, batch, epoch, training=True): 
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch, shortest_path_idx_batch = \
            batch["graph"], batch["a_entity"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"], batch["shortest_path_idx"]
        hard_idx_batch, relevant_idx_in_path_batch = batch["hard_idx"], batch["relevant_idx_in_path"]

        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)  # [B, N]
        logits_batch, hard_idx_batch_ = [], []
        reward_loggings = {"recall": [], "step": [], "ranking": []}
        
        for i, dat_id in enumerate(id_batch):
            attn_logit = attn_logits_batch[triplet_batch_idx == i]
            logits_batch.append(attn_logit)
            select_idx = self.sampling_v3(attn_logit, K=self.K if not training else 200)

            hard_idx, pos_idx = hard_idx_batch[i], relevant_idx_batch[i]
            # TODO: add hard positive or not ?
            hard_idx = [i for i in hard_idx if i in select_idx] + [j for j in pos_idx if j not in select_idx[:50]]
            hard_idx_batch_.append(hard_idx)

            if not training:
                all_triplets, ans_list = triplet_batch[i], answer_batch[i]
                selected_triplets = [all_triplets[idx] for idx in select_idx]
                selected_entities = [t[0] for t in selected_triplets] + [t[-1] for t in selected_triplets]
                
                recall = [ans for ans in ans_list if ans in selected_entities]
                step = [t for t in selected_triplets if (t[0] in ans_list or t[-1] in ans_list)]
                
                reward_loggings["recall"].append(float(len(recall) / len(ans_list)))
                reward_loggings["step"].append(float(len(step) / len(selected_triplets)))
                reward_loggings["ranking"].append(get_avg_ranks(all_triplets, select_idx, ans_list))
                
        if not training:
            reward_loggings = {k:np.mean(v).item() for k,v in reward_loggings.items()}
            return 0, reward_loggings
        
        # loss, reward_loggings = self.cal_loss(logits_batch, relevant_idx_batch, relevant_idx_in_path_batch, epoch=epoch)
        loss, reward_loggings = self.cal_loss_with_weights(logits_batch, relevant_idx_batch, hard_idx_batch_)
        return loss, reward_loggings

    def sampling_v3(self, tensor, training=False, K=6):
        sorted_indices = torch.argsort(tensor, descending=True)
        try:
            positive_indices = sorted_indices[:K]
        except:
            positive_indices = sorted_indices
        select_idx = positive_indices.tolist()

        return select_idx if type(select_idx) is list else [select_idx]

    def sampling_v2(self, tensor, training=False, K=6):
        indices = torch.arange(len(tensor), device=tensor.device)
    
        positive_indices = indices[tensor > -3.5]
        positive_values = tensor[tensor > -3.5]

        # if len(positive_indices) > K:
        sorted_indices = torch.argsort(positive_values, descending=True)
        positive_indices = positive_indices[sorted_indices]
        # else:
        #     sorted_indices = torch.argsort(tensor, descending=True)
        #     positive_indices = sorted_indices[:K]
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

    def reorganize(self, all_triplets, q_entities, select_idx, logits, ):
        with_topics, non_topics = [], []
        for idx in select_idx:
            triple = all_triplets[idx]
            if triple[0] in q_entities or triple[-1] in q_entities:
                with_topics.append(triple)
            else:
                non_topics.append(triple)
        
        detected_paths, visited = [], []
    
        for i in with_topics:
            for j in non_topics:
                s, t = (i[-1], j[0]) if i[0] in q_entities else (j[-1], i[0])
                motif = [i, j] if i[0] in q_entities else [j, i]
                # s == t represents 2-hop neighbors from source.
                if s == t:
                    detected_paths.append(motif)
                    visited.extend(motif)
                    continue
        
        non_topics = [i for i in non_topics if i not in visited]
        detected_paths_three_hops, detected_paths_four_hops = [], []

        for i in non_topics:
            for j in detected_paths:
                s, t = (i[-1], j[0][0]) if j[-1][-1] in q_entities else (j[-1][-1], i[0])
                motif =  [i]+j if j[-1][-1] in q_entities else j+[i]
                if s == t:
                    detected_paths_three_hops.append(motif)

        def score(lst, all_triplets, logits):
            logit_values = logits[[all_triplets.index(k) for k in lst]]
            return logit_values.mean().item()

        detected_paths = detected_paths + detected_paths_three_hops

        detected_paths = sorted(detected_paths, 
                                key=lambda lst: score(lst, all_triplets, logits), 
                                reverse=True)
        detected_paths = [p for p in detected_paths if not (p[0][0].startswith('m.') or p[0][0].startswith('g.') or p[-1][-1].startswith('m.') or p[-1][-1].startswith('g.'))]

        # triplet level
        detected_triplets = [all_triplets.index(item) for path in detected_paths for item in path]
        
        detected_paths.extend([[all_triplets[i]] for i in select_idx if i not in detected_triplets])
        detected_triplets.extend([i for i in select_idx if i not in detected_triplets])
        
        # TODO: remove duplicated
        detected_triplets = remove_duplicates(detected_triplets)

        if self.retrieval_clip:
            final_paths, triplets_budget = [], 0
            for i in detected_paths:
                final_paths.append(i)
                triplets_budget += len(i)
                if triplets_budget > self.budget:
                    break
            return detected_triplets[:self.budget], final_paths
        else:
            return detected_triplets, detected_paths


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def build_graphs(all_triplets):
    G = nx.DiGraph()
    for (a, b, c) in all_triplets:
        G.add_node(a)
        G.add_node(c)
        G.add_edge(a, c, relation=b)
    return G

def gen_path(G, source, target):
    shortest_paths = nx.all_shortest_paths(G, source=source, target=target)
    all_triplets_in_path = []
    for path in shortest_paths:
        triplets_in_path = [(path[i], G[path[i]][path[i+1]]['relation'], path[i+1]) for i in range(len(path) - 1)]
        all_triplets_in_path.append(triplets_in_path)
    return all_triplets_in_path

def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"

def get_avg_ranks(all_triples, sorted_indices, ans_list):
    ranks = []
    ans_list_ = deepcopy(remove_duplicates(ans_list))
    for rank, idx in enumerate(sorted_indices):
        to_check = all_triples[idx]
        if to_check[0] in ans_list_:
            ans_list_.remove(to_check[0])
            ranks.append(rank)
        if to_check[-1] in ans_list_:
            ans_list_.remove(to_check[-1])
            ranks.append(rank)
        if not ans_list_:
            break
    if len(ans_list_) > 0:
        ranks += [len(sorted_indices)] * len(ans_list_)
    return -np.mean(ranks)


    # TODO: Path level
    # detected_triplets = [item for path in detected_paths for item in path]
    # scattered_triplets = [[all_triplets[i]] for i in select_idx if all_triplets[i] not in detected_triplets]
    # detected_paths.extend(scattered_triplets)

    # final_paths = []
    # triplets_budget = 0
    # for i in detected_paths:
    #     final_paths.append(i)
    #     triplets_budget += len(i)
    #     if triplets_budget > len(select_idx):
    #         break
    # return final_paths


    # order 2. scattered topics + detected + scattered non-topics
    # with_topics = [i for i in select_idx if (i not in detected_triplets) and (all_triplets[i] in with_topics)]
    # non_topics = [i for i in select_idx if (i not in detected_triplets) and (all_triplets[i] in non_topics)]
    
    # detected_triplets = detected_triplets + with_topics
        # detected_paths = sorted(detected_paths, 
    #                         key=lambda lst: score(lst, all_triplets, logits), 
    #                         reverse=True)
    
    # detected_paths_three_hops = sorted(detected_paths_three_hops,
    #                                    key=lambda lst: score(lst, all_triplets, logits),
    #                                    reverse=True)


        # path level inferene
        # elif self.level == 'path':
        #     generation_batch = await self.llms.forward_pass(question_batch, hints_batch, select_batch)
        # for dat_id, g in zip(id_batch, generation_batch):
        #     print_log(dat_id, logging)
        #     print_log(g, logging)