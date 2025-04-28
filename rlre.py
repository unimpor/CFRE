import asyncio
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from src.utils import RewardMetrics, get_chunk_set, dc, gumbel_topk, print_log, triplet_to_str, TO_FILTER_REL
import copy
from itertools import chain, combinations
import random
import math
from collections import Counter, defaultdict, deque
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
        self.dataset = self.llms.data_name
        self.coeff1 = float(config['coeff1'])
        self.coeff2 = float(config['coeff2'])
        self.metrics_name = config["reward_metrics"]
        self.reward_metrics = RewardMetrics(self.metrics_name, dataset_name=self.dataset)
        self.K = config["ret_num"]
        self.Ktrain = config["ret_train"]
        self.evaluation = {}
        # hard negatives
        self.add_hard = kwargs.get("add_hard", True)
        self.weight = kwargs.get("weight", 1.0)
        # triplets arrangement
        self.reorder = kwargs.get("reorder", False)
        self.retrieval_clip = kwargs.get("clipping", False)
        self.budget = kwargs.get("budget", 100)
        self.level = kwargs.get("level", "triplet")
        self.mode = config['mode']
        print(self.add_hard, self.reorder, self.level, self.K, self.Ktrain)
        
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
            "hit": []
        }
        for g,q,a,d in zip(g_batch, q_batch, a_batch, id_batch):
            (f1, prec, recall), hit, (matched, not_prec, missing) = self.reward_metrics.calc_r(g,a,q)
            self.evaluation[d].update({
                "F1": f1,
                "hit": hit,
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
            reward_batch["hit"].append(hit)
        return reward_batch   
    
    def cal_loss_multi_cat(self, attn_logits, scores):
        warmup_loss = F.cross_entropy(attn_logits, scores)
        return warmup_loss, {"warmup": warmup_loss.item()}

    def cal_loss_with_weights(self, attn_logtis, relevant_idx, proxy_idx):
        logit_list, target_list, weight_list = [], [], []
        for logit, pos, prox in zip(attn_logtis, relevant_idx, proxy_idx):
            if len(pos) == 0:
                continue
            target = torch.zeros_like(logit, device=logit.device)
            target[pos] = 1.

            weight = torch.ones_like(logit, device=logit.device)
            weight[pos] = len(prox) / len(pos)
            
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

        return warmup_loss, {"warmup": warmup_loss.item()}

    def preparing(self, batch):
        # batch size = 1
        graph_batch, answer_batch, triplet_batch, entity_batch, q_embd_batch, id_batch, hints_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['entities'], batch["q_embd"], batch["id"], batch["q_entity"]
        dat_id, all_triplets, q_entities, all_entities = id_batch[0], triplet_batch[0], hints_batch[0], entity_batch[0]
        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)
        if self.mode == 'gnn_rag':
            entity2idx = {entity: idx for idx, entity in enumerate(all_entities)}
            heads, tails = [entity2idx[triple[0]] for triple in all_triplets], [entity2idx[triple[-1]] for triple in all_triplets]
            attn_logits_batch = attn_logits_batch[heads] + attn_logits_batch[tails]
            # attn_logits_batch = [attn_logits_batch[all_entities.index(triple[0])] + attn_logits_batch[all_entities.index(triple[-1])] for triple in all_triplets]
            
        select_idx = self.sampling(attn_logits_batch, K=self.K)
        select_triplets = [str(all_triplets[i]).replace("'", "") for i in select_idx]
        select_paths = self.reorganize(all_triplets, q_entities, select_idx, attn_logits_batch)

        # select_triplets = [triplet_to_str(all_triplets[i]) for i in select_idx]
        
        self.evaluation[dat_id] = {"select": select_idx, 
                                   "select_paths": select_paths,
                                   "select_triplets": select_triplets}

    async def inference(self, batch, logging=None, retrieval=True): 
        answer_batch, question_batch, id_batch, hints_batch = batch["y"], batch["q"], batch["id"], batch["q_entity"]
        cache = self.evaluation[id_batch[0]]
        select_triplets, select_paths = cache["select_triplets"], cache["select_paths"]
        # if self.level == "triplet":
        #     generation_batch = await self.llms.forward_pass(question_batch, hints_batch, [select_triplets])
        # elif self.level == "path":
        generation_batch = await self.llms.forward_pass(question_batch, hints_batch, [select_paths])
        reward_loggings = self.cal_reward([generation_batch], question_batch, answer_batch, id_batch, 0, training=False)
        return reward_loggings

    @torch.no_grad()
    def oracle_detection(self, batch, logging):
        answer_batch, question_batch, id_batch, paths_batch = batch["y"], batch["q"], batch["id"], batch["relevant_paths"]
        
        selected_batch = self.llms.oracle_detection(id_batch, question_batch, answer_batch, paths_batch, logging=logging)

        for d, s in zip(id_batch, selected_batch):
            # grailqa NOTE type(d) is int.
            # d = d.split("+")[0]
            if d in self.evaluation:
                self.evaluation[d].extend(s)
            else:
                self.evaluation[d] = s
            # self.evaluation[d] = {"selected": s}
            
    def forward_pass(self, batch, epoch, training=True): 
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, entity_batch_idx, relevant_idx_batch, hard_idx_batch, question_batch, q_embd_batch, id_batch, entity_batch, hints_batch = \
            batch["graph"], batch["a_entity"], batch["triplets"], batch['triplet_batch_idx'], batch['entity_batch_idx'], batch["relevant_idx"], batch['shortest_path_idx'], batch["q"], batch["q_embd"], batch["id"], batch["entities"], batch["q_entity"]

        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)  # [B, N]
        logits_batch = []
        reward_loggings = {"recall": [], "step": [], "ranking": [], "hard_len": []}
        
        for i, dat_id in enumerate(id_batch):
            all_triplets, all_entities, ans_list = triplet_batch[i], entity_batch[i], answer_batch[i]

            # gnn-rag baseline: triplets score from entities score
            if self.mode == 'gnn_rag':
                attn_logit = attn_logits_batch[entity_batch_idx == i]
                logits_batch.append(attn_logit.clone())
                
                entity2idx = {entity: idx for idx, entity in enumerate(all_entities)}
                heads, tails = [entity2idx[triple[0]] for triple in all_triplets], [entity2idx[triple[-1]] for triple in all_triplets]
                attn_logit = attn_logit[heads] + attn_logit[tails]
            else:
                attn_logit = attn_logits_batch[triplet_batch_idx == i]
                logits_batch.append(attn_logit.clone())
            select_idx = self.sampling(attn_logit, K=self.Ktrain if training else self.K)

            if not training and len(ans_list) > 0:
                selected_triplets = [all_triplets[idx] for idx in select_idx]
                selected_entities = [t[0] for t in selected_triplets] + [t[-1] for t in selected_triplets]
                
                recall = [ans for ans in ans_list if ans in selected_entities]
                step = [t for t in selected_triplets if (t[0] in ans_list or t[-1] in ans_list)]
                
                reward_loggings["recall"].append(float(len(recall) / len(ans_list)))
                reward_loggings["step"].append(float(len(step) / len(selected_triplets)))
                reward_loggings["ranking"].append(get_avg_ranks(all_triplets, select_idx, ans_list))
                reward_loggings["hard_len"].append(0)
                
        if not training:
            reward_loggings = {k:np.mean(v).item() for k,v in reward_loggings.items()}
            return 0, reward_loggings
        
        # loss, reward_loggings = self.cal_loss(logits_batch, relevant_idx_batch, relevant_idx_in_path_batch, epoch=epoch)
        loss, reward_loggings = self.cal_loss_with_weights(logits_batch, relevant_idx_batch, hard_idx_batch)
        return loss, reward_loggings

    def sampling(self, tensor, training=False, K=6):
        sorted_indices = torch.argsort(tensor, descending=True)
        try:
            positive_indices = sorted_indices[:K]
        except:
            positive_indices = sorted_indices
        select_idx = positive_indices.tolist()

        return select_idx if type(select_idx) is list else [select_idx]

    def reorganize(self, all_triplets, q_entities, select_idx, logits, max_len=2):
        
        if not self.reorder or len(select_idx) <= 5:
            select_triples = [all_triplets[i] for i in select_idx]
            select_triples = [f"{h} → [{r}] → {t}" for (h,r,t) in select_triples]
            return select_triples

        def to_match(topic, n_topic):
            s, t, motif = (topic[-1][-1], n_topic[0][0], topic+n_topic) if topic[0][0] in q_entities else (n_topic[-1][-1], topic[0][0], n_topic+topic)
            return motif if s == t else None
        
        def extract_(t):
            # extract topic and target
            return (t[0][0], t[-1][-1]) if (t[0][0] in q_entities) else (t[-1][-1], t[0][0])

        # for (i,j) in combinations(select_idx, 2):
        #     ti, tj = all_triplets[i], all_triplets[j]
        #     # remove abundant relations for the same entity pair
        #     if ti[0] == tj[0] and ti[-1] == tj[-1]:
        #         si, sj = logits[i].item(), logits[j].item()
        #         to_del = j if si > sj else i
        #     else:
        #         continue
        #     select_idx = [k for k in select_idx if k != to_del]

        with_topics, non_topics, detected_paths, logic2path, visited, detected_pairs  = [], [], [], defaultdict(set), [], set()
        for idx in select_idx:
            triple = all_triplets[idx]
            if triple[0] in q_entities or triple[-1] in q_entities:
                with_topics.append(triple)
            else:
                non_topics.append(triple)
        
        non_topics, with_topics = [[itm] for itm in non_topics if not dc(itm[1])], [[itm] for itm in with_topics if not dc(itm[1])]
        with_topics_queue = deque(with_topics)
        
        # multi-hop: BFS-based triplet expansion
        # question-centric grouping
        while with_topics_queue:
            seg = with_topics_queue.popleft()
            start, end = extract_(seg)
            if seg not in detected_paths and not end.startswith(('m.', 'g.')):
                visited.extend(seg)
                detected_paths.append(seg)
            if self.dataset != 'cwq' and len(seg) == max_len:
                continue
            for b in non_topics:
                # if dc(b[0][1]):
                #     continue
                motif = to_match(seg, b)
                if motif and frozenset(extract_(motif)) not in detected_pairs:
                    with_topics_queue.append(motif)
                    detected_pairs.add(frozenset(extract_(motif)))

        detected_paths = sorted(detected_paths, 
                                key=lambda lst: score(lst, all_triplets, logits), 
                                reverse=True)
        
        # supplementary: non-question centric grouping
        if self.dataset not in ['cwq', 'graphq',]:
            non_topics = [p[0] for p in non_topics if p[0] not in visited]
            candidates = [t for t in all_triplets if set(t) & set(q_entities)]
            for p in non_topics:
                match = next(
                    (
                        [c, p] if c[-1] == p[0] else [p, c]
                        for c in candidates
                        if c[-1] == p[0] or c[0] == p[-1]
                    ),
                    None
                )
                if match and frozenset(extract_(match)) not in detected_pairs and not check_abstract(match):
                    detected_paths.append(match)
                    detected_pairs.add(frozenset(extract_(match)))
        
        # multi-answer
        for p in detected_paths:
            p_ = list(chain.from_iterable(p))
            logics, target = (p_[:-1], {p_[-1]}) if p_[0] in q_entities else (p_[1:], {p_[0]})
            for i in range(1, len(logics) - 1):
                if logics[i].startswith(('m.', 'g.')):
                    logics[i] = 'ABS-Identifier'
            logic2path[tuple(logics)].update(target)

        detected_paths = []

        for k,v in logic2path.items():
            p = k + (v,) if k[0] in q_entities else (v,) + k
            p = [list(p[i:i+3]) for i in range(0, len(p), 3)]
            detected_paths.append(p)
        
        # multi-entity
        if len(detected_paths) > 1 and len(q_entities) > 1:
            detected_paths = merging_(detected_paths, q_entities)
        
        # detected_triplets = [item for path in detected_paths for item in path]
        # detected_triplets = post_processing(detected_triplets)
        # detected_triplets = [str(item).replace("'", "") for item in detected_triplets]
        # detected_triplets = [paths_to_string(p) for p in detected_paths]
        detected_paths = remove_duplicates(paths_to_string(detected_paths))
        
        return detected_paths


def paths_to_string(detected_paths, chunk_size=10):
    final_results = []
    for triples in detected_paths:
        for result in get_chunk_set(triples[0][0], chunk_size):
            for idx in range(len(triples)-1):
                _, p, o = triples[idx]
                if o == 'ABS-Identifier':
                    continue
                result += f" → [{p}] → {o}"
            _, p, o = triples[-1]
            for end in get_chunk_set(o, chunk_size):
                final_results.append(result + f" → [{p}] → {end}")
    return final_results


# def paths_to_string(triples):
#     result = triples[0][0]
#     result = result if type(result) is str else " | ".join(result)
#     for i, (s, p, o) in enumerate(triples):
#         o = o if type(o) is str else " | ".join(o)
#         if i < len(triples) - 1 and o == 'ABS-Identifier':
#             continue
#         result += f" → [{p}] → {o}"
#     return result.replace("'", "")

# def post_processing(all_paths):
#     expanded_triples = []
#     for triples in all_paths:
#         #[(), ()]
#         triples = [i for t in triples for i in t]
#         if isinstance(triples[0], set):
#             for chunk in get_chunk_set(triples[0], 5):
#                 expanded_triples.append([chunk] + triples[1:])
#         elif isinstance(triples[-1], set):
#             for chunk in get_chunk_set(triples[-1], 5):
#                 expanded_triples.append(triples[:-1] + [chunk])
#         else:
#             expanded_triples.append(triples)
#     return [[tuple(p[i:i+3]) for i in range(0, len(p), 3)] for p in expanded_triples]

# def post_processing(triples):

#     def chunk_set(s, size):
#         s = list(s)
#         return [set(s[i:i+size]) for i in range(0, len(s), size)]

#     expanded_triples = []
#     for h, rel, t in triples:
#         if isinstance(h, set):
#             for chunk in chunk_set(h, 5):
#                 expanded_triples.append((chunk, rel, t))
#         elif isinstance(t, set):
#             for chunk in chunk_set(t, 5):
#                 expanded_triples.append((h, rel, chunk))
#         else:
#             expanded_triples.append((h, rel, t))
#     return expanded_triples


def score(lst, all_triplets, logits):
    logit_values = logits[[all_triplets.index(k) for k in lst]]
    return logit_values.mean().item()

def check_abstract(p):
    return p[0][0].startswith(('m.', 'g.')) or p[-1][-1].startswith(('m.', 'g.'))

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def merging_(lst_, q_entities):
    """
    return: visited (used to merge), merged
    """
    visited, intersects, lst = [], [], deepcopy(lst_)

    def extract_(t):
        start, end = (t[0][0], t[-1][-1]) if (t[0][0] in q_entities) else (t[-1][-1], t[0][0])
        return start, deepcopy(end)

    for i, motif in enumerate(lst[:-1]):
        topic, final_targets = extract_(motif)
        if motif in visited:
            continue
        motif = [motif]
        for j in range(i+1, len(lst)):
            topic_cdt, targets_cdt = extract_(lst[j])
            # if topic_cdt not in final_topics and final_targets & targets_cdt and lst[j] not in visited:
            if topic_cdt != topic and final_targets & targets_cdt and lst[j] not in visited:
                motif.append(lst[j])
                visited.append(lst[j])
                final_targets = final_targets & targets_cdt
        # final_targets = next(iter(final_targets)) if len(final_targets) == 1 else final_targets
        # process motifs
        for i, path in enumerate(motif):
            for j, triple in enumerate(path):
                for k, itm in enumerate(triple):
                    if isinstance(itm, set):
                        motif[i][j][k] = deepcopy(final_targets)
        # motif = [i for t in motif for i in t]
        # motif = [i if not isinstance(i, set) else final_targets for i in motif]
        # motif = [tuple(motif[i:i+3]) for i in range(0, len(motif), 3)]

        # if motif not in intersects:
        visited.append(lst[i])
        intersects.extend(motif)
    
    return intersects


def merging(lst_, q_entities, budget=10):
    """
    return: visited (used to merge), merged
    """
    visited, intersects, lst, merged_num = [], [], deepcopy(lst_), 0

    def extract_(t):
        return (t[0][0], t[-1][-1]) if (t[0][0] in q_entities) else (t[-1][-1], t[0][0])

    if len(q_entities) == 1 or len(lst) == 1:
        return lst_

    for i, motif in enumerate(lst[:-1]):
        # if merged_num > budget:
        #     break
        merged, (topic, target) = False, extract_(motif)
        if motif in visited:
            continue
        for j in range(i+1, len(lst)):
            topic_cdt, target_cdt = extract_(lst[j])
            if topic != topic_cdt and target == target_cdt and lst[j] not in visited:
                merged = True
                motif.extend(lst[j])
                visited.append(lst[j])
        if merged and motif not in intersects:
            merged_num += 1
            visited.append(lst[i])
            intersects.append(motif)
    return intersects + [p for p in lst_ if p not in visited]

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



        #     def mask_triplet(self, triplets_batch, select_idx_batch, to_str=True):
        # """
        # `strategy` can be set as "drop" or "mask", "drop" as default.
        # Mask tokenized triplets using attn
        # # Note this method is achieved based on sample-wise not batch-wise.
        # # we prefer "drop" strategy with ordered K-sampling w.o. replacement.
        # """
        # # Example: triplets converted into strings

        # # Load the tokenizer
        # def triplet_to_str(triplet):
        #     return f"({triplet[0]},{triplet[1]},{triplet[2]})"
        # # Tokenize each triplet string
        # masked_attns_batch, masked_triplets_batch = [], []
        # for triplets, keep_idx in zip(triplets_batch, select_idx_batch):
        # # In this strategy, just drop the unselected triplets.
        #     # keep_idx = [idx for idx, score in enumerate(attns) if score.item() == 1]
        #     select_triplets = [triplet_to_str(triplets[idx]) for idx in keep_idx]
        #     masked_triplets_batch.append(select_triplets)
        #     # assert attns.shape == triplets_token_ids.shape
        #     # masked_token_ids = attns * triplets_token_ids
        # return masked_triplets_batch, masked_attns_batch
        # # return masked_token_ids


            #     for (i,j) in itertools.combinations(with_topics, 2):
            # # remove self-loop
            # if i[0] == j[-1] and i[-1] == j[0]:
            #     to_del = i if i[-1] in q_entities else j
            # # remove abundant relations for the same entity pair
            # elif i[0] == j[0] and i[-1] == j[-1]:
            #     si, sj = logits[all_triplets.index(i)].item(), logits[all_triplets.index(j)].item()
            #     to_del = j if si > sj else i
            # else:
            #     continue
            # if to_del in with_topics:
            #     with_topics.remove(to_del)
            #     select_idx.remove(all_triplets.index(to_del))


#         if self.retrieval_clip:
            # final_paths, triplets_budget = [], 0
            # for i in detected_paths:
            #     final_paths.append(i)
            #     triplets_budget += len(i)
            #     if triplets_budget > self.budget:
            #         break
            # return detected_triplets[:self.budget], final_paths
    
                # if len(all_triplets) > 10 and self.add_hard:
                # all_paths = self.reorganize(all_triplets, q_entities, select_idx, attn_logit, training_mode=True)
                # for p in all_paths:
                #     p_all_items = [e for tri in p for e in tri]
                #     p_all_idx = [all_triplets.index(tri) for tri in p]
                #     if not set(p_all_items) & set(ans_list) and not set(p_all_idx) & set(pos_idx):
                #         hard_idx.extend(p_all_idx)



                #             if self.training and self.hard_path:
                # hard_negatives = get_hard_answers(all_entities, hard_paths_cache[sample_id]['not_prec'])
                # # assert hard neg not intersected with ans
                # hard_negatives = [i for i in hard_negatives if i not in sample["a_entity"]]
                # # assert len(set(hard_negatives) & set(sample["a_entity"])) == 0
                # # hard_negatives_paths = get_paths(all_triplets, sample["q_entity"], hard_negatives)
                # # hard_positive_paths = get_pos_paths(oracle_paths, hard_paths_cache[sample_id]['missing'])
                # hard_negatives_paths = []
                # for path in hard_paths_cache[sample_id]['select_paths']:
                #     path_entities = [i for t in path for i in t]
                #     if set(path_entities) & set(hard_negatives):
                #         hard_negatives_paths.append(path)


            #     hard_neg_path_idx_ = path2tid(all_triplets, hard_negatives_paths)
            # hard_neg_path_idx = [i for i in hard_neg_path_idx_ if i not in oracle_path_idx]
            # # if len(set(oracle_path_idx) & set(hard_neg_path_idx)) > 0:
            # #     print(len(set(oracle_path_idx) & set(hard_neg_path_idx)))
            # #     num += 1
            # hard_pos_path_idx = path2tid(all_triplets, hard_positive_paths)

            # # shortest_path_idx = [all_triplets.index(item) for path in shortest_paths for item in path]
            # # shortest_path_idx = remove_duplicates(shortest_path_idx)

            # # map {-1, 0, 1} in shortest path to {1, 2, 3} 
            # scores = -torch.ones(len(all_triplets))
            # # for path, score in zip(shortest_paths, dat):
            # #     for t in path:
            # #         scores[all_triplets.index(t)] = max(score, -1)
            # # scores = (scores + 1).long()