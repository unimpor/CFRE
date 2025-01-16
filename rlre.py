import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import PLACEHOLDER, RewardMetrics, gumbel_topk
import copy
import random
import math
from collections import Counter


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
        self.baseline = torch.load("logging/webqsp/Meta-Llama-3.1-8B-Instruct/DDE/reinforce_12.24-100/baseline.pth")
        self.evaluation = {}
        for k, v in self.baseline.items():
            self.baseline[k]["exclude"] = self.list_processing(self.baseline[k]["exclude"])
            self.baseline[k]["include"] = self.list_processing(self.baseline[k]["include"])
            # self.baseline[k]["include"] = []
            # self.baseline[k]["exclude"] = []
            self.baseline[k]["K"] = self.K
        print(self.K)
        self.baseline_cache = {}
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
        self.name_func = {
            "v2": self.cal_loss_reinforce_v2,
            "v4": self.cal_loss_reinforce_v4,
            "v5": self.cal_loss_reinforce_v5,
            "v6": self.cal_loss_reinforce_v6,
        }
        # deprecated.
        # self.add_gumbel = config["gumbel"]
        # self.constant_ratio = config["constant_ratio"]
        # self.warmup_epochs = config['warmup_epochs']
        # self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        # self.strategy = config['triplet2text']
    @property
    def device(self):
        return self.retriever.device

    def cal_reward_v2(self, t, idx, q, a):
        masked_triplets_batch, _ = self.mask_triplet([t], [idx])
        g = self.llms([q], masked_triplets_batch)
        return self.reward_metrics.calc_r(g[0],a,q)

    def cal_reward(self, g_batch, q_batch, a_batch, id_batch, epoch, training):
        """
        Given query and retrieved triplets, return the reward rerlative to the ``baseline''.
        """
        reward_batch = {
            "F1": torch.empty(0, dtype=torch.float),
            "precision": torch.empty(0, dtype=torch.float),
            "recall": torch.empty(0, dtype=torch.float),
        }
        for g,q,a,d in zip(g_batch, q_batch, a_batch, id_batch):
            (f1_abs, prec_abs, recall_abs), _ = self.reward_metrics.calc_r(g,a,q)
            # only the training needs relative to baselines.
            # if self.set_moving_baseline and training:
            #     self.baseline_cache[d] = {'F1': f1, 'precision': prec, 'recall': recall}
            if training:
                baseline = self.baseline[d]
                f1, prec, recall = f1_abs-baseline["F1"], prec_abs-baseline["precision"], recall_abs-baseline["recall"]
                # if recall_abs == 1.0 and not self.baseline[d]["golden"]:
                #     print("golden first find!")
                #     self.update_num += 1
                #     self.baseline[d]["golden"] = True
                #     self.baseline[d]["selection"] = idx.cpu()

                # print(recall, baseline["recall"])
                # if (epoch+1) % 3 == 0:
                #     self.baseline[d]["positive"] = {}
                
                # if recall > 0:
                #     for i in idx.cpu().tolist():
                #         self.baseline[d]["positive"][i] = 1 if i not in self.baseline[d]["positive"] else self.baseline[d]["positive"][i] + 5
                
                # if recall > 0  and random.SystemRandom().random() < 0.5 and (epoch+1) % 5 == 0 and self.set_moving_baseline:
                #     print(f"Update baseline for {d}.")
                #     self.update_num += 1
                #     self.baseline[d]["F1"] = f1_abs
                #     self.baseline[d]["precision"] = prec_abs
                #     self.baseline[d]["recall"] = recall_abs
                #     self.baseline[d]["selection"] = idx.cpu()
            else:
                f1, prec, recall = f1_abs, prec_abs, recall_abs
                print(f1, prec, recall)
                self.evaluation[d]["F1"] = f1
                self.evaluation[d]["precision"] = prec
                self.evaluation[d]["recall"] = recall            
            reward_batch["F1"] = torch.cat((reward_batch["F1"], torch.tensor([f1])))
            reward_batch["precision"] = torch.cat((reward_batch["precision"], torch.tensor([prec])))
            reward_batch["recall"] = torch.cat((reward_batch["recall"], torch.tensor([recall])))
        return reward_batch
    
    def log_prob_ordered_sampling(self, p):
        """
        Compute the expression:
        sum_{r=1}^K log(p_{i_r}) - sum_{r=1}^K log(1 - sum_{l=1}^{r-1} p_{i_l})
        """
        if p.numel() == 0:
            return torch.tensor([0.0], device=p.device)
        return torch.log(p + self.eps).mean(dim=0, keepdim=True) - torch.log(1. - torch.cumsum(p, dim=0)[:-1] + self.eps).mean(dim=0, keepdim=True)
    
    def log_prob_(self, p):
        """
        w.o. the order prior
        """
        if p.numel() == 0:
            return torch.tensor([0.0], device=p.device)
        return torch.log(p + self.eps).mean(dim=0, keepdim=True)
    
    def log_prob_weighted(self, p_org, idx1, dat_id):
        p = p_org[idx1]
        if p.numel() == 0:
            return torch.tensor([0.0], device=p.device)
        logp = torch.log(p + self.eps)
        w = [self.baseline[dat_id]["positive"][i] for i in idx1.cpu().tolist()]
        w = torch.tensor(w, device=logp.device, dtype=logp.dtype)
        return torch.sum(logp * w, dim=0, keepdim=True) / torch.sum(w, dim=0, keepdim=True)

    def r_post_process(self, a, alpha=0.001, beta=-0.15):
        """
        Update 3rd Dec: Deprecated Feature.
        post process r. Let's see if this makes sense. Options:
        1. add a small alpha when r is non-negative:
             `a = torch.where(a >= 0, a + alpha, a)`;
        2. set negative r to zero when it does not exceed a threshold beta:
             `a = torch.where((a >= beta) & (a < 0), torch.zeros_like(a), a)`
        3. Do not discourage, which proves not effective :(
             `a = torch.where(a < 0, torch.zeros_like(a), a)`
        """
        # option 1
        # a = torch.where(a >= 0, a + alpha, a)
        # option 2
        # a = torch.where((a >= beta) & (a < 0), torch.zeros_like(a), a)
        # option 3
        # a = torch.where(a < 0, torch.zeros_like(a), a)
        
        return torch.sign(a)
    
    def cal_loss_warmup(self, attn_logtis, relevant_idx):
        target = torch.zeros_like(attn_logtis)
        target[relevant_idx] = 1.
        dir_loss = self.criterion(attn_logtis, target)
        return dir_loss
    
    def cal_loss_regularize(self, id_batch, prob_batch):
        loss = torch.empty(0, dtype=torch.float, device=self.device)
        for sample_id, P in zip(id_batch, prob_batch):
            Q = (self.baseline[sample_id]["logits"] / self.tau).softmax(dim=0).to(self.device)
            loss = torch.cat((loss, torch.sum(P * (torch.log(P+self.eps)-torch.log(Q+self.eps)), dim=0, keepdim=True)))
        return loss.mean()
        
    def cal_loss_reinforce(self, prob_batch, indices_batch, r_batch):
        # strategy 3.1 -- default
        # prob_batch = [self.log_prob_(p[idx]) for p, idx in zip(prob_batch, indices_batch["select"])]
        # strategy 3.2
        r_batch = self.r_post_process(r_batch).to(self.device)

        re_prob_batch = [self.log_prob_(p[idx1])-self.log_prob_(p[idx2]) for p, idx1, idx2 in zip(prob_batch, indices_batch["select_sub_ref"], indices_batch["ref_sub_select"])]
        gt_prob_batch = [self.log_prob_(p[idx]) for p, idx in zip(prob_batch, indices_batch['gt'])]
        
        re_prob_batch = torch.cat(re_prob_batch)
        gt_prob_batch = torch.cat(gt_prob_batch)

        rl_loss = - self.sigma(r_batch * re_prob_batch + gt_prob_batch).mean()
        return rl_loss
    
    def cal_loss_reinforce_v2(self, prob_batch, indices_batch, r_batch):
        r_batch = self.r_post_process(r_batch).to(self.device)
        # r_batch = r_batch.to(self.device)
        
        for i,r in enumerate(r_batch):
            dat_id = indices_batch['id'][i]
            ref_sub_select_indices = indices_batch['ref_sub_select'][i]
            select_sub_ref_indices = indices_batch['select_sub_ref'][i]
            if r > 0:
                self.baseline[dat_id]["exclude"] = self.combine_idx(self.baseline[dat_id]["exclude"], ref_sub_select_indices)
            if r < 0:
                self.baseline[dat_id]["exclude"] = self.combine_idx(self.baseline[dat_id]["exclude"], select_sub_ref_indices)
            if r == 0:
                self.baseline[dat_id]["exclude"] = self.combine_idx(self.baseline[dat_id]["exclude"], ref_sub_select_indices + select_sub_ref_indices)
        
        ref_prob_batch = [(self.baseline[d]["logits"] / self.tau).softmax(dim=0).to(self.device) for d in indices_batch['id']]

        ## self.combine_idx(idx1, gt_idx) ;; self.combine_idx(idx2, gt_idx)
        prob_batch_pos = [self.log_prob_(p[idx1]) if r<=0 else self.log_prob_(p[gt_idx]) for p, idx1, r, gt_idx in zip(prob_batch, indices_batch["select_sub_ref"], r_batch, indices_batch["select"])]
        prob_batch_neg = [self.log_prob_(p[idx2]) if r>0 else self.log_prob_(p[gt_idx]) for p, idx2, r, gt_idx in zip(prob_batch, indices_batch["ref_sub_select"], r_batch, indices_batch["ref"])]        
        
        prob_batch_pos_ref = [self.log_prob_(p[idx1]) if r<=0 else self.log_prob_(p[gt_idx]) for p, idx1, r, gt_idx in zip(ref_prob_batch, indices_batch["select_sub_ref"], r_batch, indices_batch["select"])]
        prob_batch_neg_ref = [self.log_prob_(p[idx2]) if r>0 else self.log_prob_(p[gt_idx]) for p, idx2, r, gt_idx in zip(ref_prob_batch, indices_batch["ref_sub_select"], r_batch, indices_batch["ref"])]        
        
        prob_batch_pos = torch.cat(prob_batch_pos)
        prob_batch_neg = torch.cat(prob_batch_neg)
        prob_batch_pos_ref = torch.cat(prob_batch_pos_ref)
        prob_batch_neg_ref = torch.cat(prob_batch_neg_ref)

        rl_loss = self.coeff1 * r_batch * ((prob_batch_pos-prob_batch_pos_ref)-(prob_batch_neg-prob_batch_neg_ref))
        rl_loss = rl_loss[r_batch!=0]

        rl_loss = -F.logsigmoid(rl_loss).mean()
        return rl_loss

    def cal_loss_reinforce_v4(self, prob_batch, indices_batch):

        ref_prob_batch = [(self.baseline[d]["logits"] / self.tau).softmax(dim=0).to(self.device) for d in indices_batch['id']]
        r_batch = [self.baseline[d][f"d-{self.metrics_name}"] for d in indices_batch['id']]
        r_batch = torch.tensor(r_batch)
        r_batch = self.r_post_process(r_batch).to(self.device)
        # r_batch = r_batch.to(self.device)
        prob_batch_pos, prob_batch_neg, prob_batch_pos_ref, prob_batch_neg_ref = [], [], [], []
        
        for i, r in enumerate(r_batch):
            dat_id = indices_batch['id'][i]
            ref_sub_select_indices = indices_batch['ref_sub_select'][i]
            select_sub_ref_indices = indices_batch['select_sub_ref'][i]
            select_cap_ref_indices = indices_batch['select_cap_ref'][i]
            select_indices = indices_batch['select'][i]
            ref_indices = indices_batch['ref'][i]
            weak_indices = indices_batch['gt'][i]

            p, p_ref = prob_batch[i], ref_prob_batch[i]
            if r > 0:
                # self.baseline[dat_id]["exclude"] = self.combine_idx(self.baseline[dat_id]["exclude"], ref_sub_select_indices)
                # include = select_indices if r < self.baseline[dat_id]["recall"] else select_sub_ref_indices
                include = select_indices + random.sample(weak_indices, min(50, len(weak_indices)))
                exclude = ref_sub_select_indices
            elif r < 0:
                # self.baseline[dat_id]["exclude"] = self.combine_idx(self.baseline[dat_id]["exclude"], select_sub_ref_indices)
                # include = ref_indices if -r < 0.4 * self.baseline[dat_id]["recall"] else ref_sub_select_indices
                include = ref_indices + random.sample(weak_indices, min(50, len(weak_indices)))
                exclude = select_sub_ref_indices
            elif r == 0:
                # self.baseline[dat_id]["exclude"] = self.combine_idx(self.baseline[dat_id]["exclude"], ref_sub_select_indices + select_sub_ref_indices)
                include = select_cap_ref_indices + random.sample(weak_indices, min(50, len(weak_indices)))
                exclude = ref_sub_select_indices + select_sub_ref_indices
            
            prob_batch_pos.append(self.log_prob_(p[include]))
            prob_batch_pos_ref.append(self.log_prob_(p_ref[include]))
            prob_batch_neg.append(self.log_prob_(p[exclude]))
            prob_batch_neg_ref.append(self.log_prob_(p_ref[exclude]))
        
        prob_batch_pos = torch.cat(prob_batch_pos)
        prob_batch_neg = torch.cat(prob_batch_neg)
        prob_batch_pos_ref = torch.cat(prob_batch_pos_ref)
        prob_batch_neg_ref = torch.cat(prob_batch_neg_ref)

        rl_loss = self.coeff1 * ((prob_batch_pos-prob_batch_pos_ref)-(prob_batch_neg-prob_batch_neg_ref))
        
        rl_loss = -F.logsigmoid(rl_loss).mean()
        return rl_loss

    def cal_loss_reinforce_v5(self, prob_batch, indices_batch):

        ref_prob_batch = [(self.baseline[d]["logits"] / self.tau).softmax(dim=0).to(self.device) for d in indices_batch['id']]

        prob_batch_pos, prob_batch_neg, prob_batch_pos_ref, prob_batch_neg_ref = [], [], [], []
        
        for i, dat_id in enumerate(indices_batch['id']):
            select_indices = indices_batch['select'][i]
            ref_indices = indices_batch['ref'][i]
            weak_indices = indices_batch['gt'][i]

            p, p_ref = prob_batch[i], ref_prob_batch[i]
            
            if self.baseline[dat_id][f"d-{self.metrics_name}"] > 0:
                # exclude = [i for i in range(len(p)) if i not in self.combine_idx(self.baseline[dat_id]["include"], weak_indices)]
                # exclude = self.combine_idx(exclude, self.baseline[dat_id]["exclude"])
                include = self.sort_by_freq(self.baseline[dat_id]["include"], weak_indices)
                include = [i for i in self.baseline[dat_id]["select"].tolist() if i not in include] + include               
                # include = random.sample(include, min(len(include), len(exclude)))

            elif self.baseline[dat_id][f"d-{self.metrics_name}"] <= 0:
                include = ref_indices
            
            # if self.baseline[dat_id][f"rf-{self.metrics_name}"] + self.baseline[dat_id][f"d-{self.metrics_name}"] == 1:
            #     include = self.baseline[dat_id]["select"].tolist()
                
            exclude = [i for i in range(len(p)) if i not in include]
            exclude = self.combine_idx(exclude, self.baseline[dat_id]["exclude"])
            exclude = list(set(exclude) & set(select_indices))

            include = [x for x in include if x not in select_indices]
            include = random.sample(include, min(len(include), len(exclude)))
            # if self.baseline[dat_id][f"d-{self.metrics_name}"] == 0:
            #     include = select_cap_ref_indices
            # if self.baseline[dat_id][f"rf-{self.metrics_name}"] + self.baseline[dat_id][f"d-{self.metrics_name}"] == 1:
            #     include = self.baseline[dat_id]["selection"]
            #     exclude = [i for i in range(len(p)) if i not in include]
            # include = self.combine_idx(weak_indices, include)
            # exclude = [i for i in range(len(p)) if i not in self.combine_idx(include, weak_indices)]

            # include = random.sample(include, min(100, len(include)))
            # exclude = random.sample(exclude, min(100, len(exclude)))

            prob_batch_pos.append(self.log_prob_(p[include]))
            prob_batch_pos_ref.append(self.log_prob_(p_ref[include]))
            prob_batch_neg.append(self.log_prob_(p[exclude]))
            prob_batch_neg_ref.append(self.log_prob_(p_ref[exclude]))
        
        prob_batch_pos = torch.cat(prob_batch_pos)
        prob_batch_neg = torch.cat(prob_batch_neg)
        prob_batch_pos_ref = torch.cat(prob_batch_pos_ref)
        prob_batch_neg_ref = torch.cat(prob_batch_neg_ref)

        rl_loss = self.coeff1 * ((prob_batch_pos-prob_batch_pos_ref)-(prob_batch_neg-prob_batch_neg_ref))
        # rl_loss = self.coeff1 * (prob_batch_pos-prob_batch_neg)
        rl_loss = -F.logsigmoid(rl_loss).mean()
        return rl_loss

    def cal_loss_reinforce_v6(self, prob_batch, indices_batch):
        ref_prob_batch = [(self.baseline[d]["logits"] / self.tau).softmax(dim=0).to(self.device) for d in indices_batch['id']]

        prob_batch_pos, prob_batch_neg, prob_batch_pos_ref, prob_batch_neg_ref = [], [], [], []
        
        for i, dat_id in enumerate(indices_batch['id']):
            select_indices = indices_batch['select'][i]
            ref_indices = indices_batch['ref'][i]
            weak_indices = indices_batch['gt'][i]

            p, p_ref = prob_batch[i], ref_prob_batch[i]
            
            if self.baseline[dat_id][f"d-{self.metrics_name}"] > 0:
                # exclude = [i for i in range(len(p)) if i not in self.combine_idx(self.baseline[dat_id]["include"], weak_indices)]
                # exclude = self.combine_idx(exclude, self.baseline[dat_id]["exclude"])
                include = self.sort_by_freq(self.baseline[dat_id]["include"], weak_indices)
                include = [i for i in self.baseline[dat_id]["select"].tolist() if i not in include] + include               
                # include = random.sample(include, min(len(include), len(exclude)))

            elif self.baseline[dat_id][f"d-{self.metrics_name}"] <= 0:
                include = ref_indices
            
            # if self.baseline[dat_id][f"rf-{self.metrics_name}"] + self.baseline[dat_id][f"d-{self.metrics_name}"] == 1:
            #     include = self.baseline[dat_id]["select"].tolist()

            _, select_indices = self.sampling(p, training=False, K=len(include))
            select_indices = select_indices.tolist()

            exclude = [i for i in range(len(p)) if i not in include]
            exclude = self.combine_idx(exclude, self.baseline[dat_id]["exclude"])
            exclude = list(set(exclude) & set(select_indices))

            include = [x for x in include if x not in select_indices]

            prob_batch_pos.append(self.log_prob_(p[include]))
            prob_batch_pos_ref.append(self.log_prob_(p_ref[include]))
            prob_batch_neg.append(self.log_prob_(p[exclude]))
            prob_batch_neg_ref.append(self.log_prob_(p_ref[exclude]))
        
        prob_batch_pos = torch.cat(prob_batch_pos)
        prob_batch_neg = torch.cat(prob_batch_neg)
        prob_batch_pos_ref = torch.cat(prob_batch_pos_ref)
        prob_batch_neg_ref = torch.cat(prob_batch_neg_ref)

        rl_loss = self.coeff1 * ((prob_batch_pos-prob_batch_pos_ref)-(prob_batch_neg-prob_batch_neg_ref))
        # rl_loss = prob_batch_pos - prob_batch_neg
        rl_loss = -F.logsigmoid(rl_loss).mean()
        return rl_loss

    @torch.no_grad()
    def pre_processing(self, batch, epoch, update_reference=False):
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"]
        
        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)
        
        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)
        
        for i, (d, q, a, t) in enumerate(zip(id_batch, question_batch, answer_batch, triplet_batch)):
            attn_logit = attn_logits_batch[triplet_batch_idx == i]
            attn_prob = (attn_logit / self.tau).softmax(dim=0)
            # if update_reference:
            _, ref_idx = self.sampling(attn_logit, training=False, K=self.baseline[d]["K"])

            if self.baseline[d]["rf-F1"] + self.baseline[d]["d-F1"] == 1.0 or self.baseline[d]["rf-F1"] == 1.0:
                print("perfect!")
                continue

            print("evaluate reference...")
            (f1_ref, prec_ref, recall_ref), ga_ref = self.cal_reward_v2(t, ref_idx, q, a)
            self.baseline[d]["rf-F1"] = f1_ref
            self.baseline[d]["rf-precision"] = prec_ref
            self.baseline[d]["rf-recall"] = recall_ref
            self.baseline[d]["logits"] = attn_logit.detach().cpu()
            self.baseline[d]["ref"] = ref_idx

            # TODO: exclude args set to []
            for j in range(20):
                _, select_idx = self.sampling(attn_logit, seed=10*epoch+j, training=True, include=relevant_idx_batch[i], exclude=[], K=self.baseline[d]["K"])  # get the index of chosen triplets
                if self.baseline_order_invariant:
                    select_idx = self.keep_order_v2(ref_idx, select_idx)
                
                select_sub_ref = select_idx[~torch.isin(select_idx, ref_idx)].tolist()
                ref_sub_select = ref_idx[~torch.isin(ref_idx, select_idx)].tolist()
                select_cap_ref = ref_idx[torch.isin(ref_idx, select_idx)].tolist()

                (f1, prec, recall), ga = self.cal_reward_v2(t, select_idx, q, a)
                print(f1 - f1_ref)
                if f1 - f1_ref > self.baseline[d].get("d-F1", -1):
                    self.baseline[d]["d-F1"] = f1 - f1_ref
                    self.baseline[d]["d-precision"] = prec - prec_ref
                    self.baseline[d]["d-recall"] = recall - recall_ref 
                    self.baseline[d]["select"] = select_idx
                if f1 == 1:
                    break
                if f1 > f1_ref:
                    self.baseline[d]["include"] += select_idx
                
                if f1 == 0:
                    # self.baseline[d]["exclude"] = self.combine_idx(self.baseline[d]["exclude"], select_idx)
                    self.baseline[d]["exclude"] += select_idx
                if f1_ref == 0:
                    self.baseline[d]["exclude"] += ref_idx
                
                if set(ga) <= set(ga_ref):
                    self.baseline[d]["exclude"] += select_sub_ref
                if set(ga_ref) <= set(ga):
                    self.baseline[d]["exclude"] += ref_sub_select
                # if f1 - f1_ref > 0:
                #     self.baseline[d]["exclude"] = self.combine_idx(self.baseline[d]["exclude"], ref_sub_select)
                # elif f1 - f1_ref == 0:
                #     self.baseline[d]["exclude"] = self.combine_idx(self.baseline[d]["exclude"], select_sub_ref + ref_sub_select)
                # else:
                #     self.baseline[d]["exclude"] = self.combine_idx(self.baseline[d]["exclude"], select_sub_ref)  
            print("-----")


    def forward_pass(self, batch, epoch, warmup=False, training=True): 
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"]

        graph_batch, q_embd_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device)

        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)

        prob_batch = []
        indices_batch = {
            "select": [],
            "select_sub_ref": [],
            "ref_sub_select": [],
            "select_cap_ref": [],
            "ref": [],
            "gt": relevant_idx_batch,
            "id": id_batch
        }
        for i, dat_id in enumerate(id_batch):
            attn_logit = attn_logits_batch[triplet_batch_idx == i]
            attn_prob = (attn_logit / self.tau).softmax(dim=0)
            prob_batch.append(attn_prob)

            if training:
                # select_idx = torch.tensor(self.baseline[id_batch[i]]["select"], device=self.device)
                _, select_idx = self.sampling(attn_logit, training=False)
                ref_idx = torch.tensor(self.baseline[id_batch[i]]["ref"], device=self.device)
                # indices_batch["select_sub_ref"].append(select_idx[~torch.isin(select_idx, ref_idx)].tolist())
                # indices_batch["ref_sub_select"].append(ref_idx[~torch.isin(ref_idx, select_idx)].tolist())
                # indices_batch["select_cap_ref"].append(ref_idx[torch.isin(ref_idx, select_idx)].tolist())
                indices_batch["ref"].append(ref_idx.tolist())
            else:
                _, select_idx = self.sampling(attn_logit, training=False)  # get the index of chosen triplets          
            indices_batch["select"].append(select_idx.tolist())
                
            # self.baseline[dat_id]["logits"] = attn_logit.detach()
        
        # if training and epoch == 0:
        #     masked_triplets_batch, _ = self.mask_triplet(triplet_batch, indices_batch["ref"])
        #     generation_batch = self.llms(question_batch, masked_triplets_batch)
        #     for g,q,a,d,s in zip(generation_batch, question_batch, answer_batch, id_batch, indices_batch["ref"]):
        #         f1, prec, recall = self.reward_metrics.calc_r(g,a,q)
        #         self.baseline[d]["F1"] = f1
        #         self.baseline[d]["precision"] = prec
        #         self.baseline[d]["recall"] = recall
        #         self.baseline[d]["selection"] = s
                
        if not training:
            masked_triplets_batch, _ = self.mask_triplet(triplet_batch, indices_batch["select"])
            generation_batch = self.llms(question_batch, answer_batch, masked_triplets_batch)
            reward_batch = self.cal_reward(generation_batch, question_batch, answer_batch, id_batch, epoch, training=training)
            reward_loggings = {k:torch.mean(v).item() for k,v in reward_batch.items()}
            return 0, reward_loggings
        
        loss = self.name_func[self.algo](prob_batch, indices_batch)
        reward_loggings = {"reinforce": loss.item()}
 
        print(reward_loggings)
        if self.regularize == "KL":
            KL_reg_loss = self.coeff2 * self.cal_loss_regularize(id_batch, prob_batch)
            loss += KL_reg_loss
            reward_loggings["KL"] = KL_reg_loss.item()
        if self.regularize == "wp":
            wp_loss = self.coeff2 * self.cal_loss_warmup(attn_logits_batch, relevant_idx_batch)
            loss += wp_loss
            reward_loggings["wp"] = wp_loss.item()
        # Update 3rd Dec: Deprecated Feature.

        # if self.set_moving_baseline:
        #     for d, logits, s_idx in zip(id_batch, logits_batch, select_idx_batch):
        #         _, topk_indices = logits.topk(s_idx.shape[0], dim=0, largest=True, sorted=True)
        #         self.baseline_cache[d] = {
        #             # "logits": logits.detach().cpu().clone(),
        #             "selection": topk_indices.detach().cpu().clone()
        #         }
        return loss, reward_loggings

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

    def sampling(self, att_log_logit, seed=None, training=True, **kwargs):
        """
        strategy = "idp-bern" or "topk"
        K only applies when `strategy` set to "topk"
        """
        if self.strategy == "topk":
            K = kwargs.get("K", None)
            if not K:
                K = self.filter_num_or_ratio if type(self.filter_num_or_ratio) is int else math.ceil(len(att_log_logit) * self.filter_num_or_ratio)
            K = min(K, att_log_logit.shape[0])
            if not training:
                _, topk_indices = att_log_logit.topk(K, dim=0, largest=True, sorted=True)
                y_hard = torch.zeros_like(att_log_logit, memory_format=torch.legacy_contiguous_format).scatter_(0, topk_indices, 1.0)
                return y_hard, topk_indices
            else:
                return gumbel_topk(att_log_logit, K=K, tau=self.tau, mode="hard", dim=0, eta=self.gumbel_strength, g=self.noise_generator, seed=seed, exclude=kwargs["exclude"], include=kwargs["include"])
        else:
            raise NotImplementedError

    def mask_triplet(self, triplets_batch, select_idx_batch):
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

    def update_baseline(self, loader):
        """
        Update 3rd Dec: Deprecated Feature.
        """
        if not self.set_moving_baseline:
            print("Not Update.")
            return False
        
        with torch.no_grad():
            for _, batch in enumerate(loader):
                graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch = \
                    batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"]
                
                select_idx_batch = [self.baseline_cache[d]["selection"] for d in id_batch]
                masked_triplets_batch, _ = self.mask_triplet(triplet_batch, select_idx_batch)
                generation_batch = self.llms(question_batch, masked_triplets_batch)

                for g,q,a,d in zip(generation_batch, question_batch, answer_batch, id_batch):
                    (f1, prec, recall), _ = self.reward_metrics.calc_r(g,a,q)
                    self.baseline_cache[d]["F1"] = f1
                    self.baseline_cache[d]["precision"] = prec
                    self.baseline_cache[d]["recall"] = recall
        # get average of baseline and baseline cache to determine if we update
        # print(len(self.baseline), len(self.baseline_cache))
        assert len(self.baseline) == len(self.baseline_cache)
        # print(len(self.baseline), len(self.baseline_cache))
        # baseline_metrics = mean(member[self.metrics_name] for member in self.baseline.values())
        # baseline_cache_metrics = mean(member[self.metrics_name] for member in self.baseline_cache.values())
        update_num = 0
        # self.baseline = {k: self.baseline[k] for k in self.baseline_cache.keys()}
        for k, v in self.baseline.items():
            if v[self.metrics_name] < self.baseline_cache[k][self.metrics_name]:
                update_num += 1
                self.baseline[k] = copy.deepcopy(self.baseline_cache[k])
        return update_num
        # if baseline_cache_metrics > baseline_metrics:
        #     self.baseline = copy.deepcopy(self.baseline_cache)
        #     print(f"{baseline_cache_metrics} larger than {baseline_metrics}. Update.")
        #     return True
        # print(f"{baseline_cache_metrics} smaller than {baseline_metrics}. Not Update.")
        # return False

    def keep_order(self, a, b):
        overlap = [x for x in b.tolist() if x in a.tolist()]
        if not overlap:
            return b
        ordered_overlap = sorted(overlap, key=lambda x: a.tolist().index(x))

        adjusted_b = []
        overlap_index = 0

        for element in b.tolist():
            if element in a.tolist():
                adjusted_b.append(ordered_overlap[overlap_index])
                overlap_index += 1
            else:
                adjusted_b.append(element)

        return torch.tensor(adjusted_b, device=self.device)

    def keep_order_v2(self, b, a):
        overlap_indices_in_b = [i for i, x in enumerate(b) if x in a]
        overlap_values = [x for x in b if x in a]

        a_prime = torch.full_like(a, fill_value=-1, device=self.device)

        for i, val in zip(overlap_indices_in_b, overlap_values):
            a_prime[i] = val

        non_overlap_values = [x for x in a if x not in b]

        non_overlap_idx = 0
        for i in range(len(a_prime)):
            if a_prime[i] == -1:
                a_prime[i] = non_overlap_values[non_overlap_idx]
                non_overlap_idx += 1
        return a_prime

    @staticmethod
    def combine_idx(idx1, idx2):
        return list(set(idx1) | set(idx2))

    @staticmethod
    def sort_by_freq(l, ref):
        result = Counter(l)
        for k, v in result.items():
            result[k] = 10 * v if k not in ref else 10 * v + 1
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return [d[0] for d in result]
    
    @staticmethod
    def list_processing(l):
        return [c.item() if torch.is_tensor(c) else c for c in l]

    def case_study(self, id_batch, question_batch, answer_batch, triplet_batch, select_idx, gt_idx):
        
        def print_log(content):
            content = content.replace("\n", "") if isinstance(content, str) else ", ".join(map(str, content))
            save_path = "fig/case_study.txt"
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(content + "\n")
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