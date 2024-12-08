import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import PLACEHOLDER, RewardMetrics, gumbel_topk
import copy
import math


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
        self.algo = "REINFORCE"  # the default RL algorithm
        self.metrics_name = config["reward_metrics"]
        self.reward_metrics = RewardMetrics(self.metrics_name)
        self.perturb_per_sample = config["perturb_per_sample"]  # Use 1. Not used in this method.
        self.regularize = config["regularize"]
        self.baseline = torch.load("/home/comp/cscxliu/derek/CFRE/datasets/webqsp/checkpoints/metrics_rev_scored_100.pth")
        self.baseline_cache = {}
        self.set_moving_baseline = config["set_moving_baseline"]
        self.eps = 1e-5  # for numerical stability
        self.tau = float(config["tau"])  # for building distribution
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.strategy = config["filtering"]
        self.filter_num_or_ratio = config["filtering_num_or_ratio"]
        self.gumbel_strength = config["gumbel_strength"]
        self.tau = float(config["tau"])
        self.baseline_order_invariant = config["baseline_order_invariant"]
        self.noise_generator = torch.Generator(device=self.device)
        self.sigma = nn.LogSigmoid()
        # deprecated.
        # self.add_gumbel = config["gumbel"]
        # self.constant_ratio = config["constant_ratio"]
        # self.warmup_epochs = config['warmup_epochs']
        # self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        # self.strategy = config['triplet2text']
    @property
    def device(self):
        return self.retriever.device

    def cal_reward(self, g_batch, q_batch, a_batch, id_batch, idx_batch, training):
        """
        Given query and retrieved triplets, return the reward rerlative to the ``baseline''.
        # TODO: We may need to change it to single inference version.
        """
        reward_batch = {
            "F1": torch.empty(0, dtype=torch.float),
            "precision": torch.empty(0, dtype=torch.float),
            "recall": torch.empty(0, dtype=torch.float),
        }
        for g,q,a,d,idx in zip(g_batch, q_batch, a_batch, id_batch, idx_batch):
            f1_abs, prec_abs, recall_abs = self.reward_metrics.calc_r(g,a,q)
            # only the training needs relative to baselines.
            # if self.set_moving_baseline and training:
            #     self.baseline_cache[d] = {'F1': f1, 'precision': prec, 'recall': recall}
            if training:
                baseline = self.baseline[d]
                f1, prec, recall = f1_abs-baseline["F1"], prec_abs-baseline["precision"], recall_abs-baseline["recall"]
                if recall > 0:
                    print(f"Update baseline for {d}.")
                    self.baseline[d] = {
                        "F1": f1_abs,
                        "precision": prec_abs,
                        "recall": recall_abs,
                        "selection": idx.cpu().clone()
                    }
            else:
                f1, prec, recall = f1_abs, prec_abs, recall_abs
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
            return torch.tensor(0.0, device=p.device)
        return torch.log(p + self.eps).mean(dim=0, keepdim=True) - torch.log(1. - torch.cumsum(p, dim=0)[:-1] + self.eps).mean(dim=0, keepdim=True)
    
    def log_prob_(self, p):
        """
        w.o. the order prior
        """
        if p.numel() == 0:
            return torch.tensor(0.0, device=p.device)
        return torch.log(p + self.eps).mean(dim=0, keepdim=True)
    
    def log_prob_weighted(self, p):
        pass

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
        prob_batch = [self.log_prob_(p[idx1])-self.log_prob_(p[idx2]) for p, idx1, idx2 in zip(prob_batch, indices_batch["select_sub_ref"], indices_batch["ref_sub_select"])]
        prob_batch = torch.cat(prob_batch)

        r_batch = self.r_post_process(r_batch).to(prob_batch.device)
        assert prob_batch.shape == r_batch.shape

        rl_loss = - self.sigma(r_batch * prob_batch).mean()
        return rl_loss
    
    def forward_pass(self, batch, epoch=None, warmup=False, training=True): 
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"]
        # if epoch == 0 and training:
        #     # Prepare baseline. Will commented later.
        #     select_idx_batch = [torch.arange(math.ceil(len(t) * 0.3)) for t in triplet_batch]
        #     masked_triplets_batch, _ = self.mask_triplet(triplet_batch, select_idx_batch)
        #     generation_batch = self.llms(question_batch, masked_triplets_batch)
        #     for g,q,a,d,s in zip(generation_batch, question_batch, answer_batch, id_batch, select_idx_batch):
        #         f1, prec, recall = self.reward_metrics.calc_r(g,a,q)
        #         self.baseline[d] = {
        #             "F1": f1,
        #             "precision": prec,
        #             "recall": recall,
        #             "selection": s
        #         }

        # batch_size = graph_batch.num_graphs
        graph_batch, q_embd_batch, relevant_idx_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device), relevant_idx_batch.to(self.device)
        # prob_batch, _, _, select_idx_batch, logits_batch = self.retriever(graph_batch, triplet_batch_idx, q_embd_batch, epoch=epoch) if not training \
            # else self.retriever(graph_batch, triplet_batch_idx, q_embd_batch, epoch, id_batch=id_batch, baseline=self.baseline)
        attn_logits_batch = self.retriever(graph_batch, q_embd_batch)

        prob_batch = []
        indices_batch = {
            "select": [],
            "select_sub_ref": [],
            "ref_sub_select": [],
            "select_cap_ref": []
        }
        for i in range(len(id_batch)):
            attn_logit = attn_logits_batch[triplet_batch_idx == i]
            attn_prob = (attn_logit / self.tau).softmax(dim=0)
            prob_batch.append(attn_prob)
            _, select_idx = self.sampling(attn_logit, seed=10*epoch, training=training)  # get the index of chosen triplets
            if training:
                ref_idx = torch.tensor(self.baseline[id_batch[i]]["selection"], device=self.device)
                if self.baseline_order_invariant:
                    select_idx = self.keep_order(ref_idx, select_idx)
                indices_batch["select_sub_ref"].append(select_idx[~torch.isin(select_idx, ref_idx)])
                indices_batch["ref_sub_select"].append(ref_idx[~torch.isin(ref_idx, select_idx)])
                indices_batch["select_cap_ref"].append(ref_idx[torch.isin(ref_idx, select_idx)])
            indices_batch["select"].append(select_idx)

        # TODO: 3. This part can be converted to single inference, not batch inference. Single inference: can be moved into the loop.    
        masked_triplets_batch, _ = self.mask_triplet(triplet_batch, indices_batch["select"])
        generation_batch = self.llms(question_batch, masked_triplets_batch)
        reward_batch = self.cal_reward(generation_batch, question_batch, answer_batch, id_batch, indices_batch["select"], training=training)
        reward_loggings = {k:torch.mean(v).item() for k,v in reward_batch.items()}
        
        if not training:
            return 0, reward_loggings

        loss = self.coeff1 * self.cal_loss_reinforce(prob_batch, indices_batch, reward_batch[self.metrics_name])
        reward_loggings["reinforce"] = loss.item()
        
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
        print(reward_loggings)
        return loss, reward_loggings

    def sampling(self, att_log_logit, seed=None, training=True):
        """
        strategy = "idp-bern" or "topk"
        K only applies when `strategy` set to "topk"
        """
        if self.strategy == "topk":
            K = self.filter_num_or_ratio if type(self.filter_num_or_ratio) is int else math.ceil(len(att_log_logit) * self.filter_num_or_ratio)
            K = min(K, att_log_logit.shape[0])
            if not training:
                _, topk_indices = att_log_logit.topk(K, dim=0, largest=True, sorted=True)
                y_hard = torch.zeros_like(att_log_logit, memory_format=torch.legacy_contiguous_format).scatter_(0, topk_indices, 1.0)
                return y_hard, topk_indices
            else:
                return gumbel_topk(att_log_logit, K=K, tau=self.tau, mode="hard", dim=0, eta=self.gumbel_strength, g=self.noise_generator, seed=seed)
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
                    f1, prec, recall = self.reward_metrics.calc_r(g,a,q)
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

