import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import PLACEHOLDER, RewardMetrics


class RLRE(nn.Module):
    """
    RL driven Retriver, better aligned with LLM reasoning capability.
    """
    def __init__(self, fg_retriever, llm_model, config, **kwargs):
        super().__init__()
        self.ibtn = fg_retriever
        self.llms = llm_model
        self.coeff1 = float(config['coeff1'])  # trade-off coeff between RL and regularization
        self.coeff2 = float(config['coeff2'])
        self.algo = "REINFORCE"  # the default RL algorithm
        self.metrics_name = config["reward_metrics"]
        self.reward_metrics = RewardMetrics(self.metrics_name)
        self.perturb_per_sample = config["perturb_per_sample"]  # Use 1. Not used in this method.
        self.regularize = config["regularize"]
        self.baseline = torch.load("/home/comp/cscxliu/derek/CFRE/datasets/webqsp/processed/metrics.pth")
        self.baseline_cache = {}
        self.set_moving_baseline = config["set_moving_baseline"]
        self.eps = 1e-5  # for numerical stability
        self.tau = float(config["tau"])  # for building distribution
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        # TODO: We may also consider perturb_per_sample > 1
        # deprecated.
        # self.warmup_epochs = config['warmup_epochs']
        # self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        # self.strategy = config['triplet2text']
    @property
    def device(self):
        return list(self.parameters())[0].device

    def cal_rel_r(self, g_batch, q_batch, a_batch, id_batch, training):
        """
        Given query and retrieved triplets, return the reward rerlative to the ``baseline''.
        """
        reward_batch = {
            "F1": torch.empty(0, dtype=torch.float),
            "precision": torch.empty(0, dtype=torch.float),
            "recall": torch.empty(0, dtype=torch.float),
        }
        for g,q,a,d in zip(g_batch, q_batch, a_batch, id_batch):
            f1, prec, recall = self.reward_metrics.calc_r(g,a,q)
            # only the training needs relative to baselines.
            if self.set_moving_baseline and training:
                self.baseline_cache[d] = {'F1': f1, 'precision': prec, 'recall': recall}
            if training:
                baseline = self.baseline[d]
                f1, prec, recall = f1-baseline["F1"], prec-baseline["precision"], recall-baseline["recall"]
            reward_batch["F1"] = torch.cat((reward_batch["F1"], torch.tensor([f1])))
            reward_batch["precision"] = torch.cat((reward_batch["precision"], torch.tensor([prec])))
            reward_batch["recall"] = torch.cat((reward_batch["recall"], torch.tensor([recall])))
        return reward_batch
    
    def log_prob_ordered_sampling(self, p):
        """
        Compute the expression:
        sum_{r=1}^K log(p_{i_r}) - sum_{r=1}^K log(1 - sum_{l=1}^{r-1} p_{i_l})
        """
        return torch.log(p + self.eps).sum(dim=0, keepdim=True) - torch.log(1. - torch.cumsum(p, dim=0)[:-1] + self.eps).sum(dim=0, keepdim=True)
    
    def log_prob_(self, p):
        """
        w.o. the order prior
        """
        return torch.log(p + self.eps).sum(dim=0, keepdim=True)
    
    def cal_loss_warmup(self, attn_logtis, relevant_idx):
        target = torch.zeros_like(attn_logtis)
        target[relevant_idx] = 1.
        dir_loss = self.criterion(attn_logtis, target)
        return dir_loss
    
    def r_post_process(self, a, alpha=0.005, beta=-0.15):
        # TODO: needs to modify
        a = torch.where(a >= 0, a + alpha, a)
        # a = torch.where(a < 0, -a, a)
        # constant-ratio
        # a = torch.where(a < 0, torch.zeros_like(a), a)
        a = torch.where((a >= beta) & (a < 0), torch.zeros_like(a), a)
        return a
    
    def cal_loss_regularize(self, id_batch, logits_batch):
        loss = torch.empty(0, dtype=torch.float, device=self.ibtn.device)
        for sample_id, attn in zip(id_batch, logits_batch):
            P = (attn / self.tau).softmax(dim=0)
            Q = (self.baseline[sample_id]["logits"] / self.tau).softmax(dim=0).to(self.ibtn.device)
            loss = torch.cat((loss, torch.sum(P * (torch.log(P+self.eps)-torch.log(Q+self.eps)), dim=0, keepdim=True)))
        return loss.mean()
        
    def cal_loss_reinforce(self, prob_batch, dropped_prob_batch, r_batch):
        # version 2 -- constant ratio
        prob_batch = [self.log_prob_ordered_sampling(pr) for pr in prob_batch]
        # version 1
        # prob_batch = [self.log_prob_ordered_sampling(pr) if r>=0 else self.log_prob_(dpr) for pr, dpr, r in zip(prob_batch, dropped_prob_batch, r_batch)]
        prob_batch = torch.cat(prob_batch)
        r_batch = self.r_post_process(r_batch).to(self.ibtn.device)
        assert prob_batch.shape == r_batch.shape
        rl_loss = - (r_batch * prob_batch).mean()
        return rl_loss
    
    def forward_pass(self, batch, epoch=None, warmup=False, training=True):
        # 2. generate mask for the coarsely retrieved samples.
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"]
        # batch_size = graph_batch.num_graphs
        graph_batch, q_embd_batch, relevant_idx_batch = \
            graph_batch.to(self.ibtn.device), q_embd_batch.to(self.ibtn.device), relevant_idx_batch.to(self.ibtn.device)
        # Prob_batch is used to calculate loss
        # sorted_idx_batch is used to specify the selected triplet.
        prob_batch, dropped_prob_batch, _, sorted_idx_batch, logits_batch = self.ibtn(graph_batch, triplet_batch_idx, q_embd_batch, epoch=epoch)
        masked_triplets_batch, _ = self.mask_triplet(triplet_batch, sorted_idx_batch)

        # calculate rewards relative to the baseline.
        # TODO: LLMs calc rewards
        generation_batch = self.llms(question_batch, masked_triplets_batch)
        reward_batch = self.cal_rel_r(generation_batch, question_batch, answer_batch, id_batch, training=training)
        reward_loggings = {k:torch.mean(v).item() for k,v in reward_batch.items()}
        loss = self.coeff1 * self.cal_loss_reinforce(prob_batch, dropped_prob_batch, reward_batch[self.metrics_name])
        reward_loggings["reinforce"] = loss.item()
        if self.regularize and training:
            # only training calculates regularization
            KL_reg_loss = self.coeff2 * self.cal_loss_regularize(id_batch, logits_batch)
            loss += KL_reg_loss
            reward_loggings["KL"] = KL_reg_loss.item()
        if not self.regularize and training:
            wp_loss = self.coeff2 * self.cal_loss_warmup(torch.concat(logits_batch, dim=0), relevant_idx_batch)
            loss += wp_loss
            reward_loggings["wp"] = wp_loss.item()
        # print(loss, reward_loggings)
        if self.set_moving_baseline and training:
            for d, attn in zip(id_batch, logits_batch):
                self.baseline_cache[d]["logits"] = attn.detach().cpu().clone()
        return loss, reward_loggings

    def mask_triplet(self, triplets_batch, sorted_idx_batch):
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
        for triplets, keep_idx in zip(triplets_batch, sorted_idx_batch):
        # In this strategy, just drop the unselected triplets.
            # keep_idx = [idx for idx, score in enumerate(attns) if score.item() == 1]
            select_triplets = [triplet_to_str(triplets[idx]) for idx in keep_idx]
            masked_triplets_batch.append(select_triplets)
            # assert attns.shape == triplets_token_ids.shape
            # masked_token_ids = attns * triplets_token_ids
        return masked_triplets_batch, masked_attns_batch
        # return masked_token_ids

    @property
    def trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
    
    def freeze_llm(self):
        # Freeze LLM parameters after some epochs
        for _, param in self.llms.named_parameters():
            param.requires_grad = False
