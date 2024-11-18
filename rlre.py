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
        self.coeff = float(config['coeff'])  # trade-off coeff between RL and regularization
        self.algo = "REINFORCE"  # the default RL algorithm
        self.reward_metrics = RewardMetrics(metrics_name=config["reward_metrics"])
        self.perturb_per_sample = config["perturb_per_sample"]  # Use 1. Not used in this method.
        self.regularize = config["regularize"]
        # TODO: We may also consider perturb_per_sample > 1
        # deprecated.
        # self.warmup_epochs = config['warmup_epochs']
        # self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        # self.strategy = config['triplet2text']
    @property
    def device(self):
        return list(self.parameters())[0].device

    def reward_func(self, ):
        """
        Given query and retrieved triplets, return the rewards.
        """
        
        # LLM generation
        
        # Calculate metrics
    
    def log_prob(self, p):
        """
        Compute the expression:
        sum_{r=1}^K log(p_{i_r}) - sum_{r=1}^K log(1 - sum_{l=1}^{r-1} p_{i_l})
        """
        return torch.log(p).sum() - torch.log(1. - torch.cumsum(p, dim=0)[:-1]).sum()
    
    def cal_loss_warmup(self, ):
        pass
    
    def cal_loss_reinforce(self, prob_batch, r_batch):
        prob_batch = [self.log_prob(pr) for pr in prob_batch]
        prob_batch = torch.concar(prob_batch)
        
        r_batch = torch.concat(r_batch)
        
        rl_loss = - r_batch * prob_batch
        if self.regularize:
            pass
        return rl_loss
    
    def forward_pass(self, batch, epoch=None, warmup=False):
        # 2. generate mask for the coarsely retrieved samples.
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"]
        # batch_size = graph_batch.num_graphs
        graph_batch, q_embd_batch, relevant_idx_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device), relevant_idx_batch.to(self.device)
        # Prob_batch is used to calculate loss
        # sorted_idx_batch is used to specify the selected triplet.
        prob_batch, _, sorted_idx_batch = self.ibtn(graph_batch, triplet_batch_idx, q_embd_batch, epoch=epoch)

        masked_triplets_batch, _ = self.mask_triplet(triplet_batch, sorted_idx_batch)

        # calculate rewards relative to the baseline.
        # TODO: LLMs calc rewards
        reward_batch = self.llms(masked_triplets_batch, question_batch, answer_batch)
        
        return self.cal_loss_reinforce(prob_batch, reward_batch)

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
