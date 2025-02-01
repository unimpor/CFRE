import torch
import torch.nn as nn
import torch.nn.functional as F


class CFRE(nn.Module):
    def __init__(self, fg_retriever, llm_model, config, **kwargs):
        super().__init__()
        self.ibtn = fg_retriever
        self.llms = llm_model
        self.strategy = config['triplet2text']
        self.grad_normalize = config['grad_normalize']
        self.coeff = float(config['coeff'])
        self.warmup_epochs = config['warmup_epochs']
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
    @property
    def device(self):
        return list(self.parameters())[0].device

    def __loss__(self, attn_logtis, relevant_idx,):
        """
        Calculate attn-related loss
        return loss and loss_dict
        """
        # Info loss: deprecated in K-sampling w.o. replacement
        # r = self.fix_r if self.fix_r else get_r(self.decay_interval, self.decay_r, epoch, self.init_r, self.final_r)
        # info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

        # con_loss = None
        #
        # TODO: only make it as warmup, change it to bi-clfs problem.
        # attn_target = attn_logtis[relevant_idx]
        target = torch.zeros_like(attn_logtis)
        target[relevant_idx] = 1.
        dir_loss = self.criterion(attn_logtis, target)

        return dir_loss, {"dir": dir_loss.item()}

    def forward_pass(self, batch, epoch=None, warmup=False):
        # 2. generate mask for the coarsely retrieved samples.
        graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"]
        # batch_size = graph_batch.num_graphs
        graph_batch, q_embd_batch, relevant_idx_batch = \
            graph_batch.to(self.device), q_embd_batch.to(self.device), relevant_idx_batch.to(self.device)

        attn_logtis, attns_batch, sorted_idx_batch = self.ibtn(graph_batch, triplet_batch_idx, q_embd_batch, epoch=epoch)
        # 4. When we need to consider the loss?
        
            # attn_loss, loss_dict = self.__loss__(attn_logtis, relevant_idx_batch,)  # calculate attn-related loss
        loss, loss_dict = self.__loss__(attn_logtis, relevant_idx_batch,)
        
        if warmup:
            return loss, loss_dict
        # 3. generate filtered retrieval results
        # loss, loss_dict["predict"] = attn_loss, 0.
        # if not self.warmup or epoch >= self.warmup_epochs:
            # batch_masked_triple_token_ids = self.mask_triplet(triplets, attns)
        else:
            # attns = [attns[triplet_batch_idx==i] for i in range(batch_size)] 
            masked_triplets_batch, masked_attns_batch = self.mask_triplet(triplet_batch, attns_batch, sorted_idx_batch)
            # 4. LLM supervisions
            # outputs = self.llms.forward_pass(batch_masked_triple_token_ids, question, answer)
            pred_loss = self.llms.forward_pass(masked_attns_batch, masked_triplets_batch, question_batch, answer_batch)
            loss += self.coeff * pred_loss
            loss_dict["predict"] = pred_loss.item()
        return loss, loss_dict

    def mask_triplet(self, triplets_batch, attns_batch, sorted_idx_batch):
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

        def create_hook(length, indices):

            def custom_backward_hook(grad):
                lengths = torch.ones_like(grad, dtype=torch.float32)
                lengths[indices] = torch.tensor(length, dtype=torch.float32).to(self.device)
                normalized_grad = grad / lengths  # Normalize by expansion lengths
                return normalized_grad

            return custom_backward_hook

        if self.strategy == "drop":
            masked_attns_batch, masked_triplets_batch = [], []
            for triplets, attns, keep_idx in zip(triplets_batch, attns_batch, sorted_idx_batch):
            # In this strategy, just drop the unselected triplets.
                assert len(triplets) == attns.shape[0]
                # keep_idx = [idx for idx, score in enumerate(attns) if score.item() == 1]
                
                tokenized_triplets = [self.llms.tokenizer(triplet_to_str(triplets[idx]), return_tensors="pt")
                                    for idx in keep_idx]
                triplets_token_ids = torch.cat(
                    [tokenized_triplet["input_ids"].squeeze() for tokenized_triplet in tokenized_triplets]).to(self.device)

                triplet_lengths = [len(tokenized_triplet["input_ids"].squeeze()) for tokenized_triplet in
                                tokenized_triplets]
                # TODO: we may consider batch size > 1
                if self.ibtn.training and self.grad_normalize:
                    attns.register_hook(create_hook(length=triplet_lengths, indices=keep_idx))

                attns_exp = torch.cat([
                    attns[idx].expand(length) for idx, length in zip(keep_idx, triplet_lengths)
                ]).unsqueeze(1)  # expand and cut.
                
                masked_attns_batch.append(attns_exp)
                masked_triplets_batch.append(triplets_token_ids)
                # assert attns.shape == triplets_token_ids.shape
                # masked_token_ids = attns * triplets_token_ids

        # elif self.strategy == "mask":
        #     tokenized_triplets = [self.llms.tokenizer(triplet_to_str(triplet), return_tensors="pt")
        #                           for (triplet, score) in zip(triplets, attns)]
        #     triplets_token_ids = torch.cat(
        #         [tokenized_triplet["input_ids"].squeeze() for tokenized_triplet in tokenized_triplets])

        #     # Get the lengths of the tokenized triplets (number of tokens in each triplet)
        #     triplet_lengths = [len(tokenized_triplet["input_ids"].squeeze()) for tokenized_triplet in tokenized_triplets]
        #     masks = torch.cat([attns[i].expand(triplet_lengths[i]) for i in range(len(attns))])

        #     mu = self.llms.tokenizer.encode(PLACEHOLDER, add_special_tokens=False)[0]  # Using the [MASK] token's ID
        #     # Create a placeholder tensor with the same length as concatenated token IDs
        #     placeholders = torch.full_like(triplets_token_ids, mu)
            # Apply the mask (element-wise multiplication)
            # masked_token_ids = masks * triplets_token_ids + (1 - masks) * placeholders

        else:
            raise NotImplementedError
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
