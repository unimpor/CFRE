import torch
import torch.nn as nn


class CFRE(nn.Module):
    def __init__(self, ibtn_module, llm_model, args, **kwargs):
        super().__init__()
        self.ibtn = ibtn_module
        self.llms = llm_model

    @property
    def device(self):
        return list(self.parameters())[0].device

    def __loss__(self, att):
        """
        Calculate attn-related loss
        # TODO: try different loss components: info/dir/con
        """

        # input: attn_score
        def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
            r_ = init_r - current_epoch // decay_interval * decay_r
            return final_r if r_ < final_r else r_

        r = self.fix_r if self.fix_r else get_r(self.decay_interval, self.decay_r, epoch, self.init_r, self.final_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

        con_loss = None

        dir_loss = None

        return info_loss + con_loss + dir_loss, {"info": info_loss.item(), "con": con_loss.item(), "dir": dir_loss.item()}

    def forward_pass(self, data):
        # 1. post-extract data items

        # 2. generate mask for the coarsely retrieved samples.
        attn_bern = self.ibtn()
        attn__loss, loss_dict = self.__loss__()  # calculate attn-related loss

        # 3. generate filtered retrieval results
        masked_triple_token_id = self.mask_triplet(triplets, attn_bern)

        # 4. LLM supervisions
        outputs = self.llms.forward_pass()
        loss = attn__loss + outputs.loss
        loss_dict["predict"] = outputs.loss.item()
        return loss, loss_dict

    def mask_triplet(self, triplets, attn):
        """
        Mask tokenized triplets using attn
        """
        # Example: triplets converted into strings

        # Load the tokenizer
        def triplet_to_str(triplet):
            return f"({triplet[0]},{triplet[1]},{triplet[2]})"
        # Tokenize each triplet string
        tokenized_triplets = [self.llms.tokenizer(triplet_to_str(triplet), return_tensors="pt")
                              for triplet in triplets]

        # Get the lengths of the tokenized triplets (number of tokens in each triplet)
        triplet_lengths = [len(tokenized_triplet["input_ids"].squeeze()) for tokenized_triplet in tokenized_triplets]

        masks = torch.cat([attn[i].expand(triplet_lengths[i]) for i in range(len(attn))])

        # Concatenate all tokenized triplets into a single tensor sequence
        concatenated_token_ids = torch.cat(
            [tokenized_triplet["input_ids"].squeeze() for tokenized_triplet in tokenized_triplets])

        mu = self.llms.tokenizer.encode("\n", add_special_tokens=False)[0]  # Using the [MASK] token's ID

        # Create a placeholder tensor with the same length as concatenated token IDs, filled with the [MASK] token ID
        placeholder_tensor = torch.full_like(concatenated_token_ids, mu)
        # Apply the mask (element-wise multiplication)
        masked_token_ids = masks * concatenated_token_ids + (1 - masks) * placeholder_tensor
        return masked_token_ids

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
