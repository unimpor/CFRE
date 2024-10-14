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


        # 4. LLM supervisions
        outputs = self.llms.forward_masked()
        loss = attn__loss + outputs.loss
        loss_dict["predict"] = outputs.loss.item()
        return loss, loss_dict

    def mask_triplet(self, tokenized_triplet, attn):
        # Example: triplets converted into strings


        # Mask from Gumbel-Softmax (N-dimensional)
        m = torch.tensor([0.8, 0.1, 0.9], requires_grad=True)  # Example mask from Gumbel-Softmax

        # Create the full mask via broadcasting instead of repeat
        expanded_masks = torch.cat([m[i].expand(triplet_lengths[i]) for i in range(len(m))])

        # Concatenate triplet strings into a single sequence (you can tokenize or vectorize these)
        T = "".join(triplet_strings)  # Concatenated triplet strings as a single sequence

        # Placeholder string for masked-out triplets
        mu = "[MASK]" * max(triplet_lengths)  # Placeholder string matching the longest triplet

        # Apply the mask (element-wise multiplication)
        # Use a tensor representation of the concatenated triplet strings for element-wise ops
        # Assuming triplet strings are converted to tensors (e.g., embeddings or token IDs)
        masked_triplet_text = expanded_masks * torch.tensor(list(T), dtype=torch.float32) \
                              + (1 - expanded_masks) * torch.tensor(list(mu), dtype=torch.float32)

        # Example backward pass (just for checking gradient flow)
        loss = masked_triplet_text.sum()  # Dummy loss for gradient check
        loss.backward()

        # Check gradients for mask values
        print(m.grad)  # Should not be None, confirming gradient flow
