import torch
import torch.nn as nn
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import FORMER, LATTER, LABEL

IGNORE_INDEX = -100

class LLMs(nn.Module):
    def __init__(self, args):
        super().__init__()

        kwargs = {
            # "max_memory": {0: '80GiB', 1: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        self.ignore_idx = IGNORE_INDEX  # not as supervision signal

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False

        self.model = model

    def forward_pass(self, bm_triplet_ids, sample):
        """
        Calculate prediction loss given post-processed retrival contents.
        """

        batch_size =
        question =
        label =

        # TODO: batch-wise prompt. Now this is sample-wise
        former_pmt = FORMER
        latter_pmt = LATTER.format(question=question)
        label_pmt = LABEL.format(label=label)

        former_pmt_ids = self.tokenizer(former_pmt, add_special_tokens=False)
        latter_pmt_ids = self.tokenizer(latter_pmt, add_special_tokens=False)
        label_pmt_ids = self.tokenizer(label_pmt, add_special_tokens=False)

        batch_inputs_embeds, batch_attention_mask, batch_label_input_ids = [], [], []
        for i in range(batch_size):
            label_ids = label_pmt_ids.input_ids[i]
            # TODO: if we should keep `self.max_txt_len`
            input_ids = former_pmt_ids.input_ids[i] + bm_triplet_ids.input_ids[i][:self.max_txt_len] + latter_pmt_ids.input_ids[i] + label_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [self.ignore_idx] * (inputs_embeds.shape[0] - len(label_ids)) + label_ids
            batch_label_input_ids.append(label_input_ids)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [self.ignore_idx] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=label_input_ids,
        )
        return outputs.loss

    def inference(self, samples):
        pass

    def __loss__(self):
        pass
