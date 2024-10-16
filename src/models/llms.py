import torch
import torch.nn as nn
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    def forward_pass(self, data):
        """
        Calculate prediction loss given post-processed retrival contents.
        """
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

    def prepare_textual_input(self):
        """
        prepare textual input (token ids) for LLM.
        Devided into three parts: system prompt, query, retrieval results.
        """
