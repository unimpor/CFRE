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
    def __init__(self, config):
        super().__init__()

        kwargs = {
            # "max_memory": {0: '80GiB'},
            "device_map": config['device'],
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(config['llm_model_path'], use_fast=False,
                                                       revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        self.ignore_idx = IGNORE_INDEX  # not as supervision signal

        # Note torch.dtype
        model = AutoModelForCausalLM.from_pretrained(
            config['llm_model_path'],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if config['llm_frozen'] is True:
            print("Freezing LLAMA!")
            for _, param in model.named_parameters():
                param.requires_grad = False

        self.model = model
        self.word_embedding = self.model.model.get_input_embeddings()

    def forward_pass(self, attns, bm_triplet_ids, question_batch, label_batch, training=True):
        """
        Calculate prediction loss given post-processed retrival contents.
        Used in the training and eval process, not in inference.
        """

        # TODO: batch-wise prompt. Now this is sample-wise
        label_batch = ["[" + ", ".join(label) + "]" for label in label_batch]
        batch_size = len(bm_triplet_ids)
        former_pmt = FORMER
        latter_pmt = [LATTER.format(question=question) for question in question_batch]
        label_pmt = [LABEL.format(label=label) for label in label_batch]

        former_pmt_ids = self.tokenizer(former_pmt, add_special_tokens=False)["input_ids"]
        latter_pmt_ids = self.tokenizer(latter_pmt, add_special_tokens=False)["input_ids"]
        label_pmt_ids = self.tokenizer(label_pmt, add_special_tokens=False)["input_ids"]

        batch_inputs_embeds, batch_attention_mask, batch_label_input_ids = [], [], []
        for i in range(batch_size):
            # label_ids = label_pmt_ids["input_ids"]
            # TODO: if we should keep `self.max_txt_len`
            # input_ids = former_pmt_ids["input_ids"][i] + bm_triplet_ids["input_ids"][i][:self.max_txt_len] + \
                        # latter_pmt_ids["input_ids"][i] + label_ids
            # input_ids = former_pmt_ids["input_ids"] + bm_triplet_ids.tolist() + latter_pmt_ids["input_ids"] + label_ids
            # inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            former_pmt_embeds = self.word_embedding(torch.tensor(former_pmt_ids).to(self.model.device))

            bm_triplet_embeds = attns[i] * self.word_embedding(bm_triplet_ids[i].to(self.model.device)) if training else \
                                self.word_embedding(bm_triplet_ids[i].to(self.model.device))

            latter_pmt_embeds = self.word_embedding(torch.tensor(latter_pmt_ids[i] + label_pmt_ids[i]).to(self.model.device))
            # TODO: combine with attns
            inputs_embeds = torch.concat([former_pmt_embeds, 
                                          bm_triplet_embeds, 
                                          latter_pmt_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [self.ignore_idx] * (inputs_embeds.shape[0] - len(label_pmt_ids[i])) + label_pmt_ids[i]
            batch_label_input_ids.append(label_input_ids)
   
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
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

    def inference(self, ):
        # For inference of the project, follow SubgraphRAG first.
        raise NotImplementedError

    def __loss__(self):
        pass
