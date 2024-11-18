from vllm import LLM, SamplingParams
import openai
from openai import OpenAI
from functools import partial
import torch.nn as nn
from src.utils.prompts import ICL_USER_PROMPT, ICL_ASS_PROMPT, SYS_PROMPT


class LLMs(nn.Module):
    def __init__(self, config):
        """
        I only finish llama-3.1-8B version. GPT-4o to be done.
        See https://github.com/Graph-COM/SubgraphRAG/blob/main/reason/llm_utils.py
        """
        super().__init__()
        self.prompt_mode = config["prompt_mode"]
        self.model_name = config["llm_model_name_or_path"]
        if "gpt" not in self.model_name:
            client = LLM(model=self.model_name, 
                         tensor_parallel_size=config["tensor_parallel_size"], 
                         max_seq_len_to_capture=config["max_seq_len_to_capture"])
            sampling_params = SamplingParams(temperature=config["temperature"], 
                                            max_tokens=config["max_tokens"],
                                            frequency_penalty=config["frequency_penalty"])
            self.llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
        else:
            # api_key = input("Enter OpenAI API key: ")
            # os.environ["OPENAI_API_KEY"] = api_key
            client = OpenAI()
            self.llm = partial(client.chat.completions.create, 
                               model=self.model_name, 
                               seed=config["seed"], 
                               temperature=config["temperature"], 
                               max_tokens=config["max_tokens"])
    
    def generate_prompt(self, query, input_triplets):
        """
        Generation conversation given a query-triplet pair.
        """
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
        question_prompt = "Question:\n" + query
        if question_prompt[-1] != '?':
            question_prompt += '?'
        user_query = "\n\n".join([triplet_prompt, question_prompt])
        
        conversation = []
        if 'sys' in self.prompt_mode:
            conversation.append({"role": "system", "content": SYS_PROMPT})

        if 'icl' in self.prompt_mode:
            conversation.append({"role": "user", "content": ICL_USER_PROMPT})
            conversation.append({"role": "assistant", "content": ICL_ASS_PROMPT})

        if 'sys' in self.prompt_mode:
            conversation.append({"role": "user", "content": user_query})
        return conversation                

    def forward(self, query_batch, triplet_batch):
        conversation_batch = [self.generate_prompt(q,t) for q, t in zip(query_batch, triplet_batch)]
        outputs = self.llm(messages=conversation_batch)
        generations = [output.outputs[0].text for output in outputs]
        return generations