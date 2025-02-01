import re
import time
from vllm import LLM, SamplingParams
import openai
from openai import OpenAI
from functools import partial
import torch.nn as nn
from src.utils.prompts import *
from src.utils import print_log

API_KEY = "215e0f164f9f445ea2aaa64db2e1c135"
MAX_RETRIES = 5

class LLMs(nn.Module):
    def __init__(self, config):
        """
        See https://github.com/Graph-COM/SubgraphRAG/blob/main/reason/llm_utils.py
        """
        super().__init__()
        self.prompt_mode = config["prompt_mode"]
        self.model_name = config["llm_model_name_or_path"]
        # prompt config
        self.system_prompt = SYS_PROMPT
        self.icl_prompt = (ICL_USER_PROMPT, ICL_ASS_PROMPT)
        if "gpt" not in self.model_name:
            client = LLM(model=self.model_name, 
                         tensor_parallel_size=config["tensor_parallel_size"], 
                         max_seq_len_to_capture=config["max_seq_len_to_capture"],
                         gpu_memory_utilization=config["gpu_memory_utilization"],
                         )
            sampling_params = SamplingParams(temperature=config["temperature"], 
                                            max_tokens=config["max_tokens"],
                                            frequency_penalty=config["frequency_penalty"])
            self.llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
        else:
            client = OpenAI(
                base_url="https://api.aimlapi.com/v1/",
                api_key=API_KEY,  
            )
            self.llm = partial(client.chat.completions.create, 
                               model=self.model_name, 
                               seed=config["seed"], 
                               temperature=config["temperature"], 
                               max_tokens=1000)
    
    def generate_prompt(self, query, answer, triplets_or_paths):
        """
        Generation conversation given a query-triplet pair.
        """
        if "reverse" in self.prompt_mode:
            triplets_or_paths.reverse()
        triplet_prompt = "Triplets:\n" + "\n".join(triplets_or_paths)
        question_prompt = "Question:\n" + query
        if question_prompt[-1] != '?':
            question_prompt += '?'
        user_query = "\n\n".join([triplet_prompt, question_prompt])
 
        return self.pack_prompt(user_query)                

    def pack_prompt(self, user_query):
        conversation = []
        if 'sys' in self.prompt_mode:
            conversation.append({"role": "system", "content": self.system_prompt})

        if 'icl' in self.prompt_mode:
            conversation.append({"role": "user", "content": self.icl_prompt[0]})
            conversation.append({"role": "assistant", "content": self.icl_prompt[1]})

        if 'sys' in self.prompt_mode:
            conversation.append({"role": "user", "content": user_query})
        return conversation

    def llm_inf(self, messages):
        if "gpt" not in self.model_name:
            output = self.llm(messages=messages)
            return output[0].outputs[0].text
        else:
            # inference with retry
            retries, max_retries = 0, MAX_RETRIES
            while retries < max_retries:
                try:
                    output = self.llm(messages=messages)
                    return output.choices[0].message.content
                except openai.RateLimitError as e:
                    wait_time = (2 ** retries) * 5  # Exponential backoff
                    print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
            # raise Exception("Max retries exceeded. Please check your rate limits or try again later.")
            print("Max retries exceeded. Please check your rate limits or try again later.")
            return "ans: Not available"

    def forward(self, query_batch, ans_batch, triplet_or_path_batch):
        outputs = [self.llm_inf(messages=self.generate_prompt(q,a,t)) for q, a, t in zip(query_batch, ans_batch, triplet_or_path_batch)]
        # if "gpt" not in self.model_name:
        #     generations = [output[0].outputs[0].text for output in outputs]
        # else:
        #     generations = [output.choices[0].message.content for output in outputs]
        return outputs

    def bforward(self, query_batch, ans_batch, triplet_batch):
        # batch generation
        conversation_batch = [self.generate_prompt(q,a,t) for q, a, t in zip(query_batch, ans_batch, triplet_batch)]
        outputs = self.llm(messages=conversation_batch)
        generations = [output.outputs[0].text for output in outputs]
        return generations

class LLMs_Ret_Paths(LLMs):
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = SYS_PROMPT_PATH
        self.icl_prompt = (ICL_USER_PROMPT_PATH, ICL_ASS_PROMPT_PATH)
        self.level = "path"
    
    def identify_(self, query, answer, triplets_or_paths, all_triplets):
        """
        Note: not a batch-wise version.
        """
        formatted_paths = []
        for i, path in enumerate(triplets_or_paths):
            formatted_path = f"Path{i}.\n"
            # Join each triple as a string and separate them by commas
            formatted_path += ', '.join([str(triplet) for triplet in path])
            formatted_paths.append(formatted_path)

        triplet_or_path_prompt = "Paths:\n" + "\n".join(formatted_paths)
        
        question_prompt = "Question:\n" + query
        answer_prompt = "Answer(s):\n" + ", ".join(answer)
        if question_prompt[-1] != '?':
            question_prompt += '?'

        user_query = "\n\n".join([triplet_or_path_prompt, question_prompt, answer_prompt])
        response = self.llm_inf(messages=self.pack_prompt(user_query))

        print_log(user_query + "\n" + response, save_path=f"fig/webqsp_label_{self.level}.txt")
        identified = get_pred(response)
        try:
            identified = [triplets_or_paths[i] for i in identified]
        except IndexError as e:
            identified = []

        selected_triplets = [triplet for path in identified for triplet in path]
        selected_triplets = [all_triplets.index(triplet) for triplet in selected_triplets]
        
        return selected_triplets

        
class LLMs_Ret_Triplets(LLMs):
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = SYS_PROMPT_TRI
        self.icl_prompt = (ICL_USER_PROMPT_TRI, ICL_ASS_PROMPT_TRI)
        self.level = "triplet"
    
    def identify_(self, query, answer, triplets_or_paths, all_triplets):
        all_triplets_ = path2triple(triplets_or_paths)
        formatted_triplets = [f"Triplet{i}.\n" + str(triplet) for i, triplet in enumerate(all_triplets_)]
        
        triplet_prompt = "Triplets:\n" + "\n".join(formatted_triplets)
        question_prompt = "Question:\n" + query
        answer_prompt = "Answer(s):\n" + ", ".join(answer)
        if question_prompt[-1] != '?':
            question_prompt += '?'

        user_query = "\n\n".join([triplet_prompt, question_prompt, answer_prompt])
        response = self.llm_inf(messages=self.pack_prompt(user_query))

        print_log(user_query + "\n" + response, save_path=f"fig/webqsp_label_{self.level}.txt")
        identified = get_pred(response)
        try:
            identified = [all_triplets_[i] for i in identified]
        except IndexError as e:
            identified = []

        selected_triplets = [all_triplets.index(triplet) for triplet in identified]
        
        return selected_triplets


class LLMs_Ret_Relations(LLMs):
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = SYS_PROMPT_R
        self.icl_prompt = (ICL_USER_PROMPT_R, ICL_ASS_PROMPT_R)
        self.level = "relation"
    
    def identify_(self, query, answer, triplets_or_paths, all_triplets):
        all_relations = path2relation(triplets_or_paths)
        formatted_relations = [f"Relation{i}.\n" + r for i, r in enumerate(all_relations)]
        
        triplet_or_path_prompt = "Relations:\n" + "\n".join(formatted_relations)
        
        question_prompt = "Question:\n" + query
        if question_prompt[-1] != '?':
            question_prompt += '?'

        user_query = "\n\n".join([triplet_or_path_prompt, question_prompt])
        response = self.llm_inf(messages=self.pack_prompt(user_query))

        print_log(user_query + "\n" + response, save_path=f"fig/{self.data_name}_iter2_label_{self.level}.txt")

        identified = get_pred(response)
        try:
            identified = [all_relations[i] for i in identified]
        except IndexError as e:
            identified = []

        selected_triplets = [triplet for triplet in path2triple(triplets_or_paths) if triplet[1] in identified]
        selected_triplets = [all_triplets.index(triplet) for triplet in selected_triplets]
        
        return selected_triplets      


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def get_pred(prediction):
    ans_list = []
    ans_list_ = [p for p in prediction.split("\n") if 'ans:' in p]
    for item in ans_list_:
        try:
            ans_list.append(int(re.findall(r'\d+', item[5:].replace(" ", ""))[0]))
        except:
            continue
    return remove_duplicates(ans_list)

def path2relation(paths):
    all_relation = []
    for path in paths:
        for triplet in path:
            if triplet[1] not in all_relation:
                all_relation.append(triplet[1])
    return all_relation

def path2triple(paths):
    all_triplet = []
    for path in paths:
        for triplet in path:
            if triplet not in all_triplet:
                all_triplet.append(triplet)
    return all_triplet