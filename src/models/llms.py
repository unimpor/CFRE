import random
from openai import AsyncOpenAI
import asyncio
import re
import time
from vllm import LLM, SamplingParams
import openai
from openai import OpenAI
from functools import partial
import torch.nn as nn
from src.utils.prompts import *
from src.utils import print_log, triplet_to_str

API_KEY = "215e0f164f9f445ea2aaa64db2e1c135"
# API_KEY="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"
MAX_RETRIES = 5

class LLMs(nn.Module):
    def __init__(self, config):
        """
        See https://github.com/Graph-COM/SubgraphRAG/blob/main/reason/llm_utils.py
        """
        super().__init__()
        self.prompt_mode = config["prompt_mode"]
        self.model_name = config["llm_model_name_or_path"]
        self.data_name = config["data_name"]
        self.cot_mode = config["cot"]
        self.fast_thinking = config["fast_thinking"]
        # prompt config
        print(config["frequency_penalty"])
        self.cot_prompt = None
        if not self.fast_thinking:
            self.system_prompt = SYS_PROMPT
            self.icl_prompt = [
                            (ICL_USER_PROMPT, ICL_ASS_PROMPT), 
                            # (ICL_USER_PROMPT_2, ICL_ASS_PROMPT_2),
                            # (ICL_USER_PROMPT_3, ICL_ASS_PROMPT_3)
                            ]
            # self.icl_prompt = []
        # else:
        #     self.system_prompt = SYS_PROMPT_brief
        #     self.icl_prompt = [(ICL_USER_PROMPT, ICL_ASS_PROMPT_brief), 
        #                     (ICL_USER_PROMPT_2, ICL_ASS_PROMPT_2_brief),
        #                     (ICL_USER_PROMPT_3, ICL_ASS_PROMPT_3_brief)
        #                     ]
            # self.system_prompt = SYS_PROMPT_brief_path_level_inf
            # self.icl_prompt = [(ICL_USER_PROMPT_path_level_inf, ICL_ASS_PROMPT_brief_path_level_inf)]
        if self.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
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
            client = AsyncOpenAI(
                base_url="https://api.aimlapi.com/v1/",
                api_key=API_KEY,  
            )
            self.semaphore = asyncio.Semaphore(20)
            print(config['seed'], config['temperature'])
            self.llm = partial(client.chat.completions.create, 
                               model=self.model_name, 
                               seed=config["seed"], 
                               temperature=config["temperature"], 
                               max_tokens=1000)
    
    def initialize_prompt_template(self, ):
        if not self.fast_thinking:
            self.system_prompt = SYS_PROMPT
            self.icl_prompt = [(ICL_USER_PROMPT, ICL_ASS_PROMPT), 
                            (ICL_USER_PROMPT_2, ICL_ASS_PROMPT_2),
                            (ICL_USER_PROMPT_3, ICL_ASS_PROMPT_3)
                            ]
        else:
            self.system_prompt = SYS_PROMPT_brief
            self.icl_prompt = [(ICL_USER_PROMPT, ICL_ASS_PROMPT_brief), 
                            (ICL_USER_PROMPT_2, ICL_ASS_PROMPT_2_brief),
                            (ICL_USER_PROMPT_3, ICL_ASS_PROMPT_3_brief)
                            ]
    
    def prompt_update(self, ):
        self.cot_prompt = COT_PROMPT

    def generate_prompt(self, query, hints, triplets_or_paths):
        """
        Generation conversation given a query-triplet pair.
        """
        if type(triplets_or_paths[0]) is str:
            # triplet level prompt
            triplet_prompt = "Triplets:\n" + "\n".join(triplets_or_paths)
        elif type(triplets_or_paths[0]) is list:
            # path level prompt
            formatted_paths = []
            for i, path in enumerate(triplets_or_paths):
                if len(path) == 1:
                    break
                formatted_path = f"Path {i}. "
                formatted_path += ', '.join([triplet_to_str(triplet) for triplet in path])
                formatted_paths.append(formatted_path)
            triplets_or_paths = [triplet_to_str(item[0]) for item in triplets_or_paths if len(item) == 1]
            if len(triplets_or_paths) > 0:
                triplet_prompt = "Paths:\n" + "\n".join(formatted_paths) + '\n' + "Scattered Triplets:\n" + "\n".join(triplets_or_paths)
            else:
                triplet_prompt = "Paths:\n" + "\n".join(formatted_paths)
        
        question_prompt = "Question:\n" + query
        if question_prompt[-1] != '?':
            question_prompt += '?'
        
        # hint_prompt = "Hints:\n" + "\n".join(hints)
        
        # user_query = "\n\n".join([triplet_prompt, question_prompt, hint_prompt])
        user_query = "\n\n".join([triplet_prompt, question_prompt])
        return self.pack_prompt(user_query)                

    def pack_prompt(self, user_query):
        conversation = []
        # if self.model_name == "deepseek/deepseek-r1":
        #     conversation.append({"role": "user", "content": self.system_prompt + '\n\n' + user_query})
        #     return conversation
        
        if 'sys' in self.prompt_mode:
            conversation.append({"role": "system", "content": self.system_prompt})

        if 'icl' in self.prompt_mode:
            for item in self.icl_prompt:
                conversation.append({"role": "user", "content": item[0]})
                conversation.append({"role": "assistant", "content": item[1]})

        if 'sys' in self.prompt_mode:
            conversation.append({"role": "user", "content": user_query})
        if self.cot_prompt:
            conversation.append({"role": "user", "content": COT_PROMPT})
        return conversation

    def llm_inf(self, messages):
        if self.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            outputs = self.llm(messages=messages)
            # non-batch version
            return outputs[0].outputs[0].text
            # batch version
            # return [output.outputs[0].text for output in outputs]
        else:
            # inference with retry
            retries, max_retries = 0, MAX_RETRIES
            while retries < max_retries:
                try:
                    output = self.llm(messages=messages)
                    return output.choices[0].message.content
                except openai.BadRequestError as e:
                    print(e)
                    break
                except openai.RateLimitError as e:
                    wait_time = (2 ** retries) * 5  # Exponential backoff
                    print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                except:
                    break
            # raise Exception("Max retries exceeded. Please check your rate limits or try again later.")
            print("Max retries exceeded. Please check your rate limits or try again later.")
            return "ans: Not available"

    def forward(self, query_batch, hint_batch, triplet_or_path_batch):
        
        # if self.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        #     conversation_batch = [self.generate_prompt(q,a,t) for q, a, t in zip(query_batch, hint_batch, triplet_or_path_batch)]
        #     return self.llm_inf(conversation_batch)
        
        outputs = [self.llm_inf(messages=self.generate_prompt(q,a,t)) for q, a, t in zip(query_batch, hint_batch, triplet_or_path_batch)]
        # if "gpt" not in self.model_name:
        #     generations = [output[0].outputs[0].text for output in outputs]
        # else:
        #     generations = [output.choices[0].message.content for output in outputs]
        return outputs

    async def forward_pass(self, query_batch, hint_batch, triplet_or_path_batch):
        q,a,t = query_batch[0], hint_batch[0], triplet_or_path_batch[0]
        return await self.llm_inf_asy(self.generate_prompt(q,a,t))

    async def llm_inf_asy(self, messages):
        retries = 0
        while retries < MAX_RETRIES:
            try:
                async with self.semaphore:
                    output = await self.llm(messages=messages)
                return output.choices[0].message.content
            except openai.BadRequestError as e:
                print(e)
                break
            except openai.RateLimitError:
                wait_time = (2 ** retries) * 5
                print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                retries += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        print("Max retries exceeded. Please check your rate limits or try again later.")
        return "ans: Not available"


class LLMs_Ret_Paths(LLMs):
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = SYS_PROMPT_PATH
        self.icl_prompt = []
        if self.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            self.icl_prompt = [(ICL_USER_PROMPT_PATH_0, ICL_ASS_PROMPT_PATH_0),
                               (ICL_USER_PROMPT_PATH_1, ICL_ASS_PROMPT_PATH_1),
                               (ICL_USER_PROMPT_PATH_2, ICL_ASS_PROMPT_PATH_2),
                               (ICL_USER_PROMPT_PATH_3, ICL_ASS_PROMPT_PATH_3),
                               ]

        self.level = "path"
    
    def oracle_detection(self, id_batch, query_batch, answer_batch, path_batch, **kwargs):
        """
        Note: not a batch-wise version.
        """
        messages_batch = []
        identified_batch, sign_batch, scores_batch = [], [], []
        for query, answer, triplets_or_paths in zip(query_batch, answer_batch, path_batch):
            logging = kwargs.get("logging", None)
            formatted_paths = []
            for i, path in enumerate(triplets_or_paths):
                formatted_path = f"Path {i}.\n"
                # Join each triple as a string and separate them by commas
                formatted_path += ', '.join([str(triplet) for triplet in path])
                formatted_paths.append(formatted_path)

            triplet_or_path_prompt = "Paths:\n" + "\n".join(formatted_paths)
            
            question_prompt = "Question:\n" + query
            answer_prompt = "Answer(s):\n" + ", ".join(answer)
            if question_prompt[-1] != '?':
                question_prompt += '?'

            user_query = "\n\n".join([triplet_or_path_prompt, question_prompt, answer_prompt])
            messages_batch.append(self.pack_prompt(user_query))
        
        if self.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            response_batch = self.llm_inf(messages=messages_batch)
        else:
            response_batch = [self.llm_inf(messages=messages) for messages in messages_batch]

        for d, response, triplets_or_paths in zip(id_batch, response_batch, path_batch):
            print_log(d, logging)
            print_log(response, logging)
            print_log('\n', logging)
            score_list = get_score(response, len(triplets_or_paths))
            # print(score_list)
            scores_batch.append(score_list)
        #     identified, sign = get_pred(response), get_sign(response)
        #     try:
        #         identified = [triplets_or_paths[i] for i in identified]
        #     except IndexError as e:
        #         identified = []
        #     identified_batch.append(identified)
        #     sign_batch.append(sign)
        return scores_batch
        # return identified_batch, sign_batch
        # selected_triplets = [triplet for path in identified for triplet in path]
        # selected_triplets = [all_triplets.index(triplet) for triplet in selected_triplets]
        # return selected_triplets

        
class LLMs_Ret_Triplets(LLMs):
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = SYS_PROMPT_TRI
        self.icl_prompt = (ICL_USER_PROMPT_TRI, ICL_ASS_PROMPT_TRI)
        self.level = "triplet"
    
    def oracle_detection(self, query, answer, triplets_or_paths, all_triplets):
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
        print(identified)
        try:
            identified = [all_triplets_[i] for i in identified]
        except IndexError as e:
            identified = []
        print(identified)
        selected_triplets = [all_triplets.index(triplet) for triplet in identified]
        return selected_triplets


class LLMs_Ret_Relations(LLMs):
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = SYS_PROMPT_R
        self.icl_prompt = (ICL_USER_PROMPT_R, ICL_ASS_PROMPT_R)
        if self.data_name == "webqsp":
            self.icl_prompt = (ICL_USER_PROMPT_R_W, ICL_ASS_PROMPT_R_W)
        self.level = "relation"
    
    def oracle_detection(self, query, answer, triplets_or_paths, all_triplets):
        # all_relations = path2relation(triplets_or_paths)
        all_relations = triplets_or_paths
        formatted_relations = [f"Relation{i}.\n" + r for i, r in enumerate(all_relations)]
        
        triplet_or_path_prompt = "Relations:\n" + "\n".join(formatted_relations)
        
        question_prompt = "Question:\n" + query
        if question_prompt[-1] != '?':
            question_prompt += '?'

        user_query = "\n\n".join([triplet_or_path_prompt, question_prompt])
        response = self.llm_inf(messages=self.pack_prompt(user_query))

        print_log(user_query + "\n" + response, save_path=f"fig/{self.data_name}_iter2_label_{self.level}_.txt")

        identified = get_pred(response)
        try:
            identified = [all_relations[i] for i in identified]
        except IndexError as e:
            identified = []

        # selected_triplets = [triplet for triplet in path2triple(triplets_or_paths) if triplet[1] in identified]
        # selected_triplets = [all_triplets.index(triplet) for triplet in selected_triplets]
        # return selected_triplets 
        return identified     


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def get_sign(prediction):
    try:
        return prediction.split("\n")[-1].split(": ")[-1]
    except:
        return "CONTINUE"

def get_score(response, num):
    path_scores, scores_list = {}, []

    lines = response.split("\n")
    for line in lines:
        if line.startswith("Path"):
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    path_num = int(parts[0].split()[1])
                    path_value = int(parts[1].strip())
                    path_scores[path_num] = path_value
                except:
                    print("Some format errors. Double check this sample.")
                    pass

    for i in range(num):
        scores_list.append(path_scores.get(i, -2))
    
    return scores_list

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