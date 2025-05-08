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
# 215e0f164f9f445ea2aaa64db2e1c135
API_KEY = "26c54bfbe3dd4ce6b01089c362b7fe9a"
# 26c54bfbe3dd4ce6b01089c362b7fe9a
# "aca3450b38024f6786313b557f4b99dd"
# API_KEY="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"
MAX_RETRIES = 5

class LLMs(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.prompt_mode = config["prompt_mode"]
        self.model_name = config["llm_model_name_or_path"]
        self.data_name = config["data_name"]
        self.cot_mode = config["cot"]
        self.inf_level = config['level']
        self.async_version = kwargs.get("async_version", True)
        # prompt config
        print(self.model_name)
        self.cot_prompt = None
        # if self.inf_level == "path":
        #     self.system_prompt = SYS_PROMPT
        #     self.icl_prompt = [(ICL_USER_PROMPT, ICL_ASS_PROMPT)]
        # elif self.inf_level == "triplet":
        self.system_prompt = SYS_PROMPT_EVIDENCE
        self.icl_prompt = [(ICL_USER_PROMPT_triple_Apr, ICL_ASS_PROMPT_triple_Apr)]
        
        if self.model_name in ['Qwen/QwQ-32B', 'deepseek/deepseek-r1']:
            self.system_prompt = SYS_PROMPT_EVIDENCE_QWQ
            self.icl_prompt = [(ICL_USER_PROMPT_triple_QWQ, ICL_ASS_PROMPT_triple_QWQ)]     
            
        client_class = AsyncOpenAI if self.async_version else OpenAI
        
        client = client_class(
            base_url="https://api.aimlapi.com/v1/",
            api_key=API_KEY,
        )
        if self.async_version:
            self.semaphore = asyncio.Semaphore(20)
        print(config['seed'], config['temperature'])
        self.llm = partial(client.chat.completions.create, 
                            model=self.model_name, 
                            seed=config["seed"], 
                            temperature=config["temperature"], 
                            max_tokens=3000)
    
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
        # if not triplets_or_paths or type(triplets_or_paths[0]) is str:
            # triplet level prompt
        triplets_or_paths = [f'Chain {i+1}.\n' + triple for i, triple in enumerate(triplets_or_paths)]
        triplet_prompt = "Evidence Chains:\n" + "\n".join(triplets_or_paths)

        question_prompt = "Question:\n" + query
        if question_prompt[-1] != '?':
            question_prompt += '?'
        
        # hint_prompt = "Hints:\n" + "\n".join(hints)
        
        # user_query = "\n\n".join([triplet_prompt, question_prompt, hint_prompt])
        user_query = "\n\n".join([triplet_prompt, question_prompt])
        return self.pack_prompt(user_query)                

    def pack_prompt(self, user_query):
        conversation = []
        # if self.model_name in ['Qwen/QwQ-32B']:
        #     for item in self.icl_prompt:
        #         conversation.append({"role": "user", "content": item[0]})
        #         conversation.append({"role": "assistant", "content": item[1]})
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
        # if self.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        #     outputs = self.llm(messages=messages)
        #     # non-batch version
        #     return outputs[0].outputs[0].text
            # batch version
            # return [output.outputs[0].text for output in outputs]
        # else:
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
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.system_prompt = SYS_PROMPT_PATH_grailqa
        self.icl_prompt = [
                            # (ICL_USER_PROMPT_PATH_0, ICL_ASS_PROMPT_PATH_0),
                            (ICL_USER_PROMPT_PATH_grailqa, ICL_ASS_PROMPT_PATH_grailqa),
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
        
        response_batch = [self.llm_inf(messages=messages) for messages in messages_batch]

        for d, response, triplets_or_paths, q in zip(id_batch, response_batch, path_batch, query_batch):
            print_log(str(d), logging)
            print_log(q, logging)
            print_log(triplet_or_path_prompt, logging)
            print_log(response, logging)
            print_log('\n', logging)
            # score_list = get_score(response, len(triplets_or_paths))
            # print(score_list)
            # scores_batch.append(score_list)
            # identified, sign = get_pred(response), get_sign(response)
            identified = get_pred(response)
            try:
                identified = [triplets_or_paths[i] for i in identified]
            except IndexError as e:
                identified = []
            identified_batch.append(identified)
            # sign_batch.append(sign)
        # return scores_batch
        return identified_batch


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