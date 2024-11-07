# %%
import os
import re
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


# %%
def extract_reasoning_paths(text):
    pattern = r"Reasoning Paths:(.*?)\n\nQuestion:"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning_paths = match.group(1).strip()
        return reasoning_paths
    else:
        return None


# %%
input_file = os.path.join("ml1996", "webqsp")
test_set = load_dataset(input_file, split="test")
# test_set = load_dataset('rmanluo/RoG-webqsp', split='test')


# %%
def get_reversed_triplet(triplet):
    head, relation, tail = triplet
    # all_relations = relation.split('.')
    # reversed_relation = '.'.join([all_relations[-1], all_relations[1], all_relations[0]])
    reversed_relation = relation
    return (tail, reversed_relation, head)


def add_reversed_triplets(graph, add=True):
    res = []
    for each in graph:
        each = tuple(each)
        res.append(each)
        if add:
            try:
                res.append(get_reversed_triplet(each))
            except IndexError:
                # print(each)
                assert "#" in each[1]
    return res


# %%
pred_file_path = "../results/KGQA/RoG-webqsp/RoG/test/results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
# pred_file_path = "../results/KGQA/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
with open(pred_file_path, "r") as f:
    raw_data = [json.loads(line) for line in f]

# %%
data = []
for i, each_qa in enumerate(tqdm(raw_data)):
    assert each_qa["id"] == test_set[i]["id"]
    each_qa["graph"] = add_reversed_triplets(test_set[i]["graph"], add=False)
    data.append(each_qa)

# %%
total_good_triplets = 0
total_good_triplets_in_graph = 0
total_good_triplets_not_in_graph = 0
for idx, each_qa in enumerate(data):
    all_paths = extract_reasoning_paths(each_qa["input"]).split("\n")
    data[idx]["good_paths"] = all_paths
    all_good_triplets = []
    for each_path in all_paths:
        each_path = each_path.split(" -> ")
        good_triplets = []
        i = 0
        while i < len(each_path):
            if i + 2 < len(each_path):
                triplet = (each_path[i], each_path[i + 1], each_path[i + 2])
                temp_triplet = (each_path[i + 2], each_path[i + 1], each_path[i])
                total_good_triplets += 1
                if triplet in each_qa["graph"] or temp_triplet in each_qa["graph"]:
                    total_good_triplets_in_graph += 1
                else:
                    total_good_triplets_not_in_graph += 1
                good_triplets.append(triplet)
            i += 2
        all_good_triplets.extend(good_triplets)
    data[idx]["good_triplets"] = list(set(all_good_triplets))

print(
    f"Total good triplets: {total_good_triplets}, in graph: {total_good_triplets_in_graph}, not in graph: {total_good_triplets_not_in_graph}, ratio: {total_good_triplets_in_graph/total_good_triplets}"
)


# %%
def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"


# %%
np.random.seed(0)
num_trash = 50
processed_data = []
for idx, each_qa in enumerate(tqdm(data)):
    all_triplets = np.array(each_qa["graph"])
    good_triplets = np.array(each_qa["good_triplets"])
    sampled_trash = np.random.permutation(all_triplets)[:num_trash]
    input_triplets = np.concatenate([good_triplets, sampled_trash]) if len(good_triplets) > 0 else sampled_trash
    input_triplets = np.array([triplet_to_str(triplet) for triplet in input_triplets])
    input_triplets = np.random.permutation(input_triplets)

    # sys_prompt = """Based on the triplets from a knowledge graph, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\n\n"""
    # sys_prompt = """Based on the triplets from a knowledge graph, please answer the given question. Please keep the answer the question using the entity in the provided triplets and return all the possible answers as a list.\n\n"""
    # sys_prompt = "Please answer the question based on the provided triplets from a knowledge graph. " + \
    #              "You need to infer the answer using these triplets, which include all the evidence you need. " + \
    #              "Just return a list with all answers for each question without other words or explanations. " + \
    #              "For each answer, use the entity shown in the provided triplets.\n\n"
    # example_prompt = "For example, each triplet is formatted as (entity1, relation, entity2). " + \
    #                  "If you find entities, e.g., entity1, entity2, in the triplets answer the question, just return '- entity1\n- entity2'. Do not modify entitiy names.\n\n"
    # sys_prompt = sys_prompt + example_prompt

    # sys_prompt = "Please answer the question using only the provided triplets from a knowledge graph. " + \
    #              "The triplets contain all the necessary evidence for you to infer the answer. " + \
    #              "Return a list of answers for each question, without any additional words or explanations. " + \
    #              "Use the exact entity names as they appear in the provided triplets.\n" +\
    #              "Each triplet is formatted as (entity1, relation, entity2). " + \
    #              "If the answer can be found by identifying relevant entities (e.g., entity1 and entity2) within the triplets, " + \
    #              "return those entities as follows: `- entity1\n- entity2`. Otherwise, return `N/A`. " + \
    #              "Do not modify the entity names in any way in your answers.\n\n"

    # sys_prompt = [
    # "Please answer the question using the provided triplets from a knowledge graph. ",
    # "The triplets contain all the necessary evidence for you to infer the answer. ",
    # "Return a list of answers for each question, using ONLY the exact entity names as they appear in the provided triplets.\n",
    # "For example, each triplet is formatted as (entity1, relation, entity2); ",
    # "if the answer can be found by identifying relevant entities (e.g., entity1 and entity2) within the triplets, ",
    # "return those entities exactly as follows: `- entity1\n- entity2`. Otherwise, return `N/A`. ",
    # "Do NOT add any additional words, explanations and just start with the list of answers.\n\n"
    # ]

    # sys_prompt = [
    #     "Please answer the question using the provided triplets from a knowledge graph by ",
    #     "returning a list of answers for each question, using ONLY the exact entity names as they appear in the provided triplets.\n",
    #     # "Some questions may require you to reason over multiple triplets to find the answer.\n",
    #     "Each triplet is formatted as (entity1, relation, entity2); ",
    #     "if the answer can be found by identifying relevant entities (e.g., entity1 and entity2) within the triplets, ",
    #     "return those entities exactly as follows: `- ans: entity1\n- ans: entity2`. Otherwise, return `- ans: N/A` or answer the question with your own knowledge using the same format. ",
    #     "Do NOT add any additional descriptions or explanations in your answers. ",
    #     # "And please order the answers according to your confidence, with the most confident answers listed first. \n\n"
    # ]

    # sys_prompt = "".join(sys_prompt)
    sys_prompt = """Based on the triplets from a knowledge graph, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list, each with a prefix "ans:"."""

    triplet_prompt = "Triplets:\n" + "\n".join(input_triplets) + "\n\n"
    question_prompt = "Question:\n" + each_qa["question"] + "?"
    all_query = sys_prompt + "\n\n" + triplet_prompt + question_prompt

    user_query = triplet_prompt + question_prompt
    each_qa["sys_query"] = sys_prompt
    each_qa["user_query"] = user_query
    each_qa["all_query"] = all_query
    processed_data.append(each_qa)

# %%
print(processed_data[1]["all_query"])

# %%


# %%


# %%


# %%
import ollama
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-64k")

# %%
new_data = []
answers = []
tk_size = []
ctx = 8000
for idx, each_qa in enumerate(tqdm(processed_data)):
    query = each_qa["all_query"]
    tk_size.append(len(tokenizer.encode(query)))
    if tk_size[-1] > ctx:
        print(f"Warning: token size {tk_size[-1]} is larger than ctx {ctx}")

    answer = ollama.chat(
        model="llama3:8b-instruct-fp16",
        messages=[
            {"role": "system", "content": each_qa["sys_query"]},
            {"role": "user", "content": each_qa["user_query"]},
        ],
        options={"num_ctx": ctx},  # , "temperature": 0.0}
    )

    # answer = ollama.generate(model='llama3:8b-instruct-fp16', prompt=query, options= {"num_ctx": ctx, "temperature": 0.0})
    # answer = ollama.generate(model='llama2:7b-chat-fp16', prompt=query, options= {"num_ctx": ctx})
    # answer = ollama.generate(model='yarn-llama2:7b-64k-fp16', prompt=query, options= {"num_ctx": ctx})

    res = answer["response"] if "response" in answer else answer["message"]["content"]
    answers.append(res)
    del each_qa["prediction"]
    each_qa["prediction"] = res
    new_data.append(each_qa)

# %%
# llama_raw_pred_file_path = "../results/KGQA/RoG-webqsp/trash_test/llm3-8b-q8-good+trash100+randPerm-predictions.jsonl"
llama_raw_pred_file_path = "../results/KGQA/RoG-webqsp/trash_test/test_new_eval-sys-50trash-predictions.jsonl"
with open(llama_raw_pred_file_path, "w") as f:
    for each_q in new_data:
        f.write(json.dumps(each_q) + "\n")

# %%
from evaluate_results_corrected import eval_result

pred_file_path = llama_raw_pred_file_path
eval_result(pred_file_path, cal_f1=True)

# %%
