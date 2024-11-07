# %%
import os
import re
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# %%
os.environ["OPENAI_API_KEY"] = ""


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

    Start_prompt = """Based on the triplets from a knowledge graph, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\n\n"""
    triplet_prompt = "Triplets:\n" + "\n".join(input_triplets) + "\n\n"
    end_prompt = "Question:\n" + each_qa["question"] + "?"
    query = Start_prompt + triplet_prompt + end_prompt
    each_qa["query"] = query
    processed_data.append(each_qa)

# %%
from openai import OpenAI

client = OpenAI()

# %%
new_data = []
answers = []
for idx, each_qa in enumerate(tqdm(processed_data)):
    query = each_qa["query"]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        # messages=[
        #     {
        #         "role": "user",
        #         "content": query
        #     }
        # ]
        messages=[
            {"role": "system", "content": query.split("\n")[0]},
            {"role": "user", "content": "\n".join(query.split("\n")[2:])},
        ],
    )
    res = completion.choices[0].message.content
    answers.append(res)
    del each_qa["prediction"]
    each_qa["prediction"] = res
    new_data.append(each_qa)
    # if idx == 10:
    #     break


# %%
# llama_raw_pred_file_path = "../results/KGQA/RoG-webqsp/trash_test/llm3-8b-q8-good+trash100+randPerm-predictions.jsonl"
llama_raw_pred_file_path = "../results/KGQA/RoG-webqsp/trash_test/gpt4o-mini+trash50+randPerm-predictions.jsonl"
with open(llama_raw_pred_file_path, "w") as f:
    for each_q in new_data:
        f.write(json.dumps(each_q) + "\n")

# %%
from evaluate_results import eval_result

pred_file_path = llama_raw_pred_file_path
eval_result(pred_file_path, cal_f1=True)

# %%
