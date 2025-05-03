import torch
import random
import numpy as np
import networkx as nx
import math
from torch_geometric.data import Batch


def write_log(print_str, log_file):
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_chunk_set(s, size):
    final = []
    s = list(s) if isinstance(s, set) else [s]
    for i in range(0, len(s), size):
        res_str = "{"
        targets = s[i:i+size]
        for i, itm in enumerate(targets):
            res_str += f"({i+1}) {itm}"
            if i < len(targets)-1:
                res_str += " "
        final.append(res_str + "}")
    return final
    # return [" | ".join(set(s[i:i+size])) for i in range(0, len(s), size)]

def adjust_learning_rate(param_group, LR, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr

def print_log(content, save_path = "fig/cwq_pure_weak_.txt"):
    # print(content)
    # replace("\n", "")
    content = content if isinstance(content, str) else "\n".join(map(str, content))
    
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")

def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
# def collate_fn(data):
#     sample = data[0]
#
#     h_id_list = sample['h_id_list']
#     h_id_tensor = torch.tensor(h_id_list)
#
#     r_id_list = sample['r_id_list']
#     r_id_tensor = torch.tensor(r_id_list)
#
#     t_id_list = sample['t_id_list']
#     t_id_tensor = torch.tensor(t_id_list)
#
#     num_non_text_entities = len(sample['non_text_entity_list'])
#
#     return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'], \
#            sample['entity_embs'], num_non_text_entities, sample['relation_embs'], \
#            sample['topic_entity_one_hot'], \
#            sample['a_entity_id_list']


# def collate_fn(data):
#     sample = data[0]
#     return sample['edge_index'], sample['entity_embd'], sample["y"], sample['edge_attr'], \
#         sample['triplets'], sample['relevant_idx'], \
#         sample['q'], sample['q_embd']

def reorder_(lst):
    return lst[::2] + lst[1::2][::-1]

def collate_fn(batch_org):
    batch = {}

    for k in batch_org[0].keys():
        batch[k] = [d[k] for d in batch_org]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    
    batch_size = len(batch_org)
    triplets_num_per_graph = [len(d["triplets"]) for d in batch_org]
    batch["q_embd"] = torch.cat([batch["q_embd"][i].expand(triplets_num_per_graph[i], -1) for i in range(batch_size)])    
    batch['triplet_batch_idx'] = torch.cat([torch.tensor([i]).expand(triplets_num_per_graph[i]) for i in range(batch_size)])

    # all_rel_idx = []
    # current_node_count = 0
    # for rel_idx, num_tr in zip(batch['relevant_idx'], triplets_num_per_graph):
    #     all_rel_idx.append(torch.tensor(rel_idx) + current_node_count)
    #     current_node_count += num_tr
    # batch['relevant_idx'] = torch.cat(all_rel_idx)
    # combined_list = sum(batch['triplets'], [])
    # a = [combined_list[idx.item()] for idx in batch['relevant_idx']]
    # b = sum([[d['triplets'][idx] for idx in d['relevant_idx']] for d in batch_org], [])
    return batch