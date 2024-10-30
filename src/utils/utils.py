import torch
import random
import numpy as np
import networkx as nx
import math


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def collate_fn(data):
    sample = data[0]
    return sample['edge_index'], sample['entity_embd'], sample["y"], sample['edge_attr'], \
        sample['triplets'], sample['relevant_idx'], \
        sample['q'], sample['q_embd']
