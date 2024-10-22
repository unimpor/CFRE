import torch
import random
import numpy as np
import networkx as nx


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_fn(data):
    sample = data[0]

    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)

    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)

    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)

    num_non_text_entities = len(sample['non_text_entity_list'])
    # TODO； sample['topic_entity_one_hot'], sample['target_triple_probs']
    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'], \
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'], \
        sample['topic_entity_one_hot'], \
        sample['a_entity_id_list']
