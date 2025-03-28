import os
from os.path import join as opj
import pickle
import torch.nn.functional as F
import torch
import csv
from itertools import chain
from torch_geometric.data.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from src.utils import print_log, remove_duplicates, match, NOISES, TO_FILTER_REL
from copy import deepcopy
import time


class RetrievalDataset:
    """
    load retrieval results and post-processing
    """

    def __init__(self, config, split, **kwargs):
        self.split = split
        self.config = config
        self.root = config['root']
        self.data_name = config['name']

        self.short_path = kwargs.get('spath', opj(self.root, self.data_name, "processed", "shortest_path.pth"))
        self.oracle_path = kwargs.get("opath", opj(self.root, self.data_name, "processed", "refined_path.pth"))

        self.training = kwargs.get("training", True)
        self.domains = kwargs.get('domains', None)  # only applicable to grailqa dataset.
        
        raw_data = self._load_data(opj(self.root, self.data_name, "processed", f"{self.split}.pkl"))
        self.processed_data = self.process(raw_data, )

    def process(self, raw_data, ):

        embs = self._load_emb()
        if self.training:
            assert os.path.exists(self.short_path), "Shortest path not exists."
            assert os.path.exists(self.oracle_path), "LLM-refined path not exists."
            shortest_paths_cache = self._load_data(self.short_path)
            oracle_paths_cache = self._load_data(self.oracle_path)

        processed_data = []
        
        for sample in raw_data:
            sample_id = sample['id']
            sample_embd = embs.get(sample_id, None)
            if not sample_embd or sample["a_entity"] == ['null']:
                continue
            
            oracle_paths =  oracle_paths_cache.get(sample_id, []) if self.training else []
            shortest_paths = shortest_paths_cache.get(sample_id, []) if self.training else []
            
            if self.training and (not oracle_paths or max(len(path) for path in oracle_paths) == 0):
                continue
            if self.split == 'test' and self.data_name == 'grailqa' and sample['domain'] not in self.domains:
                continue
            
            all_entities = sample["text_entity_list"] + sample["non_text_entity_list"]
            all_relations = sample["relation_list"]
            h_id_list, r_id_list, t_id_list = sample["h_id_list"], sample["r_id_list"], sample["t_id_list"]
            all_triplets = [(all_entities[h], all_relations[r], all_entities[t]) for (h,r,t) in zip(h_id_list, r_id_list, t_id_list)]

            topic_entity_mask = torch.zeros(len(all_entities))
            topic_entity_mask[sample['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)

            x_ent = sample_embd["entity_embs"]
            assert not torch.all(x_ent == 0, dim=1).any()
            # non-text entities as all-zeros
            x = torch.cat([x_ent, torch.zeros(len(sample['non_text_entity_list']), x_ent.size(1))], dim=0)
            edge_index = torch.stack([torch.tensor(h_id_list), 
                                      torch.tensor(t_id_list)], axis=0)
            edge_attr = sample_embd['relation_embs'][r_id_list]

            def path2tid(all_t, all_p):
                return remove_duplicates([all_t.index(item) for path in all_p for item in path])
            
            shortest_path_idx = path2tid(all_triplets, shortest_paths)
            # oracle_path_idx = path2tid(all_triplets, oracle_paths)

            # bidir
            selected_ = [(item[0], item[-1]) for path in oracle_paths for item in path]
            oracle_path_idx = [i for i,t in enumerate(all_triplets) if (t[0], t[-1]) in selected_ or (t[-1], t[0]) in selected_]
            oracle_path_idx = remove_duplicates(oracle_path_idx)

            if self.training and len(oracle_path_idx) == 0:
                continue

            processed_sample = {
                "id": sample_id,
                "q": sample["question"],
                "q_embd": sample_embd['q_emb'],
                "a_entity": [all_entities[idx] for idx in sample["a_entity_id_list"]],
                "q_entity": [all_entities[idx] for idx in sample["q_entity_id_list"] if all_entities[idx] not in NOISES],
                "triplets": all_triplets,
                "relevant_idx": oracle_path_idx,
                "shortest_path_idx": shortest_path_idx,
                "y": sample.get("a_entity", []),
                "graph": Data(x=x, edge_attr=edge_attr, edge_index=edge_index.long(), topic_signal=topic_entity_one_hot.float())
            }
            processed_data.append(processed_sample)

        return processed_data

    def _load_data(self, file_path):
        if file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith(".pth"):
            return torch.load(file_path)
        else:
            raise NotImplemented

    def _load_emb(self, ):
        if self.data_name == 'grailqa':
            return torch.load(f"/home/comp/cscxliu/derek/LTRoG/data_files/retriever/{self.data_name}/emb/gte-large-en-v1.5/{self.split}.pth")

        full_dict = dict()
        # emb_path = opj(self.root, self.data_name, "processed", "emb", self.split)
        emb_path = f"/home/comp/cscxliu/derek/LTRoG/data_files/retriever/{self.data_name}/emb/gte-large-en-v1.5/" + self.split
        for file in os.listdir(emb_path):
            dict_file = torch.load(opj(emb_path, file))
            full_dict.update(dict_file)
        return full_dict

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, i):
        return self.processed_data[i]
    
    @staticmethod
    def prepare_shortest(all_triplets, q_entities, a_entities):
        """
        Build KG and find shortest paths from q to a.
        """
        G = nx.DiGraph()
        for (a, b, c) in all_triplets:
            G.add_node(a)
            G.add_node(c)
            G.add_edge(a, c, relation=b)
        all_paths = []
        for q in q_entities:
            for a in a_entities:
                paths_q_a = _shortest_path(G, q, a)
                if len(paths_q_a) > 0:
                    all_paths.extend(paths_q_a)
        all_paths = [p for p in all_paths if p]
        return all_paths
    
    @staticmethod
    def gen_path(G, source, target):

        shortest_paths = nx.all_shortest_paths(G, source=source, target=target)
        all_triplets_in_path = []
        for path in shortest_paths:
            triplets_in_path = [(path[i], G[path[i]][path[i+1]]['relation'], path[i+1]) for i in range(len(path) - 1)]
            all_triplets_in_path.append(triplets_in_path)
        return all_triplets_in_path

    @staticmethod
    def _shortest_path(nx_g, q_entity, a_entity):
        try:
            forward_paths = gen_path(nx_g, q_entity, a_entity)
        except:
            forward_paths = []
        
        try:
            backward_paths = gen_path(nx_g, a_entity, q_entity)
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        
        return refined_paths


class RetrievalDatasetWithoutEmb(RetrievalDataset):
    """
    Subclass of RetrievalDataset that omits embeddings.
    """

    def __init__(self, config, split, **kwargs):
        super().__init__(config, split, **kwargs)

    def refine_shortest(self, sample, shortest_paths, all_triplets):

        """
        Refine shortest paths build from `prepare_shortest` and to feed to LLMs
        """

        sample_type, q_entities, a_entities = sample['function'], sample['q_entity'], sample["a_entity"]
        seed_set = set()
        ans = a_entities[0]  # fetch only one representative answer
        
        # only focus on one answer to release LLM refinement burden
        if len(a_entities) > 1:
            shortest_paths = [p for p in shortest_paths if ans in path2set(p)]
        
        shortest_paths = remove_dup_directions(shortest_paths, q_entities)
        shortest_paths = remove_dup_logics(shortest_paths)
        
        # process special question type: count
        if sample_type == 'count':
            shortest_paths = self.get_centric_grpah(all_triplets, q_entities[0])
        
        # process special question type: argmax / argmin / > / <
        if sample_type not in ['none', 'count']:
            for p in shortest_paths:
                seed_set |= set(get_entities_from_path(p))
            seed_set -= set(q_entities)
            
            for q in seed_set:
                shortest_paths += self.get_centric_grpah(all_triplets, q)

        shortest_paths = remove_dup_directions(shortest_paths, q_entities+list(seed_set))
        shortest_paths = remove_dup_logics(shortest_paths)
        shortest_paths = remove_dup_path(shortest_paths)

        return shortest_paths
        
    @staticmethod
    def get_centric_grpah(all_triplets, q):
        # We do not consider CVT nodes for simplicity
        selections, reserved_relations = [], set()
        for triple in all_triplets:
            if triple[0] == q and not triple[-1].startswith(('m.', 'g.')) and triple[1] not in reserved_relations:
                selections.append([triple])
                reserved_relations.add(triple[1])
            if triple[-1] == q and not triple[0].startswith(('m.', 'g.')) and triple[1] not in reserved_relations:
                selections.append([triple])
                reserved_relations.add(triple[1])
        return selections

    def process(self, raw_data):
        print("begin")
        processed_data = []

        # First step. Generate shortest paths
        if os.path.exists(self.short_path):
            shortest_paths_cache = self._load_data(self.short_path)
        else:
            shortest_paths_cache = {}
            for sample in raw_data:
                all_triplets, q_entities, a_entities = self.get_all_triplets(sample), sample['q_entity'], sample['a_entity']
                shortest_paths_cache[sample['id']] = self.prepare_shortest(all_triplets, q_entities, a_entities)
            torch.save(shortest_paths_cache, self.short_path)
            print("shortest paths preparation Done.")
        
        shortest_paths_processed_cache = {}
        # Second, process shortest path for LLM refinement.
        for sample in raw_data:

            ans = sample["a_entity"][0] if sample['function'] != 'count' else '1'
            all_triplets = self.get_all_triplets(sample)

            shortest_paths = shortest_paths_cache.get(sample['id'], [])
            shortest_paths = self.refine_shortest(sample, shortest_paths, all_triplets)
            shortest_paths_processed_cache[sample['id']] = shortest_paths
            
            processed_sample = {
                "id": sample['id'],
                "q": sample["question"],
                "y": [ans],  # only one answer is enough
                "triplets": all_triplets,
                "relevant_paths": shortest_paths,
            }
            if len(shortest_paths) <= 1 or max(len(path) for path in shortest_paths) == 0 or sample["a_entity"] == ['null']:
                continue
            processed_data.append(processed_sample)
        
        torch.save(shortest_paths_processed_cache, opj(self.root, self.data_name, "processed", f"intermediate.pth"))
        # input("success")
        return processed_data
    
    @staticmethod
    def get_all_triplets(sample):
        all_entities = sample["text_entity_list"] + sample["non_text_entity_list"]
        all_relations = sample["relation_list"]
        h_id_list, r_id_list, t_id_list = sample["h_id_list"], sample["r_id_list"], sample["t_id_list"]
        all_triplets = [(all_entities[h], all_relations[r], all_entities[t]) for (h, r, t) in zip(h_id_list, r_id_list, t_id_list)]
        return all_triplets
            

def path2set(path):
    return {itm for triple in path for itm in triple}


def remove_dup_directions(shortest_paths, q_entities):
    # only focus on forward if both forward and backward exists.
    path_forward = [path for path in shortest_paths if path[0][0] in q_entities]
    path_backward = [path for path in shortest_paths if path[-1][-1] in q_entities]
    all_paths = path_forward + path_backward

    cache = set()
    selected_paths = []
    for path in all_paths:
        ent = get_entities_from_path(path)
        if frozenset(ent) not in cache:
            cache.add(frozenset(ent))
            selected_paths.append(path)
    return selected_paths


def remove_dup_logics(shortest_paths):
    # fetech one representative logics, filter repeats
    reserved_logics, selected_paths = set(), []

    for p in shortest_paths:
        logic_p = (p[0][0], ) + tuple(triple[1] for triple in p) + (p[-1][-1], )
        if logic_p not in reserved_logics:
            reserved_logics.add(logic_p)
            selected_paths.append(p)
    return selected_paths


def remove_dup_path(shortest_paths):
    reserved, final_paths = [], []
    # final processing of paths: remvoe duplicate
    for p in shortest_paths:
        if len(p) == 1 and p[0] in reserved:
            continue
        reserved.extend(p)
        final_paths.append(p)
    return final_paths


def split_samples(processed_samples):
    new_samples = []
    
    for processed_sample in processed_samples:
        relevant_paths = processed_sample["relevant_paths"]
        
        if len(relevant_paths) > 20:
            num_chunks = (len(relevant_paths) + 9) // 20
            
            for i in range(num_chunks):
                chunk_paths = relevant_paths[i * 20: (i + 1) * 20]
                new_sample = deepcopy(processed_sample)
                
                new_sample["id"] = f"{processed_sample['id']}+{i}"
                new_sample["relevant_paths"] = chunk_paths
                
                new_samples.append(new_sample)
        else:
            new_samples.append(processed_sample)
    
    return new_samples

def get_entities_from_path(path):
    return [triple[0] for triple in path] + [path[-1][-1]]

def get_hard_answers(all_entities, ans_list):
    ans_list_ = []
    for a in ans_list:
        a = a.split('ans:')[-1]
        for e in all_entities:
            if match(a, e) or match(e, a) or match(e, a.split('ans:')[-1].strip()):
                ans_list_.append(e)
                break
    if len(ans_list_) < len(ans_list):
        print("abnormal.")
    return remove_duplicates(ans_list_)


def get_paths(all_triplets, q_list, a_list):
    if len(a_list) == 0:
        return []
    # build graph
    G = nx.DiGraph()
    for (a, b, c) in all_triplets:
        G.add_node(a)
        G.add_node(c)
        G.add_edge(a, c, relation=b)
    # Find shortest
    all_paths = []
    for q in q_list:
        for a in a_list:
            paths_q_a = _shortest_path(G, q, a)
            if len(paths_q_a) > 0:
                all_paths.extend(paths_q_a)
    return all_paths


def gen_path(G, source, target):
    shortest_paths = nx.all_shortest_paths(G, source=source, target=target)
    all_triplets_in_path = []
    for path in shortest_paths:
        triplets_in_path = [(path[i], G[path[i]][path[i+1]]['relation'], path[i+1]) for i in range(len(path) - 1)]
        all_triplets_in_path.append(triplets_in_path)
    return all_triplets_in_path


def _shortest_path(nx_g, q, a):

    try:
        forward_paths = gen_path(nx_g, q, a)
    except:
        forward_paths = []
    
    try:
        backward_paths = gen_path(nx_g, a, q)
    except:
        backward_paths = []
    
    full_paths = forward_paths + backward_paths
    if (len(forward_paths) == 0) or (len(backward_paths) == 0):
        return full_paths
    
    min_path_len = min([len(path) for path in full_paths])
    refined_paths = []
    for path in full_paths:
        if len(path) == min_path_len:
            refined_paths.append(path)
    
    return refined_paths

def get_pos_paths(all_paths, ans):
    all_paths = [item for item in all_paths if item != []]
    return [path for path in all_paths if path[0][0] in ans or path[-1][-1] in ans]