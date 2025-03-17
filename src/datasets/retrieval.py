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
from src.utils import print_log, remove_duplicates, match
from copy import deepcopy
import time

NOISES = ["Country", "College/University", "Male", "Continent"]

class RetrievalDataset:
    """
    load retrieval results and post-processing
    """

    def __init__(self, config, split, **kwargs):
        self.split = split
        self.config = config
        self.root = config['root']
        self.data_name = config['name']
        self.post_filter = kwargs.get("post_filter", False)  # filter bad samples.

        self.oracle_path = kwargs.get("opath", None)
        self.hard_path = kwargs.get("hpath", None)

        self.training = kwargs.get("training", True)
        raw_data = self._load_data(opj(self.root, self.data_name, "processed", f"{self.split}.pkl"))
        self.processed_data = self.process(raw_data, )

    @property
    def processed_file_names(self):
        filename = f"completed_{self.split}_{self.post_filter}.pth" if self.post_filter else f"completed_{self.split}.pth"
        return opj(self.root, self.data_name, "processed", filename)

    def process(self, raw_data, ):
        num = 0
        embs = self._load_emb()
        if self.training:
            shortest_paths_cache = torch.load("datasets/cwq/processed/shortest_path.pth")
            oracle_paths_cache = torch.load(self.oracle_path)
            if self.hard_path:
                hard_paths_cache = torch.load(self.hard_path)
        # only for the need of test
        processed_data, filtering_ids = [], []
        if self.split == "test" and self.post_filter:
            filtering_ids = torch.load(f"datasets/{self.data_name}/processed/test_filtering_bad_ids.pth")
        for sample in raw_data:
            sample_id = sample['id']
            sample_embd = embs.get(sample_id, None)
            if not sample_embd or sample_id in filtering_ids:
                continue
            # shortest path: [path0, path1, ...]
            # dat: [0, -1, 1]
            dat =  oracle_paths_cache.get(sample_id, []) if self.training else []
            if self.training and not dat:
                continue
            shortest_paths = shortest_paths_cache[sample_id] if self.training else []
            # assert len(shortest_paths) == len(dat)
            # oracle_paths = []
            # if self.training:
            #     # target = 1 if 1 in dat else 0
            #     # oracle_paths = [path for (path, s) in zip(shortest_paths, dat) if s == target]
            #     oracle_paths = [path for (path, s) in zip(shortest_paths, dat) if s in [0,1]]
            oracle_paths = dat
            hard_negatives_paths, hard_positive_paths = [], []

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
            
            if self.training and self.hard_path:
                hard_negatives = get_hard_answers(all_entities, hard_paths_cache[sample_id]['not_prec'])
                # assert hard neg not intersected with ans
                hard_negatives = [i for i in hard_negatives if i not in sample["a_entity"]]
                # assert len(set(hard_negatives) & set(sample["a_entity"])) == 0
                # hard_negatives_paths = get_paths(all_triplets, sample["q_entity"], hard_negatives)
                # hard_positive_paths = get_pos_paths(oracle_paths, hard_paths_cache[sample_id]['missing'])
                hard_negatives_paths = []
                for path in hard_paths_cache[sample_id]['select_paths']:
                    path_entities = [i for t in path for i in t]
                    if set(path_entities) & set(hard_negatives):
                        hard_negatives_paths.append(path)

            def path2tid(all_t, all_p):
                return remove_duplicates([all_t.index(item) for path in all_p for item in path])
            
            shortest_path_idx = path2tid(all_triplets, shortest_paths)
            oracle_path_idx = path2tid(all_triplets, oracle_paths)
            hard_neg_path_idx_ = path2tid(all_triplets, hard_negatives_paths)
            hard_neg_path_idx = [i for i in hard_neg_path_idx_ if i not in oracle_path_idx]
            # if len(set(oracle_path_idx) & set(hard_neg_path_idx)) > 0:
            #     print(len(set(oracle_path_idx) & set(hard_neg_path_idx)))
            #     num += 1
            hard_pos_path_idx = path2tid(all_triplets, hard_positive_paths)

            # shortest_path_idx = [all_triplets.index(item) for path in shortest_paths for item in path]
            # shortest_path_idx = remove_duplicates(shortest_path_idx)

            # map {-1, 0, 1} in shortest path to {1, 2, 3} 
            scores = -torch.ones(len(all_triplets))
            # for path, score in zip(shortest_paths, dat):
            #     for t in path:
            #         scores[all_triplets.index(t)] = max(score, -1)
            # scores = (scores + 1).long()


            processed_sample = {
                "id": sample_id,
                "q": sample["question"],
                "q_embd": sample_embd['q_emb'],
                "a_entity": [all_entities[idx] for idx in sample["a_entity_id_list"]],
                "q_entity": [all_entities[idx] for idx in sample["q_entity_id_list"] if all_entities[idx] not in NOISES],
                "triplets": all_triplets,
                "relevant_idx": oracle_path_idx,
                "relevant_idx_in_path": [[all_triplets.index(item) for item in path] for path in oracle_paths],
                "shortest_path_idx": shortest_path_idx,
                "hard_idx": hard_neg_path_idx + hard_pos_path_idx,
                "y": sample["a_entity"],
                "graph": Data(x=x, scores=scores, edge_attr=edge_attr, edge_index=edge_index.long(), topic_signal=topic_entity_one_hot.float())
            }
            processed_data.append(processed_sample)
        # print(num)
        # input("suncc")
        return processed_data

    def _load_data(self, file_path):
        if file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith(".pth"):
            return torch.load(file_path)

    def _load_emb(self, ):
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

    def _extract_paths_(
        self,
        sample,
        type="shortest",
        ret_num=1
    ):
        assert type in ["shortest", "all"]
        func_series = {"shortest": self._shortest_path,
                       "all": self._all_path}

        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs.
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                # if len(paths_q_a) > 0:
                #     shortest_len = len(paths_q_a[0]) - 1
                #     paths_q_a_longer = self._all_path(nx_g, q_entity_id, a_entity_id, cutoff=shortest_len+ret_num)
                #     path_list_.extend(paths_q_a_longer)
                # paths_q_a = func_series[func_type](nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # Each processed path is a list of triple IDs.
        path_list = []

        for path in path_list_:
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)
            triples_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                triple_id_i_list = nx_g[h_id_i][t_id_i]['triple_id']
                # triple_id_i_list = [nx_g[h_id_i][t_id_i]['triple_id']]
                triples_path.append(triple_id_i_list)

            path_list.append(triples_path)
        
        return path_list

    def _get_nx_g(
        self,
        h_id_list,
        r_id_list,
        t_id_list
    ):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)

        return nx_g

    def _all_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id,
        cutoff=None
    ):
        try:
            forward_paths = list(nx.all_simple_paths(nx_g, q_entity_id, a_entity_id, cutoff=cutoff))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_simple_paths(nx_g, a_entity_id, q_entity_id, cutoff=cutoff))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        return full_paths

    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []
        
        # return forward_paths
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        # full_paths = forward_paths if len(forward_paths) != 0 else backward_paths
        
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        
        return refined_paths

    def viz_subgraph(self, triples, sample):
        sample_id = sample["id"]
        sample_q = sample["question"]

        G = nx.DiGraph()

        for head, relation, tail in triples:
            G.add_edge(head, tail, relation=relation)

        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(12, 12))

        # nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", alpha=0.9)
        # nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

        # nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=15, edge_color="gray", connectionstyle="arc3,rad=0.1")

        # edge_labels = nx.get_edge_attributes(G, "relation")
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20, font_color="red", label_pos=0.5)

        nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_size=8, font_color="black")

        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, font_color="red")

        plt.title(f"{sample_q}", fontsize=14)
        plt.savefig(f"fig/graph_viz/{sample_id}.png", dpi=400)

class RetrievalDatasetWithoutEmb(RetrievalDataset):
    """
    Subclass of RetrievalDataset that omits embeddings.
    """

    def __init__(self, config, split):
        super().__init__(config, split)

    def process(self, raw_data):
        print("begin")
        # preserve = list(torch.load("logging/cwq/gpt-4o-mini/DDE/path-level-detection/inference-ret-50.pth").keys())
        processed_data = []
        shortest_paths_cache = torch.load(f"datasets/{self.data_name}/processed/shortest_path.pth")
        for sample in raw_data:
            sample_id = sample['id']
            # if self.split == "test" and sample_id not in preserve:
            #     continue
            # if sample_id not in ["WebQTest-2008_8c6b952c6bd963f0ece4e401c9eb731a", 
            #                      "WebQTrn-2518_1ef15e22372df70baf01b72850deb14d", 
            #                      "WebQTest-415_b6ad66a3f1f515d0688c346e16d202e6",
            #                      "WebQTest-341_0f5ccea2d11b712eda64ebf2f6aeb1ee"]:
            #     continue
            shortest_paths = shortest_paths_cache.get(sample_id, [])
            if not shortest_paths or max(len(path) for path in shortest_paths) == 0:
                continue
            # if len(shortest_paths) > 40:
            #     print("wait for next time ...")
            #     continue
            # Since embeddings are omitted, replace embedding-dependent logic
            all_entities = sample["text_entity_list"] + sample["non_text_entity_list"]
            all_relations = sample["relation_list"]
            h_id_list, r_id_list, t_id_list = sample["h_id_list"], sample["r_id_list"], sample["t_id_list"]
            all_triplets = [(all_entities[h], all_relations[r], all_entities[t]) for (h, r, t) in zip(h_id_list, r_id_list, t_id_list)]

            # Create placeholder tensors for graph features (since embs are omitted)
            x = torch.zeros(len(all_entities), 1)  # Dummy node features
            edge_index = torch.stack([torch.tensor(h_id_list), torch.tensor(t_id_list)], axis=0)
            edge_attr = torch.zeros(len(r_id_list), 1)  # Dummy edge features

            # shortest_path_idx = [all_triplets.index(item[:3]) for path in shortest_paths for item in path]
            # relevant_idx = shortest_path_idx

            def paths_stat(paths):
                paths_len = len(paths)
                all_triplets_selected = set(list(chain(*paths)))
                triple_len = len(all_triplets_selected)
                num_relations = len(set([all_triplets[i][1] for i in all_triplets_selected]))
                return paths_len, triple_len, num_relations
            
            # relevant_paths_0 = self._extract_paths_(sample=sample, ret_num=0)
            # relevant_paths_1 = self._extract_paths_(sample=sample, ret_num=1)
            # relevant_paths_2 = self._extract_paths_(sample=sample, ret_num=2)
            # relevant_paths_3 = self._extract_paths_(sample=sample, ret_num=3)

            # (pl0, tl0, rl0), (pl1, tl1, rl1), (pl2, tl2, rl2), (pl3, tl3, rl3) = paths_stat(relevant_paths_0), paths_stat(relevant_paths_1), paths_stat(relevant_paths_2), paths_stat(relevant_paths_3)
            
            # if len(relevant_idx_loaded) != tl0:
            #     print("report")
            # with open('paths_stat.csv', mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([sample_id, len(all_triplets), len(relevant_idx_loaded), pl0, tl0, rl0, pl1, tl1, rl1, pl2, tl2, rl2, pl3, tl3, rl3])
            
            # relevant_paths = self._extract_paths_(sample=sample)
            # relevant_paths = [[all_triplets[i] for i in path] for path in relevant_paths]
            # shortest_paths_cache[sample_id] = relevant_paths

            # scored_data[sample_id]["relevant_paths_1"] = relevant_paths_1
            # print("Done" + " " + sample_id)
            # for idx, path in enumerate(relevant_idx_):
            #     print(f"path{idx+1}--length {len(path)}: ")
            #     print([all_triplets[i] for i in path])
            # input(0)


            # from itertools import chain
            # relevant_idx_ = list(chain(*relevant_idx_))
            # self.viz_subgraph([[all_triplets[i][0], all_triplets[i][1].split(".")[-1], all_triplets[i][2]] for i in relevant_idx_], sample)
            
            # processed_data[sample_id] = {
            #     "q": sample["question"],
            #     "a": sample["a_entity"],
            #     "paths": relevant_paths,
            #     "select": results[sample_id]["select"],
            #     "all_triplets": all_triplets
            # }

            processed_sample = {
                "id": sample_id,
                "q": sample["question"],
                "q_embd": torch.zeros(1, 1),  # No embeddings for the question
                "a_entity": [all_entities[idx] for idx in sample["a_entity_id_list"]],
                "triplets": all_triplets,
                "relevant_paths": shortest_paths,
                "y": sample["a_entity"],
                "graph": Data(x=x, edge_attr=edge_attr, edge_index=edge_index.long())
            }
            processed_data.append(processed_sample)
        # torch.save(scored_data, opj(self.root, self.data_name, "processed", f"{self.data_name}_242028_{self.split}_path1.pth"))
        # torch.save(shortest_paths_cache, opj(self.root, self.data_name, "processed", f"shortest_path.pth"))
        # input("success")
        return split_samples(processed_data)
        # return processed_data



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