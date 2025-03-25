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

TO_FILTER_REL = {
            "common.topic.description",
            "kg.object_profile.prominent_type",
            "en",
            "common.topic.alias"
        }

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
        
        embs = self._load_emb()
        if self.training:
            shortest_paths_cache = torch.load(f"datasets/{self.data_name}/processed/shortest_path.pth")
            oracle_paths_cache = torch.load(self.oracle_path)

        processed_data, filtering_ids = [], []
        
        for sample in raw_data:
            sample_id = sample['id']
            sample_embd = embs.get(sample_id, None)
            if not sample_embd or sample_id in filtering_ids:
                continue

            oracle_paths =  oracle_paths_cache.get(sample_id, []) if self.training else []
            shortest_paths = shortest_paths_cache[sample_id] if self.training else []
            if self.training and (len(oracle_paths) <= 1 or max(len(path) for path in oracle_paths) == 0 or sample["a_entity"] == ['null']):
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
            oracle_path_idx = path2tid(all_triplets, oracle_paths)
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
                # "relevant_idx_in_path": [[all_triplets.index(item) for item in path] for path in oracle_paths],
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

    def _load_emb(self, ):
        if self.data_name == 'grailqa':
            if self.split in ['train', 'val']:
                return torch.load(f"/home/comp/cscxliu/derek/LTRoG/data_files/retriever/{self.data_name}/emb/gte-large-en-v1.5/train.pth")
            else:
                return torch.load(f"/home/comp/cscxliu/derek/LTRoG/data_files/retriever/{self.data_name}/emb/gte-large-en-v1.5/test.pth")

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

    def prepare_shortest(self, ):
        """
        Build KG and find shortest paths from q to a.
        """
        pass

    def refine_shortest(self, sample, shortest_paths, all_triplets):

        """
        Refine shortest paths build from `prepare_shortest` and to feed to LLMs
        """

        sample_type, q_entities, a_entities = sample['function'], sample['q_entity'], sample["a_entity"]
        ans = a_entities[0]  # fetch only one representative answer

        if len(a_entities) > 1:
            shortest_paths = [p for p in shortest_paths if ans in path2set(p)]

        # fetech one representative logics, filter repeats
        reserved_logics, new_shortest_paths = set(), []

        for p in shortest_paths:
            logic_p = (p[0][0], ) + tuple(triple[1] for triple in p) + (p[-1][-1], )
            if logic_p not in reserved_logics:
                reserved_logics.add(logic_p)
                new_shortest_paths.append(p)
        
        shortest_paths = new_shortest_paths

        # process special question type
        # 'count' with #q-entities > 1 would be processed without LLMs
        # TODO: integrate 'count' with #q-entities > 1 into training?
        if sample_type == 'count' and len(q_entities) == 1:
            shortest_paths, _ = self.get_centric_grpah(all_triplets, q_entities[0])
            # shortest_paths, all_candidates = [], {}
            # reserved_set = set(sample["text_entity_list"])
            
            # for q in q_entities:
            #     candidates, extracted_entities = self.get_centric_grpah(all_triplets, q)
            #     all_candidates[q] = candidates
            #     reserved_set = reserved_set & extracted_entities
            # for _,v in all_candidates.items():
            #     shortest_paths += [itm for itm in v if reserved_set & path2set(itm)]     
        
        if sample_type not in ['none', 'count']:
            seed_set = set()
            for p in shortest_paths:
                seed_set |= set(get_entities_from_path(p))
            seed_set -= set(q_entities)
            
            for q in seed_set:
                candidates, _ = self.get_centric_grpah(all_triplets, q)
                shortest_paths += candidates
        
        final_paths = []
        # final processing of paths: remvoe duplicate, filter meaningless relations
        for p in shortest_paths:
            if p in final_paths:
                continue
            if {t[1] for t in p} & TO_FILTER_REL:
                continue
            final_paths.append(p)

        return final_paths
        
    @staticmethod
    def get_centric_grpah(all_triplets, q):
        # We do not consider CVT nodes for simplicity
        # TODO: CVT nodes?
        # forward first.
        selections_forward, selections_backward, extracted_targets, reserved_relations = [], [], set(), set()
        for triple in all_triplets:
            if triple[0] == q and not triple[-1].startswith(('m.', 'g.')):
                selections_forward.append(triple)
            if triple[-1] == q and not triple[0].startswith(('m.', 'g.')):
                selections_backward.append(triple)
        
        selections = []

        for triple in selections_forward:
            if triple[1] not in reserved_relations and triple[-1] not in extracted_targets:
                selections.append([triple])
                extracted_targets.add(triple[-1])
                reserved_relations.add(triple[1])
        for triple in selections_backward:
            if triple[1] not in reserved_relations and triple[0] not in extracted_targets:
                selections.append([triple])
                extracted_targets.add(triple[0])
                reserved_relations.add(triple[1])

        return selections, extracted_targets

    def process(self, raw_data):
        print("begin")
        processed_data = []
        shortest_paths_cache = torch.load(f"datasets/{self.data_name}/processed/refined_path.pth")
        # shortest_paths_cache_new = {}
        for sample in raw_data:

            ans = sample["a_entity"][0] if sample['function'] != 'count' else '1'
            all_entities = sample["text_entity_list"] + sample["non_text_entity_list"]
            all_relations = sample["relation_list"]
            h_id_list, r_id_list, t_id_list = sample["h_id_list"], sample["r_id_list"], sample["t_id_list"]
            all_triplets = [(all_entities[h], all_relations[r], all_entities[t]) for (h, r, t) in zip(h_id_list, r_id_list, t_id_list)]
            
            # if sample_id not in ["WebQTest-2008_8c6b952c6bd963f0ece4e401c9eb731a", 
            #                      "WebQTrn-2518_1ef15e22372df70baf01b72850deb14d", 
            #                      "WebQTest-415_b6ad66a3f1f515d0688c346e16d202e6",
            #                      "WebQTest-341_0f5ccea2d11b712eda64ebf2f6aeb1ee"]:
            #     continue
            shortest_paths = shortest_paths_cache.get(sample['id'], [])
            # shortest_paths = self.refine_shortest(sample, shortest_paths, all_triplets)
            # shortest_paths_cache_new[sample['id']] = shortest_paths
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
        # torch.save(scored_data, opj(self.root, self.data_name, "processed", f"{self.data_name}_242028_{self.split}_path1.pth"))
        # torch.save(shortest_paths_cache_new, opj(self.root, self.data_name, "processed", f"refined_path.pth"))
        # input("success")
        return processed_data

def path2set(path):
    return {itm for triple in path for itm in triple}


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