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
from src.utils import print_log


class RetrievalDataset:
    """
    load retrieval results and post-processing
    """

    def __init__(self, config, split, **kwargs):
        self.split = split
        self.config = config
        self.root = config['root']
        self.data_name = config['name']
        self.post_filter = False
        self.version = kwargs.get("version", None)  # tmp arg  
        if self.version in ["path", "triplet", "relation"]:
            self.tmp_data = torch.load(f"datasets/{self.data_name}/checkpoints/reward_alloc.pth")[self.version]    
        
        raw_data = self._load_data(opj(self.root, self.data_name, "processed", f"{self.split}.pkl"))
        # embs = self._load_emb()
        scored_data = self._load_data(opj(self.root, self.data_name, "processed", f"{self.data_name}_241028_{self.split}.pth"))
        
        self.processed_data = self.process(raw_data, scored_data)

    @property
    def processed_file_names(self):
        filename = f"completed_{self.split}_{self.post_filter}.pth" if self.post_filter else f"completed_{self.split}.pth"
        return opj(self.root, self.data_name, "processed", filename)

    def process(self, raw_data, scored_data):
        embs = self._load_emb()
        # only for the need of training
        shortest_paths_cache = torch.load("datasets/cwq/processed/shortest_path_only_forward.pth")
        processed_data = []
        for sample in raw_data:
            sample_id = sample['id']
            shortest_paths =  shortest_paths_cache.get(sample_id, [])
            if self.split == "train" and not shortest_paths:
                continue
            sample_embd = embs[sample_id] # 'entity_embs', 'q_emb', 'relation_embs'
            
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
            
            # shortest_path_idx = [all_triplets.index(item[:3]) for item in sample_scored['target_relevant_triples']]
            shortest_path_idx = [all_triplets.index(item[:3]) for path in shortest_paths for item in path]
            relevant_idx = shortest_path_idx
            # TODO: relevant idx implementation
            # if self.version in ["path", "triplet", "relation"]:
            #     relevant_idx = self.tmp_data[sample_id]
            
            processed_sample = {
                "id": sample_id,
                "q": sample["question"],
                "q_embd": sample_embd['q_emb'],
                "a_entity": [all_entities[idx] for idx in sample["a_entity_id_list"]],
                "triplets": all_triplets,
                "relevant_idx": relevant_idx,
                "shortest_path_idx": shortest_path_idx,
                # "shortest_paths_plus_1": sample_scored.get("relevant_paths_1", []),
                # "y": sample_scored['a_entity_in_graph'] if self.split != "test" else sample["a_entity"],
                "y": sample["a_entity"],
                "graph": Data(x=x, edge_attr=edge_attr, edge_index=edge_index.long(), topic_signal=topic_entity_one_hot.float())
            }
            assert processed_sample["graph"].topic_signal.shape[0] == processed_sample["graph"].x.shape[0]
            processed_data.append(processed_sample)
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
                # find shortest_path + k
                # TODO: k=1 is enough.
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

        # num_triples = len(sample['h_id_list'])
        # triple_scores = self._score_triples(
        #     path_list,
        #     num_triples
        # )
        
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
        
        return forward_paths
        # try:
        #     backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        # except:
        #     backward_paths = []
        
        # if forward_paths:
        #     return forward_paths
        # else:
        #     return backward_paths
        
        # full_paths = forward_paths + backward_paths
        # if (len(forward_paths) == 0) or (len(backward_paths) == 0):
        #     return full_paths
        # # full_paths = forward_paths if len(forward_paths) != 0 else backward_paths
        
        # min_path_len = min([len(path) for path in full_paths])
        # refined_paths = []
        # for path in full_paths:
        #     if len(path) == min_path_len:
        #         refined_paths.append(path)
        
        # return refined_paths

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
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red", label_pos=0.5)

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

    def process(self, raw_data, scored_data):
        print("begin")
        # results = torch.load("logging/cwq/gpt-4o-mini/DDE/warmup_training_Jan19_downs_path/evaluation_K100_resume.pth")
        processed_data = {}
        path_cache = {}
        for sample in raw_data:
            sample_id = sample['id']
            sample_scored = scored_data.get(sample_id, None)
            # if self.split != "test":
            if not sample_scored:
                continue
            if self.config['skip_no_path'] and (sample_scored['max_path_length'] in [None, 0]):
                continue

            # Since embeddings are omitted, replace embedding-dependent logic
            all_entities = sample["text_entity_list"] + sample["non_text_entity_list"]
            all_relations = sample["relation_list"]
            h_id_list, r_id_list, t_id_list = sample["h_id_list"], sample["r_id_list"], sample["t_id_list"]
            all_triplets = [(all_entities[h], all_relations[r], all_entities[t]) for (h, r, t) in zip(h_id_list, r_id_list, t_id_list)]

            # Create placeholder tensors for graph features (since embs are omitted)
            # x = torch.zeros(len(all_entities), 1)  # Dummy node features
            # edge_index = torch.stack([torch.tensor(h_id_list), torch.tensor(t_id_list)], axis=0)
            # edge_attr = torch.zeros(len(r_id_list), 1)  # Dummy edge features

            # relevant_idx_loaded = [all_triplets.index(item[:3]) for item in sample_scored['target_relevant_triples']]
            
            # save_path = "fig/cwq_info.txt"
            # print_log(sample_id, save_path)
            # print_log(sample["question"], save_path)
            # print_log("question entities", save_path)
            # print_log([all_entities[i] for i in sample["q_entity_id_list"]], save_path)
            # print_log("answer entities", save_path)
            # print_log([all_entities[i] for i in sample["a_entity_id_list"]], save_path)
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
            
            relevant_paths = self._extract_paths_(sample=sample)
            relevant_paths = [[all_triplets[i] for i in path] for path in relevant_paths]
            
            path_cache[sample_id] = relevant_paths
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
            # processed_sample = {
            #     "id": sample_id,
            #     "q": sample["question"],
            #     "q_embd": torch.zeros(1, 1),  # No embeddings for the question
            #     "a_entity": [all_entities[idx] for idx in sample["a_entity_id_list"]],
            #     "triplets": all_triplets,
            #     "relevant_idx": relevant_idx_loaded,
            #     "relevant_paths": sample_scored["relevant_paths"],
            #     "relevant_paths_1": sample_scored["relevant_paths_1"],
            #     "y": sample_scored['a_entity_in_graph'] if self.split != "test" else sample["a_entity"],
            #     "graph": Data(x=x, edge_attr=edge_attr, edge_index=edge_index.long())
            # }

            # processed_data.append(processed_sample)
        # torch.save(scored_data, opj(self.root, self.data_name, "processed", f"{self.data_name}_241028_{self.split}_path1.pth"))
        torch.save(path_cache, opj(self.root, self.data_name, "processed", f"shortest_path_only_forward.pth"))
        input("success")
        return processed_data
