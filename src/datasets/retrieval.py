import os
from os.path import join as opj
import pickle
import torch.nn.functional as F
import torch
from torch_geometric.data.data import Data


class RetrievalDataset:
    """
    load retrieval results and post-processing
    """

    def __init__(self, config, split):
        self.split = split
        self.config = config
        self.root = config['root']
        self.data_name = config['name']
        self.post_filter = config["post_filter"]
        self.filtering_id = None
        if self.post_filter:
            reference = self._load_data(opj(self.root, self.data_name, "checkpoints", self.post_filter))
            self.filtering_id = [k for k,v in reference.items() if v["F1"]==1.0]
        
        if os.path.exists(self.processed_file_names):
            print('Load processed file..')
            self.processed_data = torch.load(self.processed_file_names)
        else:
            raw_data = self._load_data(opj(self.root, self.data_name, "processed", f"{self.split}.pkl"))
            # which contains some coarse retrieval results and shorted-path relevant info
            scored_data = self._load_data(opj(self.root, self.data_name, "processed", f"{self.data_name}_241028_{self.split}.pth"))
            # 'target_relevant_triples' 'scored_triples'
            embs = self._load_emb()
            # print(len(raw_data), len(scored_data))
            self.processed_data = self.process(raw_data, scored_data, embs)

    @property
    def processed_file_names(self):
        filename = f"completed_{self.split}_{self.post_filter}.pth" if self.post_filter else f"completed_{self.split}.pth"
        return opj(self.root, self.data_name, "processed", filename)

    def process(self, raw_data, scored_data, embs):
        processed_data = []
        for sample in raw_data:
            sample_id = sample['id']
            sample_scored = scored_data.get(sample_id, None)

            if not sample_scored:
                continue
            if self.config['skip_no_path'] and (sample_scored['max_path_length'] in [None, 0]):
                continue
            if self.split=="train" and self.post_filter and sample_id in self.filtering_id:
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

            processed_sample = {
                "id": sample_id,
                "q": sample["question"],
                "q_embd": sample_embd['q_emb'],
                "a_entity": [all_entities[idx] for idx in sample["a_entity_id_list"]],
                "triplets": all_triplets,
                "relevant_idx": [all_triplets.index(item[:3]) for item in sample_scored['target_relevant_triples']],
                "y": sample_scored['a_entity_in_graph'],
                "graph": Data(x=x, edge_attr=edge_attr, edge_index=edge_index, topic_signal=topic_entity_one_hot.float())
            }

            # double check
            assert processed_sample["graph"].topic_signal.shape[0] == processed_sample["graph"].x.shape[0]

            processed_data.append(processed_sample)
        # torch.save(processed_data, self.processed_file_names)


        return processed_data

    def _load_data(self, file_path):
        if file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith(".pth"):
            return torch.load(file_path)

    def _load_emb(self, ):
        full_dict = dict()
        emb_path = opj(self.root, self.data_name, "processed", "emb", self.split)
        for file in os.listdir(emb_path):
            if file.endswith('.pth'):
                dict_file = torch.load(opj(emb_path, file))
                full_dict.update(dict_file)
        return full_dict

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, i):
        return self.processed_data[i]
