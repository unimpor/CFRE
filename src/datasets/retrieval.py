import os
from os.path import join as opj
import pickle
import torch.nn.functional as F
import torch


class RetrievalDataset:
    """
    load retrieval results and post-processing
    """

    def __init__(self, config, split):
        self.split = split
        self.config = config
        self.root = config['root']
        self.data_name = config['name']
        self.coarse_filter = config["coarse_filter"]
        self.filter_K = self.config["coarse_num_or_ratio"]  # First try 300.
        if os.path.exists(self.processed_file_names):
            print('Load processed file..')
            self.processed_data = torch.load(self.processed_file_names)
        else:
            self.data = self._load_data(opj(self.root, self.data_name, "data", f"{self.split}.pkl"))
            # which contains some coarse retrieval results and shorted-path relevant info
            self.scored_data = self._load_data(opj(self.root, self.data_name, "scored", f"{self.split}.pkl"))
            # 'target_relevant_triples' 'scored_triples'
            self.emb = self._load_emb(config["emb_name"])
            self.processed_data = self.process(self.coarse_filter)

    @property
    def processed_file_names(self):
        filename = f"{self.split}_{self.filter_K}.pth" if self.coarse_filter else f"{self.split}.pth"
        return opj(self.root, self.data_name, "processed", filename)

    def process(self, coarse_filter=True):
        # TODO: The following would be merged with `data_processing.py`
        processed_data = []
        for sample in self.data:
            sample_id = sample['id']
            if self.config['skip_no_path'] and (self.scored_data[sample_id]['max_path_length'] in [None, 0]):
                continue

            sample.update(self.emb[sample_id])  # 'entity_embs', 'q_emb', 'relation_embs'
            try:
                sample['a_entity'] = list(set(sample['a_entity']))
                sample['a_entity_id_list'] = list(set(sample['a_entity_id_list']))
            except:
                pass

            num_entities = len(sample['text_entity_list']) + len(sample['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities)
            topic_entity_mask[sample['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample['topic_entity_one_hot'] = topic_entity_one_hot.float()

            sample['relevant:shortest'] = self.scored_data[sample_id]['target_relevant_triples']
            # TODO: GPT-4 relevant info
            # sample['relevant:gpt-4'] = {}
            if coarse_filter:
                fh_id_list, fr_id_list, ft_id_list = [], [], []

                scored_triplets = self.scored_data[sample_id]['scored_triples']
                assert len(scored_triplets) == len(sample["h_id_list"])
                filtered_triplets = [(t[0], t[1], t[2]) for idx, t in enumerate(scored_triplets) if idx < self.filter_K]

                entity_list = sample['text_entity_list'] + sample['non_text_entity_list']
                relation_list = sample['relation_list']

                for h_id, r_id, t_id in zip(sample['h_id_list'], sample['r_id_list'], sample['t_id_list']):
                    h, r, t = entity_list[h_id], relation_list[r_id], entity_list[t_id]
                    if (h, r, t) in filtered_triplets:
                        fh_id_list.append(h_id)
                        fr_id_list.append(r_id)
                        ft_id_list.append(t_id)
                assert len(fh_id_list) == len(fr_id_list) == len(ft_id_list) == self.filter_K
                sample['h_id_list'], sample['r_id_list'], sample['t_id_list'] = fh_id_list, fr_id_list, ft_id_list

            processed_data.append(sample)

        torch.save(processed_data, self.processed_file_names)

        return processed_data

    def _load_data(self, file_path):
        if file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith(".pth"):
            return torch.load(file_path)

    def _load_emb(self, emb_name):
        full_dict = dict()
        emb_path = opj(self.root, self.data_name, "emb", emb_name, self.split)
        for file in os.listdir(emb_path):
            if file.endswith('.pth'):
                full_file = opj(emb_path, file)
                dict_file = torch.load(full_file)
                full_dict.update(dict_file)
        return full_dict

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, i):
        return self.processed_data[i]
