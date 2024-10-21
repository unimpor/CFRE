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
        self.root = config['dataset']['root']
        self.data_name = config['dataset']['name']
        self.data = self._load_data(opj(self.root, self.data_name, "data", f"{self.split}.pkl"))
        # which contains some coarse retrieval results and shorted-path relevant info
        # TODO: ask mufei which file we should use from
        self.scored_data = self._load_data(opj(self.root, self.data_name, "postpr", f"{self.split}.pkl"))
        # 'target_relevant_triples' 'scored_triples'

        self.emb = self._load_emb(config[""])

        self.processed_data = self.assembly(config[""])

    def assembly(self, coarse_filter=True):
        processed_data = []
        for sample in self.data:
            sample_id = sample['id']
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

            # TODO: add shorted-path and gpt relevant
            sample['relevant:shortest'] = {}
            # sample['relevant:gpt-4'] = {}

            if coarse_filter:
                # TODO: coarse filtering: del sample['h_id_list'], sample['r_id_list'], sample['t_id_list'] according to self.scored_data
                pass

            processed_data.append(sample)

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
