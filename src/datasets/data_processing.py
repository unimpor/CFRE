import os
from os.path import join as opj
import pickle
import torch.nn.functional as F
import torch
import numpy as np

FILTER_K = 300

if __name__ == "__main__":
    raw_data = pickle.load(open("../samples/webqsp_val_raw.pkl", "rb"))
    scored_data = torch.load("../samples/webqsp_241028_val.pth")
    embedding1 = torch.load("../samples/0.pth")
    embedding2 = torch.load("../samples/1.pth")
    embeddings = {**embedding1, **embedding2}
    processed_data = []
    for sample in raw_data:
        processed_sample = {}
        sample_id = sample['id']
        scored_sample = scored_data[sample_id]
        sample_embd = embeddings[sample_id]
        if scored_sample['max_path_length'] in [None, 0]:
            continue

        fh_id_list, fr_id_list, ft_id_list = [], [], []

        scored_triplets = scored_sample['scored_triples']
        assert len(scored_triplets) == len(sample["h_id_list"])
        # filtered_triplets = [(t[0], t[1], t[2]) for idx, t in enumerate(scored_triplets) if idx < FILTER_K]
        filter_k = min(FILTER_K, len(scored_triplets))
        filtered_triplets = scored_triplets[:filter_k]
        # Find preserved (h,r,t) id.
        entity_list = sample['text_entity_list'] + sample['non_text_entity_list']
        relation_list = sample['relation_list']

        for (h, r, t, _) in filtered_triplets:
            fh_id_list.append(entity_list.index(h))
            fr_id_list.append(relation_list.index(r))
            ft_id_list.append(entity_list.index(t))
        # for h_id, r_id, t_id in zip(sample['h_id_list'], sample['r_id_list'], sample['t_id_list']):
        #     h, r, t = entity_list[h_id], relation_list[r_id], entity_list[t_id]
        #     if (h, r, t) in filtered_triplets:
        #         fh_id_list.append(h_id)
        #         fr_id_list.append(r_id)
        #         ft_id_list.append(t_id)
        relevant_ids = [filtered_triplets.index(tp) for tp in scored_data['target_relevant_triples'] if tp in filtered_triplets]
        if len(relevant_ids) < len(scored_data['target_relevant_triples']):
            print(f"Relevant info is missing from coarse retrieval: {sample_id}")

        assert len(fh_id_list) == len(fr_id_list) == len(ft_id_list) == filter_k

        # Get filtered node & edge embeddings, and edge-index
        # 1. edge attribute
        edge_attr = sample_embd['relation_embs'][fr_id_list]
        # 2. node embeddings
        entity_embeddings_org = torch.cat([
            sample_embd['entity_embs'],
            torch.zeros(len(sample['non_text_entity_list']), sample_embd['entity_embs'].size(1))
        ], dim=0)
        selected_nodes = np.unique(fh_id_list + ft_id_list)
        entity_embeddings = entity_embeddings_org[selected_nodes]
        # 3. edge index, from old idx to new idx
        idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        edge_index_org = np.stack([fh_id_list, ft_id_list], axis=0)
        edge_index = np.vectorize(idx_mapping.get)(edge_index_org)
        # TODO: q-entity, q-entity-in-graph, a-entity-in-graph
        processed_sample = {
            "edge_index": edge_index,
            "x": entity_embeddings,
            "y": scored_data['a_entity_in_graph'],
            "edge_attr": edge_attr,
            "triplets": filtered_triplets,
            'relevant_idx': relevant_ids,
            "id": sample_id,
            "q": sample['question'],
            "q_embd": sample_embd['q_emb'],
        }
        processed_data.append(processed_sample)
        break

    torch.save(processed_data, "../samples/processed_webqsp_val.pth")
