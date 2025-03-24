import os
import pickle
import torch

from datasets import load_dataset
from tqdm import tqdm


def get_emb(subset, text_encoder, save_file):
    emb_dict = dict()
    for i in tqdm(range(len(subset))):
        data = subset[i]
        text_entity_list, relation_list = data['text_entity_list'], data['relation_list']
        
        entity_embs = text_encoder.embed(text_entity_list)
        relation_embs = text_encoder.embed(relation_list)

        for query in data['queries']:
            id = query['id']
            q_text = query['question']
            q_emb = text_encoder.embed(q_text)
            emb_dict_i = {
                'q_emb': q_emb,
                'entity_embs': entity_embs,
                'relation_embs': relation_embs
            }
            emb_dict[id] = emb_dict_i
    torch.save(emb_dict, save_file)

def main(args):
    # Modify the config file for advanced settings and extensions.
    # torch.set_num_threads(config['env']['num_threads'])

    save_dir = f"/home/comp/cscxliu/derek/LTRoG/data_files/retriever/{args.dataset}/emb/{args.text_encoder_name}"
    os.makedirs(save_dir, exist_ok=True)

    # dataset
    with open("/home/comp/cscxliu/derek/WSDM2021_NSM/preprocessing/Freebase/grailqa.pkl", 'rb') as f:
        dataset = pickle.load(f)

    device = torch.device('cuda:0')
    
    
    if args.text_encoder_name == 'gte-large-en-v1.5':
        from src.models import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(args.text_encoder_name)
    
    get_emb(dataset, text_encoder, os.path.join(save_dir, 'train-val.pth'))
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, default='grailqa', 
                        choices=['webqsp', 'cwq', 'grailqa'], help='Dataset name')
    parser.add_argument('--text_encoder_name', type=str, default='gte-large-en-v1.5')
    args = parser.parse_args()
    
    main(args)