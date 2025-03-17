import argparse
import os
import yaml
import torch
import pandas as pd
# import wandb
import numpy as np
from os.path import join as opj
from tqdm import tqdm
from cfre import CFRE
from rlre import RLRE
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.models import FineGrainedRetriever, LLMs, Retriever, LLMs_Ret_Paths, LLMs_Ret_Relations, LLMs_Ret_Triplets
from src.utils import collate_fn, set_seed, save_checkpoint, reload_best_model, adjust_learning_rate, setup_wp_optimizer, setup_tr_optimizer, write_log
from src.datasets import RetrievalDataset, RetrievalDatasetWithoutEmb
import sys
import random
sys.stdout.reconfigure(encoding='utf-8')
         
    
def detect(cfre, train_loader, log_dir, ):
    for batch in tqdm(train_loader):
        cfre.oracle_detection(batch, opj(log_dir, "logging.txt"))
    torch.save(cfre.evaluation, opj(log_dir, "evaluation.pth"))

        
def main():
    parser = argparse.ArgumentParser(description='CFRE')
    parser.add_argument('--dataset', type=str, default="webqsp", help='dataset used, option: ')
    parser.add_argument('--device', type=int, default=0, help='cuda device id, -1 for cpu')
    parser.add_argument('--llm_model_name_or_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default="./config/config.yaml", help='path of config file')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path of config file')
    parser.add_argument('--proj_name', type=str, default=None)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--retriever', type=str, default="PNA")
    parser.add_argument('--version', type=str, default="path")
    parser.add_argument('--coeff1', type=float, default=0.1)
    parser.add_argument('--coeff2', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--ret_num', type=int, default=100)
    parser.add_argument('--start', type=int, default=0, help="start of the training epoch.")
    parser.add_argument('--algo', type=str, default="v2")
    parser.add_argument('--penalty', type=float, default=0.16)
    parser.add_argument('--gumbel_strength', type=float, default=1)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(f"config/{args.dataset}_config.yaml", 'r'))
    config['algorithm']['coeff1'] = args.coeff1
    config['algorithm']['coeff2'] = args.coeff2
    config['algorithm']['ret_num'] = args.ret_num
    if args.llm_model_name_or_path:
        config['llms']["llm_model_name_or_path"] = config["logging"]["llm"] = args.llm_model_name_or_path
    train_config = config['train']
    llm_config = config['llms']
    warmup_config = train_config['warmup']
    algo_config = config['algorithm']
    log_config = config['logging']
    device = torch.device(f"cuda:{args.device}")
    
    if llm_config["tensor_parallel_size"] == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
        device = torch.device(f"cuda:0")

    proj_root = opj(log_config["root"]+"_detection", log_config["dataset"], log_config["llm"].split('/')[-1], log_config["ret"])
    log_dir = opj(proj_root, args.proj_name)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(config['env']['seed'])

    
    # llms_series = {
    #     "path": LLMs_Ret_Paths,
    #     "triplet": LLMs_Ret_Triplets,
    #     "relation": LLMs_Ret_Relations
    # }
    # llms = llms_series[args.version](llm_config)
    # print(args.version, args.ckpt_path)
    llms = LLMs_Ret_Paths(llm_config, async_version=False)
    
    train_set = RetrievalDatasetWithoutEmb(config=config["dataset"], split='train', )
    print(len(train_set))
    # train_set = [train_set[i] for i in range(int(len(train_set)/2), len(train_set))]
    train_loader = DataLoader(train_set, batch_size=4, shuffle=False, collate_fn=collate_fn, drop_last=False)

    ibtn = Retriever(config['retriever']).to(device)
    if args.ckpt_path:
        warmup_ckpt = torch.load(args.ckpt_path, map_location=device)
        ibtn.load_state_dict(warmup_ckpt['model_state_dict'])
    
    cfre = RLRE(retriever=ibtn, llm_model=llms, config=config['algorithm'])

    detect(cfre, train_loader, log_dir)
    
if __name__ == '__main__':
    main()
