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


def inference(cfre, test_loader, log_dir):
    loggings = opj(log_dir, "logging.txt")
    K = cfre.K
    all_loss_dict_val = {}
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            _, loss_dict = cfre.inference(batch)
            for k, v in loss_dict.items():
                all_loss_dict_val[k] = all_loss_dict_val.get(k, []) + v
        
        for k, v in all_loss_dict_val.items():
            all_loss_dict_val[k] = np.mean(v)
        
    write_log(f"Test-Inference" + str(all_loss_dict_val), loggings)
    torch.save(cfre.evaluation, opj(log_dir, f"evaluation_K{K}.pth"))
                
    
def train(num_epochs, patience, cfre, train_loader, val_loader, optimizer, log_dir, warmup=True, **kwargs):
    start = kwargs.get("start", 0)
    # best_val_signal = -1
    best_val_signal = {"recall": -1, "step": -1}
    loggings = opj(log_dir, "logging.txt")
    K = cfre.K
    for epoch in tqdm(range(start, num_epochs)):      
        cfre.train()
        epoch_loss, val_loss = 0., 0., 0.
        all_loss_dict, all_loss_dict_val = {}, {}
        # ========= Oracle subset detection ========= 
        # if epoch == 0:
        #     for _, batch in enumerate(train_loader):
        #         cfre.forward_pass_(batch)
        #     torch.save(cfre.evaluation, opj(log_dir, "evaluation.pth"))
        #     exit(0)
        
        # ========= Training ========= 
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, loss_dict = cfre.forward_pass(batch, epoch, training=True)
            loss.backward()
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()

        train_loss = epoch_loss / len(train_loader)
        for k, v in all_loss_dict.items():
            all_loss_dict[k] = v / len(train_loader)
        write_log(f"Epoch: {epoch}|{num_epochs}. Train Loss: {train_loss}" + str(all_loss_dict), loggings)
        
        # ========= Vat set Evaluation ========= 
        cfre.eval()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                _, loss_dict = cfre.forward_pass(batch, epoch, training=False)
                for k, v in loss_dict.items():
                    all_loss_dict_val[k] = all_loss_dict_val.get(k, 0) + v
            
            for k, v in all_loss_dict_val.items():
                all_loss_dict_val[k] = v / len(val_loader)
        write_log(f"Epoch: {epoch}|{num_epochs}. Val Loss: {val_loss}" + str(all_loss_dict_val), loggings)
  
        # ========= Save checkpoint & Loggings ========= 
        for item in best_val_signal.keys():
            if all_loss_dict_val[item] > best_val_signal[item]:
                best_val_signal[item] = all_loss_dict_val[item]
                save_checkpoint(cfre.retriever, epoch, log_dir, filename=f"best-{item}.pth")
                best_epoch = epoch
                write_log(f'Update {item} at Epoch {epoch}', loggings)

        # write_log(f'Epoch {epoch} Val Loss {val_loss} Best Val signal {best_val_signal} Best Epoch {best_epoch}', loggings)
        
        if epoch - best_epoch >= patience:
            write_log(f'Early stop at epoch {epoch}', loggings)
            break

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
    parser.add_argument('--gumbel_strength', type=float, default=1)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(f"config/{args.dataset}_config.yaml", 'r'))
    config['algorithm']['coeff1'] = args.coeff1
    config['algorithm']['coeff2'] = args.coeff2
    config['algorithm']['tau'] = args.tau
    config['algorithm']['ret_num'] = args.ret_num
    config['algorithm']['algo'] = args.algo
    config['algorithm']['gumbel_strength'] = args.gumbel_strength
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

    proj_root = opj(log_config["root"], log_config["dataset"], log_config["llm"].split('/')[-1], log_config["ret"])
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

    # to_preserve = []
    # to_check = torch.load("logging/cwq/Meta-Llama-3.1-8B-Instruct/DDE/warmup_training_relation/evaluation_K50.pth")
    # for k, v in to_check.items():
    #     gen = v['gen']
    #     if 'ans:' not in gen.lower() or "ans: not available" in gen.lower() or "ans: no information available" in gen.lower():
    #         to_preserve.append(k)
    if args.mode == "inference":
        test_set = RetrievalDataset(config=config["dataset"], split='test', )
        test_loader = DataLoader(test_set, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)
    else:
        train_set = RetrievalDataset(config=config["dataset"], split='train', )
        val_set = RetrievalDataset(config=config["dataset"], split='val', )
        # TODO: False to True
        train_loader = DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)

    ibtn = Retriever(config['retriever']).to(device)
    if args.ckpt_path:
        warmup_ckpt = torch.load(args.ckpt_path, map_location=device)
        ibtn.load_state_dict(warmup_ckpt['model_state_dict'])
    
    llms = LLMs(llm_config)
    cfre = RLRE(retriever=ibtn, llm_model=llms, config=config['algorithm'])

    if args.mode == "inference":
        inference(cfre, test_loader, log_dir)
        exit(0)

    # Set up Optimizer.
    wp_optimizer = setup_wp_optimizer(cfre, warmup_config)
    optimizer = setup_tr_optimizer(cfre, train_config)   
    
    # Step 5. Training one epoch and batch
    if args.proj_name == "warmup":
        train(warmup_config["num_epochs"], warmup_config["patience"], cfre, train_loader, val_loader, wp_optimizer, log_dir, warmup=True)
    else:
        train(train_config["num_epochs"], train_config["patience"], cfre, train_loader, val_loader, optimizer, log_dir, warmup=False, start=args.start)
    
if __name__ == '__main__':
    main()
