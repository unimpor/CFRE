import argparse
import os
import yaml
import torch
# import wandb
import numpy as np
from os.path import join as opj
from tqdm import tqdm
from cfre import CFRE
from rlre import RLRE
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.models import FineGrainedRetriever, LLMs
from src.utils import collate_fn, set_seed, save_checkpoint, reload_best_model, adjust_learning_rate, setup_wp_optimizer, setup_tr_optimizer, write_log
from src.datasets import RetrievalDataset
import sys
sys.stdout.reconfigure(encoding='utf-8')


def inference(model, test_loader, log_dir):
    model.eval()
    model.set_eval()
    result = []
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"]

            graph_batch, q_embd_batch, relevant_idx_batch = \
                graph_batch.to(model.device), q_embd_batch.to(model.device), relevant_idx_batch.to(model.device)

            _, _, _, _, attn_logtis = model(graph_batch, triplet_batch_idx, q_embd_batch, 0)
            result.append({
                "q": question_batch[0],
                "logit": attn_logtis[0],
            })
    assert len(test_loader) == len(result)
    torch.save(result, opj(log_dir, "inference.pth"))    
    print("Done")
            
        
def train(num_epochs, patience, cfre, train_loader, val_loader, optimizer, log_dir, warmup=True, **kwargs):
    best_val_signal = -1.
    loggings = opj(log_dir, "logging.txt")
    for epoch in tqdm(range(num_epochs)):        
        cfre.train()
        cfre.ibtn.set_train()
        cfre.baseline_cache = {}  # set empty to baseline cache at the start of every epoch.
        epoch_loss, accum_loss, val_loss = 0., 0., 0.
        all_loss_dict, all_loss_dict_val = {}, {}
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # loss, loss_dict = cfre.forward_pass(batch, epoch)
            loss, loss_dict = cfre.forward_pass(batch, epoch, warmup=warmup, training=True)
            loss.backward()
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()

        train_loss = epoch_loss / len(train_loader)
        for k, v in all_loss_dict.items():
            all_loss_dict[k] = v / len(train_loader)
        write_log(f"Epoch: {epoch}|{num_epochs}. Train Loss: {train_loss}" + str(all_loss_dict), loggings)

        cfre.eval()
        cfre.ibtn.set_eval()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                loss, loss_dict = cfre.forward_pass(batch, epoch, warmup=warmup, training=False)
                val_loss += loss.item()
                for k, v in loss_dict.items():
                    all_loss_dict_val[k] = all_loss_dict_val.get(k, 0) + v
            
            val_loss = val_loss / len(val_loader)
            for k, v in all_loss_dict_val.items():
                all_loss_dict_val[k] = v / len(val_loader)
            write_log(f"Epoch: {epoch}|{num_epochs}. Val Loss: {val_loss}" + str(all_loss_dict_val), loggings)
            # wandb.log({'Val Loss': val_loss})

        if all_loss_dict_val[cfre.metrics_name] > best_val_signal:
            best_val_signal = all_loss_dict_val[cfre.metrics_name]
            # save fg retriever
            save_checkpoint(cfre.ibtn, epoch, log_dir)
            best_epoch = epoch
            # if epoch > 0:
            #     cfre.baseline = cfre.baseline_cache  # update baseline to moving baseline
        
        write_log(f'Epoch {epoch} Val Loss {val_loss} Best Val signal {best_val_signal} Best Epoch {best_epoch}', loggings)

        if epoch - best_epoch >= patience:
            write_log(f'Early stop at epoch {epoch}', loggings)
            save_checkpoint(cfre.ibtn, epoch, log_dir, filename="final.pth")
            break  


def main():
    parser = argparse.ArgumentParser(description='CFRE')
    parser.add_argument('--dataset', type=str, default="webqsp", help='dataset used, option: ')
    parser.add_argument('--device', type=int, default=0, help='cuda device id, -1 for cpu')
    parser.add_argument('--config_path', type=str, default="./config/config.yaml", help='path of config file')
    parser.add_argument('--proj_name', type=str, default="lora_w.o.gumbel")
    parser.add_argument('--mode', type=str, default="inference")
    parser.add_argument('--gnn', type=str, default="PNA")
    parser.add_argument('--coeff1', type=float, default=0.1)
    parser.add_argument('--coeff2', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--llm_frozen_epoch', type=int, default=None)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config_path, 'r'))
    config['algorithm']['coeff1'] = args.coeff1
    config['algorithm']['coeff2'] = args.coeff2
    config['algorithm']['tau'] = args.tau
    train_config = config['train']
    llm_config = config['llms']
    warmup_config = train_config['warmup']
    algo_config = config['algorithm']
    log_config = config['logging']
    device = torch.device(f"cuda:{args.device}")
    
    if llm_config["tensor_parallel_size"] == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
        device = torch.device(f"cuda:0")
    # log_config["ret"] = args.gnn
    # config["retriever"]["gnn"]["model_type"] = args.gnn
    # if args.gnn == "graphsage":
    #     config["retriever"]["gnn"]["num_layers"] = 5
    proj_root = opj(log_config["root"], log_config["dataset"], log_config["llm"].split('/')[-1], log_config["ret"])
    log_dir = opj(proj_root, args.proj_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # wandb.init(project=f"",
    #            name=f"",
    #            config=config)

    set_seed(config['env']['seed'])
    train_set = RetrievalDataset(config=config["dataset"], split='train', )
    val_set = RetrievalDataset(config=config["dataset"], split='val', )
    test_set = RetrievalDataset(config=config["dataset"], split='test', )

    print(len(train_set), train_config['batch_size'])
    # if config['dataset']['random_split']:
    #     train_set, val_set, test_set = random_split(
    #         train_set, val_set, test_set, config['env']['seed'])

    train_loader = DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=train_config['batch_size'], collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn)
    # Build Model. Load ibtn, llms, cfre.

    ibtn = FineGrainedRetriever(config['retriever']['gnn'], algo_config).to(device)
    if args.proj_name != "warmup":
        wp_retriever = torch.load("./logging/webqsp/Llama-3.2-1B-Instruct/PNA/warmup/best.pth", map_location=device)["model"]
        ibtn.load_state_dict(wp_retriever)
    # if args.mode == "inference" and os.path.exists(opj(log_dir, "best.pth")):
    #     print("Load Inference model..")
    #     wp_retriever = torch.load(opj(log_dir, "best.pth"))["model"]
    #     ibtn.load_state_dict(wp_retriever)
    
    llms = LLMs(llm_config)
    cfre = RLRE(fg_retriever=ibtn, llm_model=llms, config=config['algorithm'])

    # Set up Optimizer.
    wp_optimizer = setup_wp_optimizer(cfre, warmup_config)
    optimizer = setup_tr_optimizer(cfre, train_config)
        
    # Step 5. Training one epoch and batch
    # num_training_steps = args.num_epochs * len(train_loader)
    if args.proj_name == "warmup":
        train(warmup_config["num_epochs"], warmup_config["patience"], cfre, train_loader, val_loader, wp_optimizer, log_dir, warmup=True)
    else:
        train(train_config["num_epochs"], train_config["patience"], cfre, train_loader, val_loader, optimizer, log_dir, warmup=False)
    # /home/comp/cscxliu/derek/CFRE/logging/webqsp/Llama-3.2-1B-Instruct/PNA/lora_gumbel

    ibtn.load_state_dict(torch.load(opj(log_dir, "best.pth"))["model"])
    inference(ibtn, test_loader, log_dir)
    
if __name__ == '__main__':
    main()
