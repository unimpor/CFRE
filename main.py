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
from src.models import FineGrainedRetriever, LLMs, Retriever
from src.utils import collate_fn, set_seed, save_checkpoint, reload_best_model, adjust_learning_rate, setup_wp_optimizer, setup_tr_optimizer, write_log
from src.datasets import RetrievalDataset
import sys
sys.stdout.reconfigure(encoding='utf-8')


def inference(model, test_loader, log_dir):
    model.eval()
    result = {}
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch, id_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"], batch["id"]

            graph_batch, q_embd_batch = graph_batch.to(model.device), q_embd_batch.to(model.device)

            attn_logtis = model(graph_batch, q_embd_batch)
            assert len(attn_logtis) == len(triplet_batch[0])
            print(attn_logtis)
            result[id_batch[0]] = {
                "q": question_batch[0],
                "logit": attn_logtis,
                "triplets": triplet_batch[0]
            }
    assert len(test_loader) == len(result)
    torch.save(result, opj(log_dir, "inference.pth"))    
    print("Done")
            
        
def train(num_epochs, patience, cfre, train_loader, val_loader, optimizer, log_dir, warmup=True, **kwargs):
    best_val_signal = -1.
    loggings = opj(log_dir, "logging.txt")
    for epoch in tqdm(range(0, num_epochs)):        
        cfre.train()
        cfre.baseline_cache = {}  # set empty to baseline cache at the start of every epoch.
        cfre.update_num = 0
        epoch_loss, accum_loss, val_loss = 0., 0., 0.
        all_loss_dict, all_loss_dict_val = {}, {}
        if epoch == 0:
            for _, batch in enumerate(train_loader):
                cfre.pre_processing_v2(batch, epoch)
            torch.save(cfre.baseline, opj(log_dir, "baseline.pth"))
            continue
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # loss, loss_dict = cfre.forward_pass(batch, epoch)
            loss, loss_dict = cfre.forward_pass(batch, epoch, warmup=warmup, training=True)
            loss.backward()
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
            # clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()

        train_loss = epoch_loss / len(train_loader)
        for k, v in all_loss_dict.items():
            all_loss_dict[k] = v / len(train_loader)
        write_log(f"Epoch: {epoch}|{num_epochs}. Train Loss: {train_loss}" + str(all_loss_dict), loggings)

        cfre.eval()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                _, loss_dict = cfre.forward_pass(batch, epoch, warmup=warmup, training=False)
                # val_loss += loss.item()
                for k, v in loss_dict.items():
                    all_loss_dict_val[k] = all_loss_dict_val.get(k, 0) + v
            
            # val_loss = val_loss / len(val_loader)
            for k, v in all_loss_dict_val.items():
                all_loss_dict_val[k] = v / len(val_loader)
            write_log(f"Epoch: {epoch}|{num_epochs}. Val Loss: {val_loss}" + str(all_loss_dict_val), loggings)
            
            # wandb.log({'Val Loss': val_loss})
        
        if all_loss_dict_val[cfre.metrics_name] > best_val_signal:
            best_val_signal = all_loss_dict_val[cfre.metrics_name]
            # save fg retriever
            torch.save(cfre.evaluation, opj(log_dir, "evaluation.pth"))
            save_checkpoint(cfre.retriever, epoch, log_dir)
            best_epoch = epoch
        
        # reference update. Deprecated right now.
        # if (epoch + 1) % 4 == 0:
        #     update_num = cfre.update_baseline(train_loader)
        #     torch.save(cfre.baseline, opj(log_dir, "baseline.pth"))
        #     write_log(f"Epoch: {epoch}|{num_epochs}. Update {update_num} training samples to better.", loggings)

            # if best_val_signal > 0.705:
            #     cfre.baseline = cfre.baseline_cache  # update baseline to moving baseline
            #     write_log(f'Epoch {epoch} Update Baseline!', loggings)
        
        write_log(f'Epoch {epoch} Val Loss {val_loss} Best Val signal {best_val_signal} Best Epoch {best_epoch}', loggings)
        write_log(f"Epoch: {epoch}|{num_epochs}. Update {cfre.update_num} training samples to better.", loggings)
        
        if epoch - best_epoch >= patience:
            write_log(f'Early stop at epoch {epoch}', loggings)
            break  


def main():
    parser = argparse.ArgumentParser(description='CFRE')
    parser.add_argument('--dataset', type=str, default="webqsp", help='dataset used, option: ')
    parser.add_argument('--device', type=int, default=0, help='cuda device id, -1 for cpu')
    parser.add_argument('--config_path', type=str, default="./config/config.yaml", help='path of config file')
    parser.add_argument('--ckpt_path', type=str, default="datasets/webqsp/checkpoints/warmup.pth", help='path of config file')
    parser.add_argument('--proj_name', type=str, default=None)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--retriever', type=str, default="PNA")

    parser.add_argument('--coeff1', type=float, default=0.1)
    parser.add_argument('--coeff2', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--ret_num', type=int, default=1)
    parser.add_argument('--algo', type=str, default="v2")
    parser.add_argument('--gumbel_strength', type=float, default=1)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config_path, 'r'))
    config['algorithm']['coeff1'] = args.coeff1
    config['algorithm']['coeff2'] = args.coeff2
    config['algorithm']['tau'] = args.tau
    config['algorithm']['ret_num'] = args.ret_num
    config['algorithm']['algo'] = args.algo
    config['algorithm']['gumbel_strength'] = args.gumbel_strength
    
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
    
    # wandb.init(project=f"",
    #            name=f"",
    #            config=config)

    set_seed(config['env']['seed'])

    llms = LLMs(llm_config)

    train_set = RetrievalDataset(config=config["dataset"], split='train', )
    val_set = RetrievalDataset(config=config["dataset"], split='val', )

    # Deprecated in Dec. 27. -- Filtering out samples which Warmup does good
    # reference = torch.load("/home/comp/cscxliu/derek/CFRE/datasets/webqsp/checkpoints/reference.pth")
    # refined_train_set = []
    # for data in train_set:
    #     data["recall"] = reference[data["id"]]["recall"]
    #     if reference[data["id"]]["recall"] < 1.0:
    #         refined_train_set.append(data)
    # refined_train_set = sorted(refined_train_set, key=lambda x: x["recall"], reverse=False)

    # def find_ranking(a, indices):
    #     sorted_indices = torch.argsort(a, descending=True)
    #     return [torch.where(sorted_indices == i)[0].item() for i in indices]
    # for dat in refined_train_set:
    #     print(dat["recall"], len(dat["relevant_idx"]), len(dat["triplets"]))    
    # input(0)
    # for dat in refined_train_set:
    #     dat_id = dat["id"]
    #     target_indices = dat["relevant_idx"]
    #     prob = reference[dat_id]["logits"]
    #     print(find_ranking(prob, target_indices), len(target_indices), reference[dat_id]["recall"])
    #     input(0)
    train_loader = DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn)
        
    ibtn = Retriever(config['retriever']).to(device)
    if args.proj_name != "warmup":
        warmup_ckpt = torch.load(args.ckpt_path, map_location=device)
        ibtn.load_state_dict(warmup_ckpt['model_state_dict'])
    
    if args.mode == "inference":
        test_set = RetrievalDataset(config=config["dataset"], split='test', )
        test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn)
        
        ibtn.load_state_dict(torch.load(opj(log_dir, "best.pth"))["model_state_dict"])
        inference(ibtn, test_loader, log_dir)
        exit(0)

    cfre = RLRE(retriever=ibtn, llm_model=llms, config=config['algorithm'])

    # Set up Optimizer.
    wp_optimizer = setup_wp_optimizer(cfre, warmup_config)
    optimizer = setup_tr_optimizer(cfre, train_config)   
    
    # Step 5. Training one epoch and batch
    # num_training_steps = args.num_epochs * len(train_loader)
    if args.proj_name == "warmup":
        train(warmup_config["num_epochs"], warmup_config["patience"], cfre, train_loader, val_loader, wp_optimizer, log_dir, warmup=True)
    else:
        train(train_config["num_epochs"], train_config["patience"], cfre, train_loader, val_loader, optimizer, log_dir, warmup=False)
    
if __name__ == '__main__':
    main()
