import argparse
import os
import yaml
import torch
# import wandb
import numpy as np
from os.path import join as opj
from tqdm import tqdm
from cfre import CFRE
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.models import LLMs, FineGrainedRetriever
from src.utils import collate_fn, set_seed, save_checkpoint, reload_best_model, adjust_learning_rate, setup_wp_optimizer, setup_tr_optimizer, write_log
from src.datasets import RetrievalDataset


def inference(model, test_loader, log_dir):
    model.eval()
    model.set_eval()
    result = []
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            graph_batch, answer_batch, triplet_batch, triplet_batch_idx, relevant_idx_batch, question_batch, q_embd_batch = \
            batch["graph"], batch["y"], batch["triplets"], batch['triplet_batch_idx'], batch["relevant_idx"], batch["q"], batch["q_embd"]

            graph_batch, q_embd_batch, relevant_idx_batch = \
                graph_batch.to(model.device), q_embd_batch.to(model.device), relevant_idx_batch.to(model.device)

            attn_logtis, attns = model(graph_batch, triplet_batch_idx, q_embd_batch)
            result.append({
                "q": question_batch[0],
                "logit": attn_logtis,
                "attn": attns
            })
    assert len(test_loader) == len(result)
    torch.save(result, opj(log_dir, "inference.pth"))    
    print("Done")
            
        
def train(num_epochs, patience, cfre, train_loader, val_loader, optimizer, log_dir, warmup=True, **kwargs):
    print(kwargs)
    llm_frozen_epoch = kwargs.get("llm_frozen_epoch", None)
    best_val_loss = float('inf')
    loggings = opj(log_dir, "logging.txt")
    for epoch in tqdm(range(num_epochs)):
        if llm_frozen_epoch and epoch == llm_frozen_epoch:
            write_log(f"Freeze LLMs at epoch {epoch}", log_file=loggings)
            cfre.freeze_llm() 
                
        cfre.train()
        cfre.ibtn.set_train()
        epoch_loss, accum_loss, val_loss = 0., 0., 0.

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # loss, loss_dict = cfre.forward_pass(batch, epoch)
            loss = cfre.forward_pass(batch, epoch, warmup=warmup)
            loss.backward()

            # gradient and learning rate adjustment
            # only applied to retriever
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            # if (step + 1) % args.grad_steps == 0:
            #     adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            # if (step + 1) % args.grad_steps == 0:
            #     lr = optimizer.param_groups[0]["lr"]
            #     wandb.log({'Lr': lr})
            #     wandb.log({'Accum Loss': accum_loss / args.grad_steps})
            #     accum_loss = 0.

        # Average Loss per epoch
        write_log(f"Epoch: {epoch}|{num_epochs}: "
              f"Loss: {epoch_loss / len(train_loader)}", log_file=loggings
            #   f"Dir Loss: {dir_loss / len(train_loader)}"
            #   f"Supervisory Loss: {pred_loss / len(train_loader)}"
              )
        # wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        cfre.eval()
        cfre.ibtn.set_eval()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                loss = cfre.forward_pass(batch, warmup=warmup)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            write_log(f"Epoch: {epoch}|{num_epochs}: Val Loss: {val_loss}", loggings)
            # wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save fg retriever
            save_checkpoint(cfre.ibtn, epoch, log_dir)
            best_epoch = epoch

        write_log(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}', loggings)

        if epoch - best_epoch >= patience:
            write_log(f'Early stop at epoch {epoch}', loggings)
            break  


def main():
    parser = argparse.ArgumentParser(description='CFRE')
    parser.add_argument('--dataset', type=str, default="webqsp", help='dataset used, option: ')
    parser.add_argument('--device', type=int, default=0, help='cuda device id, -1 for cpu')
    parser.add_argument('--config_path', type=str, default="./config/config.yaml", help='path of config file')
    parser.add_argument('--proj_name', type=str, default="lora_w.o.gumbel")
    parser.add_argument('--mode', type=str, default="inference")
    parser.add_argument('--gnn', type=str, default="PNA")
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config_path, 'r'))
    train_config = config['train']
    warmup_config = train_config['warmup']
    algo_config = config['algorithm']
    log_config = config['logging']
    log_config["ret"] = args.gnn
    config["retriever"]["gnn"]["model_type"] = args.gnn
    proj_root = opj(log_config["root"], log_config["dataset"], log_config["llm"], log_config["ret"])
    log_dir = opj(proj_root, args.proj_name)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}")
    # wandb.init(project=f"",
    #            name=f"",
    #            config=config)

    set_seed(config['env']['seed'])
    train_set = RetrievalDataset(config=config["dataset"], split='train', )
    val_set = RetrievalDataset(config=config["dataset"], split='val', )
    test_set = RetrievalDataset(config=config["dataset"], split='test', )
    # max_degree = -1
    # for data in train_set:
    #     d = degree(data["graph"].edge_index[1], num_nodes=data["graph"].x.shape[0], dtype=torch.long)
    #     max_degree = max(max_degree, int(d.max()))

    # deg = torch.zeros(max_degree + 1, dtype=torch.long)
    # for data in train_set:
    #     d = degree(data["graph"].edge_index[1], num_nodes=data["graph"].x.shape[0], dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    
    # torch.save(deg, "pna_deg.pth")
    # input("Done.")
    # test_set = RetrievalDataset(config=config["dataset"], split='test', )
    # rel_triplets = []
    # for d in range(len(train_set)):
    #     rel_triplets.append(len(train_set[d]['relevant_idx']))
    # print(np.sort(rel_triplets).tolist())
    print(len(train_set), train_config['batch_size'])
    # if config['dataset']['random_split']:
    #     train_set, val_set, test_set = random_split(
    #         train_set, val_set, test_set, config['env']['seed'])

    train_loader = DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=train_config['batch_size'], collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn)
    # Build Model. Load ibtn, llms, cfre.

    ibtn = FineGrainedRetriever(config=config['retriever']['gnn'],
                                filtering_strategy=algo_config['filtering'],
                                filtering_num_or_ratio=algo_config['filtering_num_or_ratio'],
                                add_gumbel=algo_config['gumbel']
                                ).to(device)
    if args.proj_name != "warmup":
        wp_retriever = torch.load(opj(proj_root, "warmup", "best.pth"))["model"]
        ibtn.load_state_dict(wp_retriever)
    if args.mode == "inference" and os.path.exists(opj(log_dir, "best.pth")):
        print("Load Inference model..")
        wp_retriever = torch.load(opj(log_dir, "best.pth"))["model"]
        ibtn.load_state_dict(wp_retriever)
        inference(ibtn, test_loader, log_dir)
        exit(0)
    
    llms = LLMs(config['llms'])
    cfre = CFRE(fg_retriever=ibtn, llm_model=llms, config=config['algorithm']).to(device)
    trainable_params, all_param = cfre.trainable_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Set up Optimizer.
    wp_optimizer = setup_wp_optimizer(cfre, warmup_config)
    optimizer = setup_tr_optimizer(cfre, train_config)
        
    # Step 5. Training one epoch and batch
    # num_training_steps = args.num_epochs * len(train_loader)
    if args.proj_name == "warmup":
        train(warmup_config["num_epochs"], warmup_config["patience"], cfre, train_loader, val_loader, wp_optimizer, log_dir, warmup=True)
    else:
        train(train_config["num_epochs"], train_config["patience"], cfre, train_loader, val_loader, optimizer, log_dir, warmup=False, )
    # /home/comp/cscxliu/derek/CFRE/logging/webqsp/Llama-3.2-1B-Instruct/PNA/lora_gumbel

if __name__ == '__main__':
    main()
