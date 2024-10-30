import argparse
import yaml
import torch
# import wandb
from tqdm import tqdm
from cfre import CFRE
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.models import LLMs, FineGrainedRetriever
from src.utils import collate_fn, set_seed, save_checkpoint, reload_best_model, adjust_learning_rate
from src.datasets import RetrievalDataset


def main():
    parser = argparse.ArgumentParser(description='CFRE')
    parser.add_argument('--dataset', type=str, default="webqsp", help='dataset used, option: ')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device id, -1 for cpu')
    parser.add_argument('--config_path', type=str, default="./config/config.yaml", help='path of config file')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    train_config = config['train']
    algo_config = config['algorithm']
    # wandb.init(project=f"",
    #            name=f"",
    #            config=config)

    set_seed(config['env']['seed'])
    # train_set = RetrievalDataset(config=config["dataset"], split='train', )
    val_set = RetrievalDataset(config=config["dataset"], split='val', )
    # test_set = RetrievalDataset(config=config["dataset"], split='test', )
    print(len(val_set))

    # if config['dataset']['random_split']:
    #     train_set, val_set, test_set = random_split(
    #         train_set, val_set, test_set, config['env']['seed'])

    # train_loader = DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=train_config['batch_size'], collate_fn=collate_fn)
    # test_loader = DataLoader(test_set, batch_size=train_config['batch_size'], collate_fn=collate_fn)
    # Build Model. Load ibtn, llms, cfre.
    ibtn = FineGrainedRetriever(config=config['retriever']['gnn'],
                                filtering_strategy=algo_config['filtering'],
                                filtering_num_or_ratio=algo_config['filtering_num_or_ratio']
                                )
    llms = LLMs(config['llms'])
    cfre = CFRE(fg_retriever=ibtn, llm_model=llms, args=config['algorithm'])
    trainable_params, all_param = cfre.trainable_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


    # Set up Optimizer.
    params = [p for _, p in cfre.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': train_config["lr"], 'weight_decay': train_config["wd"]}, ],
        betas=(0.9, 0.95)
    )

    # Step 5. Training one epoch and batch

    num_training_steps = args.num_epochs * len(train_loader)
    best_val_loss = float('inf')

    for epoch in tqdm(range(args.num_epochs)):

        cfre.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, loss_dict = cfre.forward_pass(batch)
            loss.backward()

            # gradient and learning rate adjustment
            # from G-Retriever
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            # if (step + 1) % args.grad_steps == 0:
            #     lr = optimizer.param_groups[0]["lr"]
            #     wandb.log({'Lr': lr})
            #     wandb.log({'Accum Loss': accum_loss / args.grad_steps})
            #     accum_loss = 0.

        # Average Loss per epoch
        print(f"Epoch: {epoch}|{args.num_epochs}: "
              f"Train Loss: {epoch_loss / len(train_loader)}"
              f"Supervisory Loss: ")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        val_loss, best_epoch = 0., 0
        cfre.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = cfre(batch)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save fg retriever
            save_checkpoint(cfre.ibtn, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break
    # Evaluation: 1) retriever performance 2) overall performance


if __name__ == '__main__':
    main()
