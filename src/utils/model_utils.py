import os
import torch
from os.path import join as opj


def save_checkpoint(model, cur_epoch, log_dir):
    """
    Save the checkpoint at the current epoch.
    """

    os.makedirs(log_dir, exist_ok=True)

    state_dict = model.state_dict()

    save_obj = {
        "model": state_dict,
        "epoch": cur_epoch,
    }
    path = opj(log_dir, f"best.pth")
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)


def reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_best.pth'

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
