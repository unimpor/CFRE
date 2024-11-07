import torch


def setup_wp_optimizer(cfre, config):
    params = [p for name, p in cfre.named_parameters() if p.requires_grad and "ibtn" in name]
    return torch.optim.AdamW(
        [{'params': params, 'lr': float(config["lr"]), 'weight_decay': config["wd"]}, ],
        betas=(0.9, 0.95)
    )


def setup_tr_optimizer(cfre, config):
    ibtn_params = [p for name, p in cfre.named_parameters() if p.requires_grad and "ibtn" in name]
    llm_params = [p for name, p in cfre.named_parameters() if p.requires_grad and "llm" in name]

    return torch.optim.AdamW(
        [
            {'params': ibtn_params, 'lr': float(config["lr_ret"]), 'weight_decay': config["wd_ret"]},
            {'params': llm_params, 'lr': float(config["lr_llm"]), 'weight_decay': config["wd_llm"]},
        ],
        betas=(0.9, 0.95)
    )