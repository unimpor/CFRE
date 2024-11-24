from typing import Callable, List, Optional, Tuple, Union
import torch
from torch.overrides import has_torch_function_unary, handle_torch_function
Tensor = torch.Tensor


def gumbel_topk(logits: Tensor, K: int, tau: float = 2, mode: str = "st", eps: float = 1e-10, dim: int = -1, add_grumbel=True, eta=1.0) -> Tensor:
    """
    Adapted this function from torch.nn.functional.gumbel_softmax.
    See https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#torch.nn.functional.gumbel_softmax
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(gumbel_topk, (logits,), logits, k=K, tau=tau, mode=mode, eps=eps, dim=dim, add_grumbel=add_grumbel, eta=eta)

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + eta * gumbels) / tau if add_grumbel else logits / tau
    # ~Gumbel(logits,tau)
    y_soft, topk_indices = gumbels.softmax(dim), None

    if mode == "st":
        # Straight through.
        # note topk_indices also sort y_soft from large to small
        _, topk_indices = y_soft.topk(K, dim=dim, largest=True, sorted=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, topk_indices, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    elif mode == "hard":
        _, topk_indices = y_soft.topk(K, dim=dim, largest=True, sorted=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, topk_indices, 1.0)
        ret = y_hard
    elif mode == "soft":
        # Reparametrization trick.
        ret = y_soft
    else:
        raise NotImplementedError
    
    return ret, topk_indices


if __name__ == '__main__':

    logits = torch.tensor([[0.3], [0.5], [2.6], [3.3]])
    print(gumbel_topk(logits, K=2, hard=True, dim=0))
