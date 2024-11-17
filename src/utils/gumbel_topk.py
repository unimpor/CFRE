from typing import Callable, List, Optional, Tuple, Union
import torch
from torch.overrides import has_torch_function_unary, handle_torch_function
Tensor = torch.Tensor


def gumbel_topk(logits: Tensor, K: int, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, add_grumbel=True) -> Tensor:
    """
    Adapted this function from torch.nn.functional.gumbel_softmax.
    See https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#torch.nn.functional.gumbel_softmax
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(gumbel_topk, (logits,), logits, k=K, tau=tau, hard=hard, eps=eps, dim=dim)

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau if add_grumbel else logits / tau
    # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        # note topk_indices also sort y_soft from large to small
        _, topk_indices = y_soft.topk(K, dim=dim, largest=True, sorted=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, topk_indices, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, topk_indices


if __name__ == '__main__':

    logits = torch.tensor([[0.3], [0.5], [2.6], [3.3]])
    print(gumbel_topk(logits, K=2, hard=True, dim=0))
