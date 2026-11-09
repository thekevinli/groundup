import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from collectives import all_gather, all_reduce, init, launch, reduce_scatter


# The four SP boundaries along the sequence (dim 1). One idiom throughout:
#   a - a.detach() + b   ->  forward value b (our collective, no autograd), gradient flows into a
# and the backward communication is a hook that runs our all_reduce on the gradient.
def gather_seq(x, group):    # column input; CP's K/V: my S/P tokens -> all S
    P, r = dist.get_world_size(group), dist.get_rank(group)
    padded = torch.cat([x if i == r else torch.zeros_like(x) for i in range(P)], dim=1)
    if padded.requires_grad:
        padded.register_hook(lambda g: all_reduce(g, group))   # backward: sum every rank's grad for my tokens
    return padded - padded.detach() + all_gather(x.detach(), 1, group)


def scatter_seq(y, group):   # row output: partial sums over all S -> the full sum for my S/P tokens
    P, r = dist.get_world_size(group), dist.get_rank(group)
    if y.requires_grad:
        y.register_hook(lambda g: all_reduce(g, group))        # backward: my zero-padded grad -> everyone's grads
    mine = y.chunk(P, dim=1)[r]
    return mine - mine.detach() + reduce_scatter(y.detach(), 1, group)


def enter_sp(x, group):      # after the embedding: keep my S/P tokens
    P, r = dist.get_world_size(group), dist.get_rank(group)
    if x.requires_grad:
        x = x.view_as(x)
        x.register_hook(lambda g: all_reduce(g, group))        # backward: my zero-padded grad -> the full grad
    return x.chunk(P, dim=1)[r]


def exit_sp(x, group):       # before the head: my S/P tokens -> all S (backward: keep my slice of the grad)
    P, r = dist.get_world_size(group), dist.get_rank(group)
    padded = torch.cat([x if i == r else torch.zeros_like(x) for i in range(P)], dim=1)
    return padded - padded.detach() + all_gather(x.detach(), 1, group)


def column_parallel(x, w_local, b_local, group, sp=False):
    # I own out/P output features. TP: the same full x on every rank. SP: my S/P tokens, gathered here
    if group is None:
        return F.linear(x, w_local, b_local)                   # one GPU
    if sp:
        x = gather_seq(x, group)                               # its backward already sums every rank's grad
    elif x.requires_grad:
        x = x.view_as(x)                                       # own node: the hook sees only this layer's dL/dx
        x.register_hook(lambda g: all_reduce(g, group))        # backward: sum every rank's partial dL/dx
    return F.linear(x, w_local, b_local)


def row_parallel(x_local, w_local, bias, group, sp=False):
    # x_local = my in/P features (the column layer's output: nothing to slice); outputs are partial sums.
    # TP: all-reduce them. SP: reduce-scatter to my S/P tokens (same bytes, 1/P of the activations)
    y = F.linear(x_local, w_local)
    if group is None:
        return y + bias
    if sp:
        y = scatter_seq(y, group)
    else:
        y = y - y.detach() + all_reduce(y.detach(), group)     # forward: the sum; backward: identity
    return y + bias                                            # bias once, after the sum (SP: mark it, see below)


def sync_sp_grads(model: nn.Module, group):
    # parameters used in the SP region (norms, row-parallel biases) saw only my slice of the
    # sequence: sum their gradients over the TP group. Megatron marks them the same way.
    for p in model.parameters():
        if getattr(p, "sequence_parallel", False) and p.grad is not None:
            p.grad = all_reduce(p.grad, group)


def _test(rank: int, tp: int):
    init(rank, tp, 29502)
    torch.manual_seed(0)                                       # same full MLP and input everywhere
    full = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16))
    x = torch.randn(2, 8, 16, requires_grad=True)
    (full(x) ** 2).sum().backward()

    group = dist.group.WORLD                                   # all ranks form the TP group here
    h = slice(rank * 64 // tp, (rank + 1) * 64 // tp)          # my hidden units
    for sp in (False, True):
        w1, b1 = full[0].weight[h].detach().clone().requires_grad_(), full[0].bias[h].detach().clone()
        w2 = full[2].weight[:, h].detach().clone().requires_grad_()
        b2 = nn.Parameter(full[2].bias.detach().clone())
        b2.sequence_parallel = sp                              # added on my slice of S under SP
        mine = (lambda t: t.chunk(tp, dim=1)[rank]) if sp else (lambda t: t)   # SP: my slice of S
        xs = mine(x.detach()).clone().requires_grad_()
        out = row_parallel(F.gelu(column_parallel(xs, w1, b1, group, sp)), w2, b2, group, sp)
        (out ** 2).sum().backward()
        if sp:
            b2.grad = all_reduce(b2.grad, group)               # sync_sp_grads: without it the bias grad is 1/tp short
        assert torch.allclose(out, mine(full(x)), atol=1e-5)
        assert torch.allclose(xs.grad, mine(x.grad), atol=1e-5)
        assert torch.allclose(w1.grad, full[0].weight.grad[h], atol=1e-5)
        assert torch.allclose(w2.grad, full[2].weight.grad[:, h], atol=1e-5)
        assert torch.allclose(b2.grad, full[2].bias.grad, atol=1e-5)
        if rank == 0:
            print(("SP" if sp else "TP"), "MLP (column -> GELU -> row) matches the single-GPU MLP")
    dist.destroy_process_group()


if __name__ == "__main__":
    launch(_test, 2)
