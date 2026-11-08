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


class ColumnParallelLinear(nn.Linear):
    # my 1/tp of the OUTPUT features. Input: the same full x on every rank (TP), or
    # my slice of the sequence (SP), all-gathered here.
    def __init__(self, in_features: int, out_features: int, group=None, sequence_parallel: bool = False):
        self.group, self.tp, self.sp = group, dist.get_world_size(group) if group else 1, sequence_parallel
        super().__init__(in_features, out_features // self.tp)

    def forward(self, x):
        if self.tp > 1 and self.sp:
            x = gather_seq(x, self.group)
        elif self.tp > 1 and x.requires_grad:
            # every rank used the same x, so dL/dx is the sum of every rank's piece: all-reduce it
            x = x.view_as(x)                                   # own node: the hook sees only this layer's dL/dx
            x.register_hook(lambda grad: all_reduce(grad, self.group))
        return super().forward(x)


class RowParallelLinear(nn.Linear):
    # my 1/tp of the INPUT features, so every rank's output is a partial sum: all-reduce it (TP),
    # or reduce-scatter it to my slice of the sequence (SP). Same bytes, 1/tp of the activations.
    # x arrives already split by features: it is the column layer's output, so nothing to slice or concat.
    def __init__(self, in_features: int, out_features: int, group=None, sequence_parallel: bool = False):
        self.group, self.tp, self.sp = group, dist.get_world_size(group) if group else 1, sequence_parallel
        super().__init__(in_features // self.tp, out_features)
        self.bias.sequence_parallel = sequence_parallel        # added in the SP region: see sync_sp_grads

    def forward(self, x):
        y = F.linear(x, self.weight)
        if self.tp > 1 and self.sp:
            y = scatter_seq(y, self.group)
        elif self.tp > 1:
            y = y - y.detach() + all_reduce(y.detach(), self.group)   # forward: the sum; backward: identity
        return y + self.bias                                   # added once, after the sum


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
        mlp = nn.Sequential(ColumnParallelLinear(16, 64, group, sp), nn.GELU(), RowParallelLinear(64, 16, group, sp))
        with torch.no_grad():
            mlp[0].weight.copy_(full[0].weight[h]); mlp[0].bias.copy_(full[0].bias[h])
            mlp[2].weight.copy_(full[2].weight[:, h]); mlp[2].bias.copy_(full[2].bias)
        mine = (lambda t: t.chunk(tp, dim=1)[rank]) if sp else (lambda t: t)   # SP: my slice of S
        xs = mine(x.detach()).clone().requires_grad_()
        out = mlp(xs)
        (out ** 2).sum().backward()
        if sp:
            sync_sp_grads(mlp, group)                          # without it the bias grad is 1/tp short
        assert torch.allclose(out, mine(full(x)), atol=1e-5)
        assert torch.allclose(xs.grad, mine(x.grad), atol=1e-5)
        assert torch.allclose(mlp[0].weight.grad, full[0].weight.grad[h], atol=1e-5)
        assert torch.allclose(mlp[2].weight.grad, full[2].weight.grad[:, h], atol=1e-5)
        assert torch.allclose(mlp[2].bias.grad, full[2].bias.grad, atol=1e-5)
        if rank == 0:
            print(("SP" if sp else "TP"), "MLP (column -> GELU -> row) matches the single-GPU MLP")
    dist.destroy_process_group()


if __name__ == "__main__":
    launch(_test, 2)
