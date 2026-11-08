import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from collectives import all_gather, all_reduce, init, launch, ranks, reduce_scatter


class Comm(torch.autograd.Function):
    # a collective whose backward we pick by hand: SP communicates in BOTH directions, and the
    # backward is always the forward's adjoint (all-gather <-> reduce-scatter, keep my slice <-> all-gather)
    @staticmethod
    def forward(ctx, x, fwd, bwd):
        ctx.bwd = bwd
        return fwd(x)

    @staticmethod
    def backward(ctx, grad):
        return ctx.bwd(grad), None, None


def my_slice(t, group):
    rank, world, _ = ranks(group)
    return t.chunk(world, dim=1)[rank].contiguous()


# the four SP boundaries along the sequence (dim 1), each a (forward, backward) pair
def gather_seq(x, group):    # column input; CP's K/V
    return Comm.apply(x, lambda t: all_gather(t, 1, group), lambda g: reduce_scatter(g, 1, group))


def scatter_seq(x, group):   # row output: sum, keep my slice
    return Comm.apply(x, lambda t: reduce_scatter(t, 1, group), lambda g: all_gather(g, 1, group))


def enter_sp(x, group):      # after the embedding
    return Comm.apply(x, lambda t: my_slice(t, group), lambda g: all_gather(g, 1, group))


def exit_sp(x, group):       # before the head
    return Comm.apply(x, lambda t: all_gather(t, 1, group), lambda g: my_slice(g, group))


class ColumnParallelLinear(nn.Linear):
    # my 1/tp of the OUTPUT features. Input: the same full x on every rank (TP), or
    # my slice of the sequence (SP), all-gathered here.
    def __init__(self, in_features: int, out_features: int, group=None, sequence_parallel: bool = False):
        self.group, self.tp, self.sp = group, ranks(group)[1] if group else 1, sequence_parallel
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
        self.group, self.tp, self.sp = group, ranks(group)[1] if group else 1, sequence_parallel
        super().__init__(in_features // self.tp, out_features)
        self.bias.sequence_parallel = sequence_parallel        # added in the SP region: see sync_sp_grads

    def forward(self, x):
        y = F.linear(x, self.weight)
        if self.tp > 1 and self.sp:
            y = scatter_seq(y, self.group)
        elif self.tp > 1:
            # forward: the sum of every rank's partial output. backward: identity (y + sum - y)
            y = all_reduce(y.detach(), self.group) + (y - y.detach())
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
