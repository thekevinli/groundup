import torch
import torch.distributed as dist
from torch.nn import functional as F

from collectives import all_gather, all_reduce, init, launch, reduce_scatter

# Tensor (and sequence) parallelism with hand-written backward, homework style: *_forward returns
# (out, cache), *_backward returns (dx, dw, db). No autograd, so every collective sits exactly where
# the chain rule needs it. Tokens are rows: x is (N, D). Weights are (in, out): y = x @ w + b.


def gelu_forward(x: torch.Tensor):
    t = torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3))       # tanh approximation (GPT-2)
    return 0.5 * x * (1 + t), (x, t)


def gelu_backward(dout: torch.Tensor, cache) -> torch.Tensor:
    x, t = cache
    du = (2 / torch.pi) ** 0.5 * (1 + 3 * 0.044715 * x ** 2)
    return dout * (0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du)


# ---------------- column-parallel: w (D, M/P), b (M/P,). I own M/P output features.
def column_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, group=None, sp: bool = False):
    if sp:
        x = all_gather(x, 0, group)                    # SP: my N/P tokens -> all N tokens
    return x @ w + b, (x, w)


def column_backward(dout: torch.Tensor, cache, group=None, sp: bool = False):
    x, w = cache
    dx = dout @ w.T                                    # partial: only my M/P hidden units
    dx = reduce_scatter(dx, 0, group) if sp else all_reduce(dx, group)   # sum over ranks (SP: keep my tokens)
    return dx, x.T @ dout, dout.sum(0)                 # dw, db: local, no communication


# ---------------- row-parallel: w (M/P, D), b (D,). Its input is the column output, already split.
def row_forward(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor, group=None, sp: bool = False):
    y = h @ w                                          # partial sum over my M/P inputs
    y = reduce_scatter(y, 0, group) if sp else all_reduce(y, group)      # sum (SP: keep my tokens)
    return y + b, (h, w)                               # bias once, after the sum


def row_backward(dout: torch.Tensor, cache, group=None, sp: bool = False):
    h, w = cache
    if sp:
        dout = all_gather(dout, 0, group)              # my h touched every token: need the full dout
    return dout @ w.T, h.T @ dout, dout.sum(0)


# ---------------- MLP: column -> GELU -> row
def mlp_forward(x: torch.Tensor, p: dict, group=None, sp: bool = False):
    h, c1 = column_forward(x, p["w1"], p["b1"], group, sp)
    a, cg = gelu_forward(h)
    y, c2 = row_forward(a, p["w2"], p["b2"], group, sp)
    return y, (c1, cg, c2)


def mlp_backward(dout: torch.Tensor, cache, group=None, sp: bool = False):
    c1, cg, c2 = cache
    da, dw2, db2 = row_backward(dout, c2, group, sp)
    dx, dw1, db1 = column_backward(gelu_backward(da, cg), c1, group, sp)
    return dx, {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}


def _test(rank: int, tp: int):
    init(rank, tp, 29503)
    torch.manual_seed(0)                               # the same full MLP, input and dout everywhere
    N, D, M = 8, 4, 16
    full = {"w1": torch.randn(D, M), "b1": torch.randn(M), "w2": torch.randn(M, D), "b2": torch.randn(D)}
    x, dout = torch.randn(N, D, requires_grad=True), torch.randn(N, D)
    leaves = {n: t.clone().requires_grad_() for n, t in full.items()}   # autograd reference
    y = F.gelu(x @ leaves["w1"] + leaves["b1"], approximate="tanh") @ leaves["w2"] + leaves["b2"]
    (y * dout).sum().backward()

    h = slice(rank * M // tp, (rank + 1) * M // tp)    # my hidden units
    t = slice(rank * N // tp, (rank + 1) * N // tp)    # my tokens (SP)
    mine = {"w1": full["w1"][:, h], "b1": full["b1"][h], "w2": full["w2"][h], "b2": full["b2"]}
    ref = {"w1": leaves["w1"].grad[:, h], "b1": leaves["b1"].grad[h], "w2": leaves["w2"].grad[h], "b2": leaves["b2"].grad}
    for sp in (False, True):
        xs, ds = (x.detach()[t], dout[t]) if sp else (x.detach(), dout)
        out, cache = mlp_forward(xs, mine, sp=sp)
        dx, grads = mlp_backward(ds, cache, sp=sp)
        assert torch.allclose(out, y.detach()[t] if sp else y.detach(), atol=1e-5)
        assert torch.allclose(dx, x.grad[t] if sp else x.grad, atol=1e-5)
        for n in grads:
            assert torch.allclose(grads[n], ref[n], atol=1e-4), n
        if rank == 0:
            print(("SP" if sp else "TP"), "MLP by hand matches autograd on the full MLP (output, dx, dw, db)")
    dist.destroy_process_group()


if __name__ == "__main__":
    launch(_test, 2)
