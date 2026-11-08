import torch
import torch.distributed as dist
from torch import nn

from collectives import all_reduce, all_to_all, init, launch
from gpt import MoE


class AllToAll(torch.autograd.Function):
    # rows x.split(send_sizes)[j] go to rank j. Backward sends the gradients back along the same
    # routes, i.e. another all-to-all with send and receive sizes swapped (its own adjoint)
    @staticmethod
    def forward(ctx, x, send_sizes, recv_sizes):
        ctx.sizes = send_sizes, recv_sizes
        return torch.cat(all_to_all(list(x.split(send_sizes)), recv_sizes))

    @staticmethod
    def backward(ctx, grad):
        send_sizes, recv_sizes = ctx.sizes
        return torch.cat(all_to_all(list(grad.split(recv_sizes)), send_sizes)), None, None


class EPMoE(nn.Module):
    # expert parallel: rank r owns experts [r*E/P, (r+1)*E/P); the router is replicated.
    # dispatch all-to-all -> my experts -> combine all-to-all, around the same math as MoE
    def __init__(self, feat_dim: int, num_expert: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.local = num_expert // dist.get_world_size()                  # experts per rank
        self.router = nn.Linear(feat_dim, num_expert, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, 4 * feat_dim), nn.GELU(), nn.Linear(4 * feat_dim, feat_dim))
            for _ in range(self.local)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank, world = dist.get_rank(), dist.get_world_size()
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        probs = torch.softmax(self.router(x), dim=-1)
        gate, idx = probs.topk(self.top_k, dim=-1)
        gate = gate / gate.sum(dim=-1, keepdim=True)
        # 1. one row per (token, choice), sorted by expert id = grouped by the rank that owns it
        tok = torch.arange(len(x)).repeat_interleave(self.top_k)
        exp = idx.flatten()
        order = torch.argsort(exp, stable=True)
        tok, exp = tok[order], exp[order]
        send = torch.bincount(exp // self.local, minlength=world).tolist()
        recv = [int(c) for c in all_to_all(list(torch.tensor(send).split(1)), [1] * world)]  # counts first
        x_in = AllToAll.apply(x[tok], send, recv)                         # tokens for my experts
        e_in = torch.cat(all_to_all(list(exp.split(send)), recv)) - rank * self.local
        # 2. run my experts on whatever arrived
        y = torch.zeros_like(x_in)
        for e, expert in enumerate(self.experts):
            sel = (e_in == e).nonzero(as_tuple=True)[0]
            if len(sel):
                y = y.index_add(0, sel, expert(x_in[sel]))
        # 3. send results home, weight by the gate, sum each token's k choices
        y = AllToAll.apply(y, recv, send)
        w = gate.flatten()[order]
        return torch.zeros_like(x).index_add(0, tok, w[:, None] * y).reshape(shape)


def _test(rank: int, world: int):
    init(rank, world, 29505)
    torch.manual_seed(0)
    full = MoE(16, num_expert=8, top_k=2)                                # same weights everywhere
    x = torch.randn(world, 6, 16)                                        # 6 tokens per rank
    ref = full(x)
    (ref ** 2).sum().backward()

    ep = EPMoE(16, num_expert=8, top_k=2)
    lo = rank * ep.local
    with torch.no_grad():
        ep.router.weight.copy_(full.router.weight)
        for e in range(ep.local):
            ep.experts[e].load_state_dict(full.experts[lo + e].state_dict())
    out = ep(x[rank:rank + 1])
    (out ** 2).sum().backward()
    ep.router.weight.grad = all_reduce(ep.router.weight.grad)            # replicated: sum over ranks
    assert torch.allclose(out, ref[rank:rank + 1], atol=1e-5)
    assert torch.allclose(ep.router.weight.grad, full.router.weight.grad, atol=1e-5)
    for e in range(ep.local):                                            # experts saw every rank's tokens
        assert torch.allclose(ep.experts[e][0].weight.grad, full.experts[lo + e][0].weight.grad, atol=1e-5)
    if rank == 0:
        print(f"EP MoE matches single-device MoE; each rank holds {ep.local} of 8 experts")
    dist.destroy_process_group()


if __name__ == "__main__":
    launch(_test, 4)
