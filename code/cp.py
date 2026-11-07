import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from collectives import all_gather, all_reduce, all_to_all, ring_send_recv
from tp import gather_seq


# ---------------- Ulysses: swap sequence-sharding for head-sharding around attention
class SeqToHeads(torch.autograd.Function):
    # all-to-all: (B, S/P, H, hd) -> (B, S, H/P, hd). Send head-chunk j of my tokens to rank j;
    # receive every rank's tokens for my heads. Backward is the inverse all-to-all (collectives.py).
    @staticmethod
    def forward(ctx, x, scatter_dim, gather_dim):
        ctx.dims = scatter_dim, gather_dim
        P = dist.get_world_size()
        got = all_to_all(list(x.chunk(P, dim=scatter_dim)), [x.shape[0]] * P)   # dim 0 is B, never split
        return torch.cat(got, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad):
        scatter_dim, gather_dim = ctx.dims
        return SeqToHeads.apply(grad, gather_dim, scatter_dim), None, None


class UlyssesAttention(nn.Module):
    # each rank holds S/P tokens with ALL heads; attention runs on the FULL sequence for H/P heads.
    # Weights are replicated (grads all-reduced over the CP group, like data parallel).
    def __init__(self, feat_dim: int, num_head: int):
        super().__init__()
        self.num_head = num_head
        self.qkv = nn.Linear(feat_dim, 3 * feat_dim)
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # x: (B, S/P, F)
        assert self.num_head % dist.get_world_size() == 0       # the limit: P <= heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b s (h f) -> b s h f", h=self.num_head) for t in (q, k, v))
        q, k, v = (SeqToHeads.apply(t, 2, 1) for t in (q, k, v))   # split heads, gather sequence
        o = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)), is_causal=True)
        o = SeqToHeads.apply(o.transpose(1, 2), 1, 2)          # back: split sequence, gather heads
        return self.out(rearrange(o, "b s h f -> b s (h f)"))   # W_O mixes all heads of a token


# ---------------- all-gather CP: queries stay put, every rank gathers all keys and values
def all_gather_kv(k: torch.Tensor, v: torch.Tensor, group=None):
    # (B, S/P, ...) -> (B, S, ...). Backward is a reduce-scatter: dK, dV go home summed (tp.py)
    return gather_seq(k, group), gather_seq(v, group)


# ---------------- Ring: K/V blocks travel around the ring, merged with FA2's online softmax
def ring_pass(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    recv = torch.empty_like(t)
    ring_send_recv(t, recv)                                     # to rank+1, from rank-1
    return recv


@torch.no_grad()
def ring_attention(q, k, v, pos):
    # causal, forward only. q, k, v: (B, H, S/P, hd); pos: global positions of my tokens.
    # Q stays put; K/V (and their positions) move one rank right per step.
    k_pos, work = pos, 0
    out = torch.zeros_like(q)
    lse = torch.full(q.shape[:-1], float("-inf"))              # (B, H, S/P)
    for step in range(dist.get_world_size()):
        mask = k_pos[None, :] <= pos[:, None]                   # causal by position, not by rank
        if mask.any():                                          # skip blocks wholly in my future
            s = (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).masked_fill(~mask, float("-inf"))
            blk_lse = torch.logsumexp(s, dim=-1)
            blk_out = torch.softmax(s, dim=-1).nan_to_num(0.0) @ v   # rows with no visible key: 0
            new_lse = torch.logaddexp(lse, blk_lse)
            out = (lse - new_lse).exp()[..., None] * out + (blk_lse - new_lse).exp()[..., None] * blk_out
            lse = new_lse
            work += int(mask.sum())                             # score entries actually computed
        if step < dist.get_world_size() - 1:                    # real code: overlap with the compute
            k, v, k_pos = ring_pass(k), ring_pass(v), ring_pass(k_pos)
    return out, work


def layout(S: int, P: int, rank: int, zigzag: bool) -> torch.Tensor:
    # global positions this rank holds. Zigzag: cut S into 2P chunks, rank r takes r and 2P-1-r,
    # so every rank owns one early and one late chunk and the causal work evens out.
    if not zigzag:
        return torch.arange(S).chunk(P)[rank]
    c = torch.arange(S).chunk(2 * P)
    return torch.cat([c[rank], c[2 * P - 1 - rank]])


class CPAttention(nn.Module):
    # each rank holds S/P tokens (any layout) and ALL heads; weights replicated
    def __init__(self, feat_dim: int, num_head: int, ring: bool = False):
        super().__init__()
        self.num_head, self.ring = num_head, ring
        self.qkv = nn.Linear(feat_dim, 3 * feat_dim)
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):     # x: (B, S/P, F); pos: my positions
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if self.ring:                                           # never holds more than S/P keys
            q, k, v = (rearrange(t, "b s (h f) -> b h s f", h=self.num_head) for t in (q, k, v))
            o, self.work = ring_attention(q, k, v, pos)
            return self.out(rearrange(o, "b h s f -> b s (h f)"))
        k, v = all_gather_kv(k, v)                              # (B, S, F): every rank's keys/values
        q, k, v = (rearrange(t, "b s (h f) -> b h s f", h=self.num_head) for t in (q, k, v))
        mask = all_gather(pos)[None, :] <= pos[:, None]         # causal by position, any layout
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(rearrange(o, "b h s f -> b s (h f)"))


def _check(attn, full, x, ref, rank, world, **kw):
    # outputs and gradients of a CP attention vs single-GPU MHSA; my tokens are x[:, pos]
    pos = kw.pop("pos", torch.arange(x.shape[1]).chunk(world)[rank])
    attn.load_state_dict(full.state_dict())
    xs = x.detach()[:, pos].clone().requires_grad_()
    out = attn(xs, pos) if kw.get("with_pos") else attn(xs)
    (out ** 2).sum().backward()
    for p in attn.parameters():                                 # replicated weights: sum grads
        p.grad = all_reduce(p.grad)
    assert torch.allclose(out, ref[:, pos], atol=1e-5)
    assert torch.allclose(xs.grad, x.grad[:, pos], atol=1e-5)
    assert torch.allclose(attn.qkv.weight.grad, full.qkv.weight.grad, atol=1e-5)


def _test(rank: int, world: int):
    from gpt import MHSA
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29503", rank=rank, world_size=world)
    torch.manual_seed(0)
    full = MHSA(16, 4)
    x = torch.randn(2, 16, 16, requires_grad=True)
    ref = full(x, is_causal=True)
    (ref ** 2).sum().backward()
    _check(UlyssesAttention(16, 4), full, x, ref, rank, world)
    _check(CPAttention(16, 4), full, x, ref, rank, world, pos=layout(16, world, rank, True), with_pos=True)
    ring, work = CPAttention(16, 4, ring=True), {}
    ring.load_state_dict(full.state_dict())
    for zigzag in (False, True):                                # ring is forward-only here
        pos = layout(16, world, rank, zigzag)
        assert torch.allclose(ring(x.detach()[:, pos], pos), ref[:, pos], atol=1e-5)
        work[zigzag] = all_gather(torch.tensor([ring.work])).tolist()
    if rank == 0:
        print("Ulysses, all-gather CP and ring CP match single-GPU attention")
        print(f"ring work per rank (score entries): contiguous {work[False]}, zigzag {work[True]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(_test, args=(4,), nprocs=4)
