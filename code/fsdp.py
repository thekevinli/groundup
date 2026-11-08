import torch
import torch.distributed as dist
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from collectives import all_gather, init, launch, ranks, reduce_scatter


class GatherParam(torch.autograd.Function):
    # forward: all-gather a parameter's dim-0 shards. backward: reduce-scatter its full gradient,
    # so each rank keeps only its shard's gradient, averaged over ranks (each saw different data)
    @staticmethod
    def forward(ctx, shard, rows, group):
        ctx.group, ctx.world = group, ranks(group)[1]
        ctx.pad = shard.shape[0] * ctx.world - rows
        return all_gather(shard, group=group)[:rows]

    @staticmethod
    def backward(ctx, grad):
        grad = torch.cat([grad, grad.new_zeros(ctx.pad, *grad.shape[1:])])
        return reduce_scatter(grad, group=ctx.group) / ctx.world, None, None


class FSDP(nn.Module):
    # FSDP2-style ZeRO-3 for one unit (here: one TransformerBlock). Every parameter is sharded on
    # dim 0 -- rank r keeps the r-th contiguous chunk of rows of each weight -- so each shard is still
    # "that weight": same name, dtype and requires_grad. Full weights exist only while the unit runs.
    def __init__(self, module: nn.Module, group=None, reshard: bool = True):
        super().__init__()
        self.group, self.reshard = group, reshard                           # the data-parallel group
        rank, world, _ = ranks(group)
        self.names, self.rows = [], []
        self.shards = nn.ParameterList()
        self.inner = nn.ModuleList(m for m in module.modules() if isinstance(m, FSDP))  # wrapped already
        theirs = {id(p) for u in self.inner for p in u.parameters()}
        for name, p in module.named_parameters():
            if id(p) in theirs:                                             # an inner unit's shard
                continue
            pad = (-p.shape[0]) % world                                     # pad dim 0 to a multiple of P
            full = torch.cat([p.detach(), p.new_zeros(pad, *p.shape[1:])])
            shard = nn.Parameter(full.chunk(world)[rank].clone())             # the only stored copy
            shard.sequence_parallel = getattr(p, "sequence_parallel", False)   # keep tp.py's marker
            self.shards.append(shard)
            self.names.append(name)
            self.rows.append(p.shape[0])
            p.data = torch.empty(0, device=p.device)                        # free the full copy
        self.unit = [module]                       # a list, so its params aren't registered here

    def __getattr__(self, name):
        # anything else (block.mlp, model.blocks, ...) comes from the wrapped module, as in FSDP2
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "unit":
                raise
            return getattr(self.unit[0], name)

    def _run(self, args, kwargs, *shards):
        full = {n: GatherParam.apply(s, r, self.group) for n, s, r in zip(self.names, shards, self.rows)}
        return functional_call(self.unit[0], full, args, kwargs)

    def forward(self, *args, **kwargs):
        if not self.reshard:          # the root unit: keep its weights gathered until backward
            return self._run(args, kwargs, *self.shards)
        # checkpoint = "reshard after forward": the gathered weights aren't kept for backward; the
        # recompute gathers them again (it also recomputes activations, which real FSDP doesn't)
        return checkpoint(self._run, args, kwargs, *self.shards, use_reentrant=False)


def fully_shard(model: nn.Module, group=None) -> FSDP:
    # one unit per block (only one block's full weights exist at a time), then the root unit for
    # whatever is left (embedding = tied head, final norm), kept gathered until backward
    model.blocks = nn.ModuleList(FSDP(b, group) for b in model.blocks)
    return FSDP(model, group, reshard=False)


def _test(rank: int, world: int):
    from gpt import TransformerBlock
    init(rank, world, 29504)
    torch.manual_seed(0)
    blocks = [TransformerBlock(18, 3) for _ in range(2)]                # 18 rows: not divisible by 4
    x = torch.randn(world * 2, 8, 18)                                   # global batch
    ref = x
    for b in blocks:
        ref = b(ref, is_causal=True)
    (ref ** 2).mean().backward()                                        # single-device reference
    ref_grads = [[p.grad.clone() for p in b.parameters()] for b in blocks]

    units = [FSDP(b) for b in blocks]
    h = x.chunk(world)[rank]                                            # my slice of the batch
    for u in units:
        h = u(h, is_causal=True)
    (h ** 2).mean().backward()
    assert torch.allclose(h, ref.chunk(world)[rank], atol=1e-5)
    for u, gs in zip(units, ref_grads):
        for s, g in zip(u.shards, gs):
            g = torch.cat([g, g.new_zeros((-g.shape[0]) % world, *g.shape[1:])]).chunk(world)[rank]
            assert s.grad.shape == s.shape and torch.allclose(s.grad, g, atol=1e-6)   # my rows of the mean grad
    if rank == 0:
        full = sum(g.numel() for gs in ref_grads for g in gs)
        mine = sum(s.numel() for u in units for s in u.shards)
        print(f"FSDP matches single-device; rank 0 stores {mine} of {full} parameters "
              f"(1/{world} plus dim-0 padding), each still a named tensor of its own")
    dist.destroy_process_group()


if __name__ == "__main__":
    launch(_test, 4)
