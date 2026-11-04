import torch
import torch.distributed as dist

from collectives import all_gather, all_to_all

D = 3072                                      # model width (Flux-sized)


def cost(l: int, d: int = D) -> float:
    # FLOPs of one transformer block on a length-l sequence: projections + MLP grow like l,
    # attention scores like l^2. Attention is half the work once l = 6d (18k tokens here).
    return 24 * l * d * d + 4 * l * l * d


def plan(lengths: list, bags: list, w=cost) -> list:
    """Greedy multi-knapsack: largest sequence first, into the bag that ends up least full.
    bags[b] = GPUs in bag b. A sequence in a g-GPU bag is split g ways (Ulysses inside the bag),
    so a bag's capacity is g times the per-GPU share of the total work."""
    c = [w(l) for l in lengths]
    cap = [g * sum(c) / sum(bags) for g in bags]
    load, where = [0.0] * len(bags), [0] * len(lengths)
    for i in sorted(range(len(c)), key=lambda i: -c[i]):
        fits = [b for b in range(len(bags)) if load[b] + c[i] <= cap[b]] or range(len(bags))
        b = min(fits, key=lambda b: (load[b] + c[i]) / cap[b])
        load[b] += c[i]
        where[i] = b
    return where


def gpu_load(lengths: list, where: list, bags: list) -> list:
    # true cost per GPU: each GPU of a g-GPU bag does 1/g of every sequence in that bag
    first = [sum(bags[:b]) for b in range(len(bags))]
    load = [0.0] * sum(bags)
    for l, b in zip(lengths, where):
        for k in range(bags[b]):
            load[first[b] + k] += cost(l) / bags[b]
    return load


def split(l: int, g: int) -> list:
    return [l // g + (k < l % g) for k in range(g)]          # g contiguous chunk sizes


# ---------------- routing: one all-to-all before the model, its inverse after
def route(seqs: list, bags: list, max_seqs: int = 64):
    """seqs: this rank's sequences, each (l, d). Returns the tokens this GPU must process and the
    bookkeeping route_back() needs. Every rank derives the same plan from the gathered lengths."""
    rank, world = dist.get_rank(), dist.get_world_size()
    mine = torch.full((max_seqs,), -1, dtype=torch.long)
    mine[: len(seqs)] = torch.tensor([len(s) for s in seqs])
    table = all_gather(mine).view(world, max_seqs)            # metadata only: a few ints per rank
    owner = [(r, i) for r in range(world) for i in range(max_seqs) if table[r, i] >= 0]
    lengths = [int(table[r, i]) for r, i in owner]
    where = plan(lengths, bags)                               # deterministic: same on every rank
    first = [sum(bags[:b]) for b in range(len(bags))]
    rows = [[0] * world for _ in range(world)]                # rows[src][dst]: tokens src -> dst
    pieces = []                                               # my sequences: [(dst, n), ...] each
    for (r, i), l, b in zip(owner, lengths, where):
        chunks = [(first[b] + k, n) for k, n in enumerate(split(l, bags[b]))]
        for dst, n in chunks:
            rows[r][dst] += n
        if r == rank:
            pieces.append(chunks)
    send = [[] for _ in range(world)]
    for s, chunks in zip(seqs, pieces):                       # chunk k of a sequence -> k-th GPU of its bag
        for (dst, _), part in zip(chunks, s.split([n for _, n in chunks])):
            send[dst].append(part)
    send = [torch.cat(p) if p else seqs[0][:0] for p in send]
    got = all_to_all(send, [rows[src][rank] for src in range(world)])
    return torch.cat(got), (rows, pieces)


def route_back(tokens: torch.Tensor, state) -> list:
    # the inverse permutation (all-to-all is its own adjoint): return every chunk to its owner
    rows, pieces = state
    rank, world = dist.get_rank(), dist.get_world_size()
    back = all_to_all(list(tokens.split([rows[src][rank] for src in range(world)])),
                      [rows[rank][dst] for dst in range(world)])
    ptr, out = [0] * world, []
    for chunks in pieces:                                     # same order they were sent in
        parts = []
        for dst, n in chunks:
            parts.append(back[dst][ptr[dst]: ptr[dst] + n])
            ptr[dst] += n
        out.append(torch.cat(parts))
    return out


# ---------------- tests
def _simulate(name: str, sizes: list, weights: list, per_gpu: int, P: int = 32, seed: int = 0):
    import random
    lengths = random.Random(seed).choices(sizes, weights=weights, k=per_gpu * P)   # as the loader delivers them
    single = [1] * P
    policies = {
        "same samples per GPU": (single, [i // per_gpu for i in range(len(lengths))]),
        "balance tokens": (single, plan(lengths, single, w=lambda l: l)),
        "balance cost": (single, plan(lengths, single)),
        "balance cost, 8-GPU bags": ([8] * (P // 8), plan(lengths, [8] * (P // 8))),
    }
    print(f"{name}: {P} GPUs, {len(lengths)} sequences, {sum(lengths):,} tokens")
    res = {}
    for policy, (bags, where) in policies.items():
        load = gpu_load(lengths, where, bags)
        assert abs(sum(load) - sum(cost(l) for l in lengths)) < 1e-6 * sum(load)   # work is conserved
        res[policy] = max(load) / (sum(load) / len(load))
        print(f"  {policy:26s} slowest GPU / mean = {res[policy]:5.2f}   (step efficiency {1 / res[policy]:4.0%})")
    return res


def _test_routing(rank: int, world: int):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29507", rank=rank, world_size=world)
    torch.manual_seed(rank)
    seqs = [torch.randn(int(torch.randint(4, 60, ())), 8) for _ in range(1 + 3 * rank)]   # uneven ranks
    tokens, state = route(seqs, bags=[1, 1, 2])
    back = route_back(tokens * 2, state)                     # stand-in for the model: any per-token map
    assert all(torch.equal(b, s * 2) for b, s in zip(back, seqs))
    # every rank calls the same collectives; printing is the only rank-0-only work
    before = all_gather(torch.tensor([sum(len(s) for s in seqs)]))
    after = all_gather(torch.tensor([len(tokens)]))
    if rank == 0:
        print(f"routing round trip ok; tokens per GPU {before.tolist()} -> {after.tolist()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    # 256px, 1k and 2k images (256, 4k, 16k tokens after an 8x VAE and 2x2 patches), 8 per GPU
    r = _simulate("mixed-resolution images", [256, 4096, 16384], [70, 22, 8], per_gpu=8)
    assert r["balance cost"] < r["balance tokens"] < r["same samples per GPU"]
    # add the odd 96k-token video, 4 samples per GPU: no single GPU can take its share
    r = _simulate("images + video", [256, 4096, 16384, 98304], [70, 20, 8, 2], per_gpu=4)
    assert r["balance cost, 8-GPU bags"] < 1.05 < r["balance cost"]     # one GPU can't hold the video's share
    torch.multiprocessing.spawn(_test_routing, args=(4,), nprocs=4)
