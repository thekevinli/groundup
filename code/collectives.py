import torch
import torch.distributed as dist


def ranks(group=None):
    # a group is a subset of ranks (one TP group, one DP group, ...); None = every rank.
    # Returns my rank in the group, its size, and group rank -> global rank for send/recv.
    if group is None:
        return dist.get_rank(), dist.get_world_size(), lambda r: r
    return dist.get_rank(group), dist.get_world_size(group), lambda r: dist.get_global_rank(group, r)


def ring_send_recv(send: torch.Tensor, recv: torch.Tensor, group=None) -> None:
    # send to the right neighbour while receiving from the left one
    rank, world, glob = ranks(group)
    send = send.contiguous()
    req = dist.isend(send, dst=glob((rank + 1) % world), group=group)
    dist.recv(recv, src=glob((rank - 1) % world), group=group)
    req.wait()


def reduce_scatter(x: torch.Tensor, dim: int = 0, group=None) -> torch.Tensor:
    # ring, P-1 steps: pass one partial sum right, add the one arriving from the left.
    # Afterwards rank r holds chunk r summed over all ranks.
    rank, world, _ = ranks(group)
    chunks = [c.contiguous().clone() for c in x.chunk(world, dim=dim)]
    recv = torch.empty_like(chunks[0])
    for step in range(world - 1):
        send_i = (rank - step - 1) % world
        recv_i = (rank - step - 2) % world
        ring_send_recv(chunks[send_i], recv, group)
        chunks[recv_i] += recv
    return chunks[rank]


def all_gather(x: torch.Tensor, dim: int = 0, group=None) -> torch.Tensor:
    # ring, P-1 steps: forward the chunk that arrived last step. Everyone ends with every chunk.
    rank, world, _ = ranks(group)
    x = x.contiguous()
    chunks = [torch.empty_like(x) for _ in range(world)]
    chunks[rank] = x
    for step in range(world - 1):
        send_i = (rank - step) % world
        recv_i = (rank - step - 1) % world
        ring_send_recv(chunks[send_i], chunks[recv_i], group)
    return torch.cat(chunks, dim=dim)


def all_reduce(x: torch.Tensor, group=None) -> torch.Tensor:
    # reduce-scatter then all-gather: each rank sends 2(P-1)/P * |x|, almost independent of P
    world = ranks(group)[1]
    flat = x.flatten()
    flat = torch.cat([flat, flat.new_zeros((-flat.numel()) % world)])  # pad to a multiple of P
    return all_gather(reduce_scatter(flat, group=group), group=group)[: x.numel()].view_as(x)


def all_to_all(chunks: list[torch.Tensor], recv_sizes: list[int], group=None) -> list[torch.Tensor]:
    # chunks[j] goes to rank j; returns the chunk every rank sent me. Round k: send to rank+k,
    # receive from rank-k. recv_sizes[j] = rows arriving from rank j (exchange counts first).
    rank, world, glob = ranks(group)
    out = [None] * world
    out[rank] = chunks[rank]
    for k in range(1, world):
        dst, src = (rank + k) % world, (rank - k) % world
        send = chunks[dst].contiguous()
        recv = send.new_empty((recv_sizes[src], *send.shape[1:]))
        req = dist.isend(send, dst=glob(dst), group=group) if send.numel() else None   # skip empty messages
        if recv.numel():
            dist.recv(recv, src=glob(src), group=group)
        if req:
            req.wait()
        out[src] = recv
    return out


# Adjoints: every collective is a linear map, so its backward is the transposed map.
#   all-gather  <->  reduce-scatter    (forward copies my chunk to everyone; backward sums them back)
#   all-reduce  <->  all-reduce        (sum then copy; transpose = sum then copy)
#   all-to-all  <->  all-to-all        (a permutation; transpose = the inverse permutation)


def _test(rank: int, world: int):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29501", rank=rank, world_size=world)
    torch.manual_seed(rank)
    x = torch.randn(world * 2, 3)
    everyone = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(everyone, x)                       # built-in, only to check ours
    total = sum(everyone)
    assert torch.allclose(reduce_scatter(x), total.chunk(world)[rank], atol=1e-6)
    assert torch.allclose(all_gather(x), torch.cat(everyone), atol=1e-6)
    assert torch.allclose(all_reduce(x), total, atol=1e-6)
    sizes = [(rank + j) % 3 for j in range(world)]               # uneven, sometimes empty
    chunks = [torch.full((n, 2), 10.0 * rank + j) for j, n in enumerate(sizes)]
    got = all_to_all(chunks, recv_sizes=[(j + rank) % 3 for j in range(world)])
    assert all(torch.equal(got[j], torch.full(((j + rank) % 3, 2), 10.0 * j + rank)) for j in range(world))
    pairs = [dist.new_group([0, 1]), dist.new_group([2, 3])]     # every rank creates every group
    mine = pairs[rank // 2]
    assert torch.allclose(all_reduce(x, group=mine), everyone[rank // 2 * 2] + everyone[rank // 2 * 2 + 1], atol=1e-6)
    if rank == 0:
        print("collectives OK, on the whole world and on sub-groups")
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(_test, args=(4,), nprocs=4)
