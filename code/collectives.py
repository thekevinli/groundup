import os
import sys

import torch
import torch.distributed as dist

CUDA = "device=cuda" in sys.argv          # torchrun --nproc-per-node P <file> device=cuda: NCCL, one GPU each


def ring_send_recv(send: torch.Tensor, recv: torch.Tensor, group=None) -> None:
    # send right, receive from the left. NCCL: use dist.batch_isend_irecv, or 2 ranks can deadlock
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    req = dist.isend(send.contiguous(), group=group, group_dst=(rank + 1) % world)   # returns at once
    dist.recv(recv, group=group, group_src=(rank - 1) % world)                       # waits for the data
    req.wait()


def reduce_scatter(x: torch.Tensor, dim: int = 0, group=None) -> torch.Tensor:
    # ring, P-1 steps: pass one partial sum right, add the one arriving from the left.
    # Afterwards rank r holds chunk r summed over all ranks.
    # in training: dist.reduce_scatter(out, list(x.chunk(P)), group=group)
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    # send/recv ship one flat block of memory; a chunk along dim>0 is a strided view. clone: we add into it
    chunks = [c.contiguous().clone() for c in x.chunk(world, dim=dim)]
    recv = torch.empty_like(chunks[0])
    for step in range(world - 1):
        # chunk index, not a rank; the -1 makes me end on chunk `rank` after P-1 steps
        send_i = (rank - step - 1) % world
        # my left neighbour's send_i: (rank - 1) - step - 1
        recv_i = (rank - step - 2) % world
        ring_send_recv(chunks[send_i], recv, group)
        chunks[recv_i] += recv
    return chunks[rank]


def all_gather(x: torch.Tensor, dim: int = 0, group=None) -> torch.Tensor:
    # ring, P-1 steps: forward the chunk that arrived last step. Everyone ends with every chunk.
    # in training: dist.all_gather(out_list, x, group=group)
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    # send/recv ship one flat block of memory, so a strided view (e.g. a transpose or slice) must be copied first
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
    # in training: dist.all_reduce(x, group=group)
    world = dist.get_world_size(group)
    # any shape: flatten to 1-D and zero-pad so it splits into P equal chunks
    pad = -x.numel() % world
    flat = torch.cat([x.flatten(), x.new_zeros(pad)])
    out = all_gather(reduce_scatter(flat, group=group), group=group)
    return out[: x.numel()].reshape(x.shape)                          # drop the padding, restore the shape


def all_to_all(chunks: list[torch.Tensor], recv_sizes: list[int], group=None) -> list[torch.Tensor]:
    # chunks[j] goes to rank j; returns the chunk every rank sent me. Round k: send to rank+k,
    # receive from rank-k. recv_sizes[j] = rows arriving from rank j (exchange counts first).
    # in training: dist.all_to_all(out_list, chunks, group=group)
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    out = [None] * world
    out[rank] = chunks[rank]
    for k in range(1, world):
        dst, src = (rank + k) % world, (rank - k) % world
        out[src] = chunks[dst].new_empty((recv_sizes[src], *chunks[dst].shape[1:]))
        req = dist.isend(chunks[dst].contiguous(), group=group, group_dst=dst)
        dist.recv(out[src], group=group, group_src=src)
        req.wait()
    return out


def straight_through(value: torch.Tensor, grad_to: torch.Tensor) -> torch.Tensor:
    # forward: value (e.g. our collective's result, outside autograd); backward: the gradient goes to grad_to as is
    return grad_to - grad_to.detach() + value


def init(rank: int, world: int, port: int) -> None:
    # the tests' process group: gloo over TCP between the CPU processes launch() spawned, or, with
    # device=cuda under torchrun, NCCL with one GPU per process (torchrun sets rank and world)
    if CUDA:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.set_default_device("cuda")                              # the test's tensors and modules
        dist.init_process_group("nccl")
        assert dist.get_world_size() == world, f"launch {world} processes"
    else:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)


def launch(test, world: int, *args) -> None:
    # python <file>: spawn world CPU processes. torchrun ... <file> device=cuda: they exist, run my rank
    if CUDA:
        test(int(os.environ["RANK"]), world, *args)
    else:
        torch.multiprocessing.spawn(test, args=(world, *args), nprocs=world)


# Adjoints: every collective is a linear map, so its backward is the transposed map.
#   all-gather  <->  reduce-scatter    (forward copies my chunk to everyone; backward sums them back)
#   all-reduce  <->  all-reduce        (sum then copy; transpose = sum then copy)
#   all-to-all  <->  all-to-all        (a permutation; transpose = the inverse permutation)


def _test(rank: int, world: int):
    init(rank, world, 29501)
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
    launch(_test, 4)
