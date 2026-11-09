import json
import math
import os
import sys
import time
import urllib.request

import torch
from torch.nn import functional as F
import torch.distributed as dist

from collectives import all_reduce
from fsdp import fully_shard
from gpt import GPT, load_tp, mesh
from tp import sync_sp_grads

# ---- Muon: orthogonalized momentum for the hidden matrices (AdamW for everything else) ----
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    # approximately map G = U S V^T to U V^T, i.e. push every singular value to ~1
    a, b, c = 3.4445, -4.7750, 2.0315                  # quintic tuned to grow small singular values fast
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)                          # Frobenius >= spectral norm, so all sigma <= 1
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T                                         # iterate on the smaller Gram matrix
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X            # odd polynomial in X: same U, V; sigma -> p(sigma)
    if tall:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    # SGD-momentum whose update is orthogonalized. Only for 2D hidden weights: use AdamW for
    # embeddings, the LM head, norms and biases.
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])              # Nesterov
                update = newton_schulz(g)
                update *= max(1, p.shape[0] / p.shape[1]) ** 0.5          # similar RMS for tall matrices
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


# ---- data: character-level Tiny Shakespeare (1.1M chars, 65 symbols) ----
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(URL, "input.txt")
text = open("input.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
train_data, val_data = data[: int(0.9 * len(data))], data[int(0.9 * len(data)):]

# ---- settings, as key=value on the command line:
#   python train.py                                              one GPU, as before
#   python train.py parity=1 out=1gpu.json                       the parity run below, one process
#   torchrun --nproc-per-node 8 train.py parity=1 dp=2 cp=2 tp=2 ref=1gpu.json     FSDP x CP x TP/SP
args = dict(a.split("=") for a in sys.argv[1:])
arg = lambda k, default: type(default)(args.pop(k, default))
parity = arg("parity", 0)                         # 1: the small run below, repeated with and without parallelism
dp, cp, tp = arg("dp", 1), arg("cp", 1), arg("tp", 1)   # world = dp * cp * tp processes
out, ref = arg("out", ""), arg("ref", "")         # save this run's log / compare its losses with another log

# ---- config: Karpathy's shakespeare_char shape, ~10.7M params; dropout because 1M chars is little data ----
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
block_size, batch_size = 128, 32
max_steps, warmup = 2000, 100
cfg = dict(n_layer=6, feat_dim=384, num_head=6, dropout=0.2)
optim = "muon"                                                      # or "adamw"
if parity:   # ~0.5M params, AdamW, no dropout, on CPU (device=cuda for GPUs). Scale it up here:
    device, block_size, batch_size = "cpu", 64, 16                  # S % (tp * cp) == 0, B % dp == 0
    max_steps, warmup = 200, 20
    cfg = dict(n_layer=4, feat_dim=104, num_head=4, num_kv_head=2, dropout=0.0)   # tp | heads, tp | kv heads
    optim = "adamw"                                                 # bf16 mixed precision, as in step 25
device, max_steps = arg("device", device), arg("max_steps", max_steps)
assert not args, f"unknown settings: {args}"

# ---- parallel layout: torchrun starts world processes; mesh() (gpt.py) builds the TP, CP and FSDP groups
world = dp * cp * tp
rank = 0
if world > 1:
    if device == "cuda":
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")   # rank, world: from torchrun
    assert dist.get_world_size() == world, "launch dp * cp * tp processes"
    rank, me = dist.get_rank(), mesh(dp, cp, tp)
print0 = print if rank == 0 else lambda *a: None
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")         # needed by deterministic cuBLAS
torch.use_deterministic_algorithms(True, warn_only=True)            # a rerun gives the same bits
gen = torch.Generator().manual_seed(1337)                           # data only: model init can't shift it


def get_batch(split: str):
    # every rank draws the same global batch, then keeps its part: DP ranks different rows,
    # CP ranks different chunks of the sequence (contiguous, as gpt.py's positions assume), TP ranks the same
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([d[i: i + block_size] for i in ix])
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])     # targets = inputs shifted by 1
    if world > 1:
        x, y = (t.chunk(dp)[me["d"]].chunk(cp, dim=1)[me["c"]] for t in (x, y))   # same cut for targets
    return x.to(device), y.to(device)


torch.manual_seed(1337)                                             # the same full model on every rank
model = GPT(vocab_size=len(chars), block_size=block_size, **cfg)
n_params = sum(p.numel() for p in model.parameters())
if world > 1:     # my TP slice of that model, with CP in attention; FSDP-sharded below
    full, model = model, GPT(vocab_size=len(chars), block_size=block_size, **cfg,
                             tp_group=me["tp_group"], cp_group=me["cp_group"])
    load_tp(model, full, me["t"], tp)
model = model.to(device)
for p in model.parameters():
    p.data = p.data.bfloat16()                                      # buffers (RoPE freqs) stay fp32
if world > 1:
    model = fully_shard(model, me["fsdp_group"])                    # CP ranks hold the same weights too
print0(f"{n_params / 1e6:.2f}M params on {world} x {device}: FSDP {dp} x CP {cp} x TP/SP {tp}")

# ---- mixed precision by hand: the model holds bf16 weights and grads; the optimizer holds an fp32
# master copy plus fp32 Adam moments. Bytes per parameter: 2 + 2 + 4 + 4 + 4 = 16.
master = {p: p.detach().float().requires_grad_() for p in model.parameters()}
torch.manual_seed(1337 + rank)    # dropout: SP/CP/DP ranks hold different tokens, so different masks

# ---- optimizer: AdamW, weight decay only on matrices (not norms, biases) ----
assert optim == "adamw" or world == 1, "Muon needs whole matrices: gather the shards first"
is_hidden = lambda n, p: p.dim() == 2 and n.startswith("blocks.")   # Muon gets the hidden matrices
named = list(model.named_parameters())
rest = [p for n, p in named if not (optim == "muon" and is_hidden(n, p))]
decay = [master[p] for p in rest if p.dim() >= 2]                    # the optimizer sees fp32 masters
no_decay = [master[p] for p in rest if p.dim() < 2]
optimizers = [torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.95))]
if optim == "muon":
    optimizers.append(Muon([master[p] for n, p in named if is_hidden(n, p)], lr=0.02, momentum=0.95))
for opt in optimizers:
    for group in opt.param_groups:
        group["base_lr"] = group["lr"]


def lr_scale(step: int) -> float:
    # linear warmup, then cosine decay from 1.0 to 0.1
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / (max_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))


def clip_grad_norm(max_norm: float) -> torch.Tensor:
    # the norm of the FULL gradient. Ranks hold disjoint shards, except that every TP rank holds the
    # same copy of the embedding and final norm (the root unit) and of the SP-marked norms and row biases
    if world == 1:
        return torch.nn.utils.clip_grad_norm_(master.values(), max_norm)
    copies = {id(p) for p in model.shards} | {id(p) for p in model.parameters() if p.sequence_parallel}
    sq = sum((m.grad ** 2).sum() / (tp if id(p) in copies else 1) for p, m in master.items())
    norm = all_reduce(sq) ** 0.5                                    # count each copy once
    for m in master.values():
        m.grad.mul_((max_norm / (norm + 1e-6)).clamp(max=1.0))
    return norm


def lm_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # next-token cross-entropy in fp32, plus the MoE balance loss (0 for a dense model)
    logits = model(x)
    return F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) + 0.01 * model.aux_loss


@torch.no_grad()
def eval_loss(iters: int = 20) -> float:
    model.eval()
    loss = sum(lm_loss(*get_batch("val")) for _ in range(iters)) / iters
    model.train()
    return (all_reduce(loss) / world if world > 1 else loss).item()


t0, clock, losses = time.time(), 0.0, []
for step in range(max_steps):
    tic = time.time()
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * lr_scale(step)
    x, y = get_batch("train")
    loss = lm_loss(x, y)                                           # bf16 compute, fp32 loss: mean over MY tokens
    loss.backward()                                                 # FSDP averages grads over DP x CP
    if world > 1:
        sync_sp_grads(model, me["tp_group"])                        # SP-marked params saw 1/tp of the tokens
        loss = all_reduce(loss.detach()) / world                    # the global mean (TP ranks: equal copies)
    for p, m in master.items():                                     # bf16 grads -> fp32 master grads
        m.grad, p.grad = p.grad.float(), None
    clip_grad_norm(1.0)
    for opt in optimizers:
        opt.step()                                                  # update in fp32
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p, m in master.items():
            p.copy_(m)                                              # round the new weights back to bf16
    clock += (time.time() - tic) * (step > 0)                       # step 0 warms up
    losses.append(loss.item())
    if step % 200 == 0 or step == max_steps - 1:
        print0(f"step {step:4d} | train {loss.item():.3f} | val {eval_loss():.3f} | {time.time() - t0:.0f}s")

tok_s = batch_size * block_size * (max_steps - 1) / clock
print0(f"{tok_s:.0f} tokens/s over {world} process(es)")
if out:           # model states on this rank, counted: bf16 weights and grads, fp32 masters, Adam m and v
    nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts)
    mem = 2 * nbytes(model.parameters()) + nbytes(master.values())
    mem += nbytes(t for opt in optimizers for s in opt.state.values() for t in s.values())
    if rank == 0:
        json.dump(dict(params=n_params, tok_per_s=tok_s, mem=mem, loss=losses), open(out, "w"))
if ref:           # parity: the same losses as another layout's run, up to bf16 rounding
    other = json.load(open(ref))["loss"]
    gap = max(abs(a - b) for a, b in zip(losses, other))
    print0(f"max |loss - ref| over {len(other)} steps: {gap:.2e}")
if world == 1:
    prompt = torch.tensor([[stoi[c] for c in "ROMEO:\n"]], device=device)   # greedy: expect some loops
    print("".join(chars[i] for i in model.generate(prompt, 300)[0].tolist()))
else:
    dist.destroy_process_group()
