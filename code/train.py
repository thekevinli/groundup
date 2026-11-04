import math
import os
import time
import urllib.request

import torch

from gpt import GPT

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

# ---- config: Karpathy's shakespeare_char shape, ~10.7M params; dropout because 1M chars is little data ----
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
block_size, batch_size = 128, 32
max_steps, warmup = 2000, 100
model = GPT(vocab_size=len(chars), block_size=block_size, n_layer=6, feat_dim=384, num_head=6,
            dropout=0.2).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {device}")

# ---- mixed precision by hand: the model holds bf16 weights and grads; the optimizer holds an fp32
# master copy plus fp32 Adam moments. Bytes per parameter: 2 + 2 + 4 + 4 + 4 = 16.
for p in model.parameters():
    p.data = p.data.bfloat16()                                      # buffers (RoPE freqs) stay fp32
master = {p: p.detach().float().requires_grad_() for p in model.parameters()}


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i: i + block_size] for i in ix])
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])     # targets = inputs shifted by 1
    return x.to(device), y.to(device)


# ---- optimizer: AdamW, weight decay only on matrices (not norms, biases) ----
optim = "muon"                                                      # or "adamw"
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


@torch.no_grad()
def eval_loss(iters: int = 20) -> float:
    model.eval()
    loss = sum(model(*get_batch("val"))[1].item() for _ in range(iters)) / iters
    model.train()
    return loss


t0 = time.time()
for step in range(max_steps):
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * lr_scale(step)
    x, y = get_batch("train")
    _, loss = model(x, y)                                           # bf16 compute, fp32 loss
    loss.backward()
    for p, m in master.items():                                     # bf16 grads -> fp32 master grads
        m.grad, p.grad = p.grad.float(), None
    torch.nn.utils.clip_grad_norm_(master.values(), 1.0)
    for opt in optimizers:
        opt.step()                                                  # update in fp32
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p, m in master.items():
            p.copy_(m)                                              # round the new weights back to bf16
    if step % 200 == 0 or step == max_steps - 1:
        print(f"step {step:4d} | train {loss.item():.3f} | val {eval_loss():.3f} | {time.time() - t0:.0f}s")

prompt = torch.tensor([[stoi[c] for c in "ROMEO:\n"]], device=device)   # greedy: expect some loops
print("".join(chars[i] for i in model.generate(prompt, 300)[0].tolist()))
