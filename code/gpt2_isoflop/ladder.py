"""Predict GPT-2's loss from small models: an IsoFLOP ladder in GPT-2's own setup.

Same tokenizer (GPT-2 BPE), data (FineWeb sample-10BT, <|endoftext|> before each document, like llm.c),
context (1024) and architecture as llm.c's GPT-2 124M run, so the loss we predict is the loss it reports.

  1. calibrate the lr rule (two lr sweeps, one width, two budgets)
  2. 6 rungs 2x apart: walk each from the left, stop 2 sizes past the minimum
  3. fit N_opt ~ C^a (Approach 2) and L(N, D) = E + A/N^alpha + B/D^beta (Approach 3)
  4. pre-register GPT-2 124M/355M/774M/1.5B at 9.89B tokens, then train a held-out 124M-shape check run
  5. leftover time: a second seed at each rung's minimum, then 0.5x / 2x lr probes there

Parameters and FLOPs are TOTAL counts, embedding included (Pearce & Song 2024).
"""
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK = 1024                                  # GPT-2's context
VOCAB = 50304                                 # GPT-2's 50257 padded to a multiple of 64 (nanoGPT, llm.c)
WIDTHS = [24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 576, 640, 768]
RUNGS = [8e14, 1.6e15, 3.2e15, 6.4e15, 1.28e16, 2.56e16]   # 2x apart, ~1 H100-hour in total
CALIB = (64, 8e14, 3.2e15)                    # lr rule: tune lr at this width at these two budgets
CHECK = (768, 4e16)                           # held-out check: the 124M shape at 4e16 FLOPs (~47M tokens)
UPPER_FROM = 6.4e15                           # second fit on these rungs only (less embedding-dominated)
MICRO = 16                                    # sequences per forward pass; gradient accumulation above
EVAL_TOKENS = 2 ** 21                         # a 2M-token prefix of llm.c's val shard
GPT2 = {"124M": (768, 12), "355M": (1024, 24), "774M": (1280, 36), "1558M": (1600, 48)}
LLMC_TOKENS = 18865 * 524288                  # llm.c's 124M run: 18,865 steps of 0.5M tokens = 9.89B


def depth(width: int) -> int:
    return max(2, int(width / 64 + 0.5))      # d/L = 64, GPT-2 124M's aspect ratio


# ---------------- model: GPT-2 (learned positions, pre-norm, GELU-tanh MLP, biases, tied head)
class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.h = max(1, d // 64)                                   # head_dim 64 (24-48 for the tiny widths)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # flash on GPU
        x = x + self.proj(y.transpose(1, 2).reshape(B, S, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, width: int, n_layer: int = None):
        super().__init__()
        L = n_layer or depth(width)
        self.wte, self.wpe = nn.Embedding(VOCAB, width), nn.Embedding(BLOCK, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(L)])
        self.ln_f = nn.LayerNorm(width)
        for name, p in self.named_parameters():                    # GPT-2 init; residual outputs / sqrt(2L)
            if name.endswith("bias"):
                nn.init.zeros_(p)
            elif p.dim() == 2:
                nn.init.normal_(p, std=0.02 / (2 * L) ** 0.5 if name.endswith(("proj.weight", "fc2.weight")) else 0.02)

    def forward(self, idx, targets):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.ln_f(x) @ self.wte.weight.T                  # tied head
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


def size(width: int, n_layer: int = None):
    """Total parameters N, and training FLOPs per token: 6 per matmul weight (the position table is a
    lookup, not a matmul), plus 12 * L * T * d for the attention scores."""
    L, d = n_layer or depth(width), width
    N = VOCAB * d + BLOCK * d + L * (12 * d * d + 13 * d) + 2 * d
    return N, 6 * (N - BLOCK * d) + 12 * L * BLOCK * d


# ---------------- hyperparameters as functions of the run's compute (DeepSeek LLM 2024, Sec. 3.1)
def batch_seqs(C: float) -> int:
    tokens = 0.292 * C ** 0.3271                                   # 22k tokens at 8e14 ... 0.43M at GPT-2 124M
    return max(MICRO, round(tokens / BLOCK / MICRO) * MICRO)       # whole micro-batches: one compiled shape


def lr_prior(C: float) -> float:
    return 0.3118 * C ** -0.125                                    # only the starting point of the lr sweep


def seeds(C: float) -> int:
    return 2 if C <= 1.6e15 else 1                                # the two cheapest rungs are the noisiest


# ---------------- data: uint16 token files written by write_tokens()
class Data:
    def __init__(self, data_dir: str):
        tr = np.memmap(f"{data_dir}/train.bin", dtype=np.uint16, mode="r")
        va = np.fromfile(f"{data_dir}/val.bin", dtype=np.uint16)[:EVAL_TOKENS]
        self.n_seq = len(tr) // (BLOCK + 1)
        self.train = tr[: self.n_seq * (BLOCK + 1)].reshape(self.n_seq, BLOCK + 1)
        self.val = torch.from_numpy(va[: len(va) // (BLOCK + 1) * (BLOCK + 1)].astype(np.int64)).view(-1, BLOCK + 1)

    def order(self, seed: int):
        # one permutation per seed: every size on a rung sees the same sequences in the same order
        return np.random.default_rng(seed).permutation(self.n_seq)


def write_tokens(docs, out_dir: str, n_val: int, n_train: int, batch: int = 1024, threads: int = 8):
    """GPT-2 ids, <|endoftext|> BEFORE each document (llm.c's convention). The first n_val tokens are
    val.bin, the next n_train are train.bin: disjoint documents from the same distribution."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    out, n = np.empty(n_val + n_train, dtype=np.uint16), 0
    docs = iter(docs)
    while n < len(out):
        texts = list(itertools.islice(docs, batch))
        assert texts, "ran out of documents"
        for ids in enc.encode_ordinary_batch(texts, num_threads=threads):
            ids = ([enc.eot_token] + ids)[: len(out) - n]          # the last document may be cut
            out[n: n + len(ids)] = ids
            n += len(ids)
    os.makedirs(out_dir, exist_ok=True)
    out[:n_val].tofile(f"{out_dir}/val.bin")
    out[n_val:].tofile(f"{out_dir}/train.bin")
    json.dump(dict(tokenizer="gpt2", val_tokens=n_val, train_tokens=n_train), open(f"{out_dir}/meta.json", "w"))


# ---------------- one training run
def evaluate(model, data, device):
    model.eval()
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        losses = [model(v[:, :-1].to(device), v[:, 1:].to(device)).item() * len(v) for v in data.val.split(8)]
    model.train()
    return sum(losses) / len(data.val)


def train(data: Data, width: int, C: float, lr: float, seed: int, device: str) -> dict:
    N, fpt = size(width)
    B = batch_seqs(C)
    steps = int(C / (fpt * B * BLOCK))
    assert steps * B <= data.n_seq, f"width {width} at C={C:.1e} would repeat data"
    torch.manual_seed(seed)
    model = GPT(width).to(device)
    fwd = torch.compile(model) if device == "cuda" else model
    decay = [p for p in model.parameters() if p.dim() == 2]       # matrices and embeddings, not norms/biases
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=device == "cuda")
    order = data.order(seed)
    t0 = time.time()
    for step in range(steps):
        # warmup 10%, cosine to 10%, ending exactly at this run's budget
        for g in opt.param_groups:
            g["lr"] = lr * min(1, (step + 1) / max(1, steps // 10)) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        rows = np.sort(order[step * B: (step + 1) * B])            # sorted: faster reads from the memmap
        for mb in torch.from_numpy(data.train[rows].astype(np.int64)).split(MICRO):
            mb = mb.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = fwd(mb[:, :-1], mb[:, 1:]) * len(mb) / B    # mean over the whole batch
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    val = evaluate(model, data, device)
    sec = time.time() - t0
    return dict(width=width, N=N, C=C, D=steps * B * BLOCK, steps=steps, B=B, lr=lr, seed=seed, val=val,
                sec=sec, flops_per_sec=C / sec)


# ---------------- planning: measured throughput -> hours
def bench(device: str, widths=WIDTHS, steps: int = 6) -> dict:
    """Training FLOP/s per width on random tokens; the first 2 steps are compile and warmup."""
    out = {}
    for w in widths:
        model = GPT(w).to(device)
        fwd = torch.compile(model) if device == "cuda" else model
        opt = torch.optim.AdamW(model.parameters(), fused=device == "cuda")
        x = torch.randint(0, 50257, (MICRO, BLOCK + 1), device=device)
        for i in range(steps):
            if i == 2:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast(device, dtype=torch.bfloat16):
                fwd(x[:, :-1], x[:, 1:]).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        out[w] = (steps - 2) * MICRO * BLOCK * size(w)[1] / (time.time() - t0)
        print(f"  width {w:4d}: {out[w] / 1e12:.1f} TFLOP/s", flush=True)
    return out


def estimate_hours(thr: dict, rungs=RUNGS, calib=CALIB, check=CHECK) -> dict:
    """~6 sizes per rung (x seeds), ~5 lrs at each calibration budget, the check run, then the extras.
    +8% for evals and ~15 s per run to compile; throughput from the benchmarked width nearest the optimum."""
    def h(C, n):
        w = min(thr, key=lambda w: abs(math.log(size(w)[0] / (C / 120) ** 0.5)))
        return n * (1.08 * C / thr[w] + 15) / 3600
    out = {f"rung {C:.1e}": h(C, 6 * seeds(C)) for C in rungs}
    out["calibration"] = h(calib[1], 5 * seeds(calib[1])) + h(calib[2], 5 * seeds(calib[2]))
    out["check run"] = 1.08 * check[1] / thr[max(thr)] / 3600
    out["seed repeats"] = sum(h(C, 1) for C in rungs if seeds(C) == 1)
    out["lr probes"] = sum(h(C, 2) for C in rungs)
    out["total"] = sum(out.values())
    return out


# ---------------- the ladder
class Ladder:
    def __init__(self, data, device, max_hours, widths=WIDTHS):
        self.data, self.device, self.widths = data, device, widths
        self.deadline = time.time() + max_hours * 3600
        self.cache, self.thr = {}, 2e13                            # FLOP/s guess, replaced by measurements

    def fits(self, C, n_runs):                                     # will n_runs more runs at C finish in time?
        return time.time() + n_runs * C / self.thr < self.deadline

    def run(self, width, C, lr, seed=0):
        key = (width, C, round(lr, 10), seed)
        if key not in self.cache:
            r = self.cache[key] = train(self.data, width, C, lr, seed, self.device)
            self.thr = r["flops_per_sec"] if len(self.cache) == 1 else 0.7 * self.thr + 0.3 * r["flops_per_sec"]
            print(f"  w={width:4d} N={r['N']:>11,} C={C:.1e} lr={lr:.3g} s{seed}: {r['val']:.4f} "
                  f"({r['steps']} steps, {r['sec']:.0f}s)", flush=True)
        return self.cache[key]

    def tokens(self, width, C):
        B = batch_seqs(C)
        return int(C / (size(width)[1] * B * BLOCK)) * B * BLOCK

    def lr(self, width, C):
        return self.rule(width, batch_seqs(C), self.tokens(width, C))

    def point(self, width, C, lr=None):
        rs = [self.run(width, C, lr or self.lr(width, C), s) for s in range(seeds(C))]
        v = [r["val"] for r in rs]
        return dict(rs[0], val=float(np.mean(v)), sem=float(np.std(v, ddof=1) / len(v) ** 0.5) if len(v) > 1 else None)

    def best_lr(self, width, C, lr):
        # sqrt(2) grid: step outward until the best lr is interior
        r2 = 2 ** 0.5
        res = {l: self.point(width, C, l)["val"] for l in (lr / r2, lr, lr * r2)}
        while (b := min(res, key=res.get)) in (min(res), max(res)):
            nl = b * r2 if b == max(res) else b / r2
            res[nl] = self.point(width, C, nl)["val"]
        return b

    def calibrate(self, w, C1, C2):
        # lr = lr0 * (w0/w) * sqrt(B/B0) * (T0/T)^t: muP width term, Adam's sqrt(B), fitted token exponent t
        lr1 = self.best_lr(w, C1, lr_prior(C1))
        lr2 = self.best_lr(w, C2, lr1)
        B1, B2, T1, T2 = batch_seqs(C1), batch_seqs(C2), self.tokens(w, C1), self.tokens(w, C2)
        t = float(np.clip(math.log((B2 / B1) ** 0.5 * lr1 / lr2) / math.log(T2 / T1), 0.0, 0.6))
        self.rule = lambda width, B, T: lr1 * (w / width) * (B / B1) ** 0.5 * (T1 / T) ** t
        print(f"lr rule: lr0={lr1:.3g} at width {w}, C={C1:.0e}; token exponent t={t:.2f}", flush=True)
        return dict(lr0=lr1, lr_c2=lr2, t=t, w0=w, B0=B1, T0=T1)

    def curve(self, C, start):
        W, pts = self.widths, {}
        best = lambda: min(pts, key=lambda j: pts[j]["val"])
        i = start
        while i < len(W) and (not pts or max(pts) - best() < 2):   # walk right: stop 2 sizes past the min
            pts[i] = self.point(W[i], C)
            i += 1
        while min(pts) > 0 and best() - min(pts) < 2:               # and keep 2 sizes left of it
            pts[min(pts) - 1] = self.point(W[min(pts) - 1], C)
        k = best()
        near = [pts[j] for j in sorted(pts) if abs(j - k) <= 2]     # parabola in log N through the bottom
        a, b, c = np.polyfit(np.log10([r["N"] for r in near]), [r["val"] for r in near], 2)
        n_opt = 10 ** (-b / (2 * a)) if a > 0 else pts[k]["N"]
        return dict(C=C, points=[pts[j] for j in sorted(pts)], fit=[a, b, c], k=k, n_opt=float(n_opt),
                    l_opt=float(np.polyval([a, b, c], math.log10(n_opt))),
                    bracketed=bool(k - min(pts) >= 2 and max(pts) - k >= 2 and a > 0))

    def repeat(self, cv):
        # one-seed rungs: rerun the minimum with seed 1, so every rung has a noise estimate
        w, C = self.widths[cv["k"]], cv["C"]
        r = self.run(w, C, self.lr(w, C), seed=1)
        cv["repeat"] = dict(r, diff=r["val"] - next(p["val"] for p in cv["points"] if p["width"] == w))

    def probes(self, cv):
        w, C = self.widths[cv["k"]], cv["C"]
        cv["probes"] = [self.point(w, C, m * self.lr(w, C)) for m in (0.5, 2.0)]


# ---------------- fits
def fit(curves):
    """Approach 2: each rung's vertex, then N_opt ~ C^a and L*(C) = E + A C^-alpha (floor by grid search)."""
    use = [c for c in curves if c["bracketed"]] or curves
    C, N, L = (np.array([c[k] for c in use]) for k in ("C", "n_opt", "l_opt"))
    a, b0 = np.polyfit(np.log10(C), np.log10(N), 1)
    floors = [(E, *np.polyfit(np.log(C), np.log(L - E), 1)) for E in np.linspace(0, L.min() - 1e-3, 400)]
    E, s, i = min(floors, key=lambda f: np.sum((f[0] + np.exp(f[2]) * C ** f[1] - L) ** 2))
    return dict(rungs=C.tolist(), a=float(a), b0=float(b0), floor=[float(E), float(np.exp(i)), float(-s)])


def fit_lnd(points, delta=1e-3):
    """Approach 3 (Hoffmann et al. 2022): L(N, D) = E + A/N^alpha + B/D^beta on every ladder point,
    Huber loss on log L, L-BFGS from a grid of starts (the surface has many local minima)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    lN, lD, y = (np.log([p[k] for p in points]) for k in ("N", "D", "val"))

    def obj(x):
        a, b, e, al, be = x
        r = logsumexp([a - al * lN, b - be * lD, np.full_like(lN, e)], axis=0) - y
        return np.sum(np.where(np.abs(r) < delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta)))

    grid = itertools.product([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], [0.0, 0.5, 1.0], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8])
    a, b, e, al, be = min((minimize(obj, x0, method="L-BFGS-B") for x0 in grid), key=lambda r: r.fun).x
    return dict(E=float(np.exp(e)), A=float(np.exp(a)), B=float(np.exp(b)), alpha=float(al), beta=float(be),
                n_points=len(points))


def L_of(p, N, D):
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D ** p["beta"]


def fit_points(curves):
    return [pt for c in curves for pt in c["points"]]        # rule-lr points only (no probes, no lr sweeps)


def predictions(curves, f2, check=None):
    """The pre-registered numbers: GPT-2 sizes at llm.c's 9.89B tokens (and the check run) from Approach 3,
    with a leave-one-rung-out range and an upper-rungs-only fit."""
    p = fit_lnd(fit_points(curves))
    loro = [fit_lnd([q for q in fit_points(curves) if q["C"] != c["C"]]) for c in curves] if len(curves) > 2 else []
    upper = [c for c in curves if c["C"] >= UPPER_FROM]
    pu = fit_lnd(fit_points(upper)) if len(upper) >= 2 else None

    def row(N, D):
        return dict(N=N, D=D, predicted=L_of(p, N, D), predicted_upper=L_of(pu, N, D) if pu else None,
                    range=[min(L_of(q, N, D) for q in loro), max(L_of(q, N, D) for q in loro)] if loro else None)

    out = dict(lnd=p, lnd_loro=loro, lnd_upper=pu, upper_from=UPPER_FROM, gpt2={})
    for name, (d, L) in GPT2.items():
        N, fpt = size(d, L)
        r = out["gpt2"][name] = dict(row(N, LLMC_TOKENS), C=fpt * LLMC_TOKENS, tokens_per_param=LLMC_TOKENS / N)
        E, A, al = f2["floor"]
        r["n_opt_at_C"] = 10 ** (f2["b0"] + f2["a"] * math.log10(r["C"]))
        r["compute_optimal_loss_at_C"] = E + A * r["C"] ** -al
    if check:
        out["check"] = dict(row(size(check[0])[0], check[1]), width=check[0])
    return out


def report(pred):
    p = pred["lnd"]
    print(f"L(N,D) = {p['E']:.3f} + {p['A']:.3g}/N^{p['alpha']:.3f} + {p['B']:.3g}/D^{p['beta']:.3f}  ({p['n_points']} points)")
    for name, r in pred["gpt2"].items():
        rng = f" [{r['range'][0]:.3f}, {r['range'][1]:.3f}]" if r["range"] else ""
        up = f", upper rungs only {r['predicted_upper']:.3f}" if r["predicted_upper"] else ""
        print(f"  GPT-2 {name:>5} ({r['tokens_per_param']:.0f} tok/param): {r['predicted']:.3f}{rng}{up}")


# ---------------- everything, in order of value per GPU-hour
def run_all(data_dir, out_dir, device, max_hours, commit=lambda: None,
            rungs=RUNGS, widths=WIDTHS, calib=CALIB, check=CHECK):
    lad = Ladder(Data(data_dir), device, max_hours, widths)
    s = dict(rungs=rungs, widths=widths, curves=[])
    os.makedirs(out_dir, exist_ok=True)

    def save():                                                    # after every stage: a kill loses little
        s["runs"] = list(lad.cache.values())
        json.dump(s, open(f"{out_dir}/summary.json", "w"), indent=1)
        commit()

    # 1-2. the lr rule, then the rungs, cheapest first, while the next rung fits in the time left
    s["calibration"] = lad.calibrate(*calib)
    start = 0
    for C in rungs:
        if not lad.fits(C, 6 * seeds(C)):
            s["stopped"] = f"out of time before rung {C:.1e}"
            break
        print(f"rung C={C:.1e} (batch {batch_seqs(C)} x {BLOCK} tokens)", flush=True)
        cv = lad.curve(C, start)
        s["curves"].append(cv)
        start = max(0, cv["k"] - 1)                                 # N_opt grows with C: start near the last one
        print(f"  -> N_opt {cv['n_opt']:,.0f}, L* {cv['l_opt']:.4f}, bracketed={cv['bracketed']}", flush=True)
        save()

    # 3-4. fit, write the predictions down, and only then train the check run
    s["fit"] = fit(s["curves"])
    w, C = check
    s["predictions"] = predictions(s["curves"], s["fit"], check=(w, lad.tokens(w, C)))
    report(s["predictions"])
    save()
    if lad.fits(C, 1):
        r = lad.run(w, C, lad.lr(w, C))
        s["check"] = dict(r, predicted=s["predictions"]["check"]["predicted"])
        print(f"check: actual {r['val']:.4f} vs predicted {s['check']['predicted']:.4f}", flush=True)
        save()

    # 5. leftover time: seed noise first (a probe 0.01 better means nothing without it), then lr probes
    for cv in s["curves"]:
        if seeds(cv["C"]) == 1 and lad.fits(cv["C"], 1):
            lad.repeat(cv)
    for cv in s["curves"]:
        if lad.fits(cv["C"], 2 * seeds(cv["C"])):
            lad.probes(cv)
    save()
    return s
