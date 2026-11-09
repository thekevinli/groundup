import json
import math
import os
import time
import urllib.request

import numpy as np
import torch
from torch.nn import functional as F

from gpt import GPT

# IsoFLOP sweep (Chinchilla approach 2). For each compute budget C: walk model sizes from small to
# large, train each on D = C / flops_per_token tokens, and stop two sizes past the lowest loss. The
# curve falls, flattens and turns up; only its bottom matters. Fit a parabola in log N to the bottom
# five points: the vertex is N_opt(C). Then fit N_opt ~ C^a and L*(C) = E + A * C^-alpha.
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(URL, "input.txt")
text = open("input.txt").read()
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])
train_data, val_data = data[: int(0.9 * len(data))], data[int(0.9 * len(data)):]

# Rungs ~3x apart. 1e11 is ~1 epoch of the 1M training tokens: stop before data repetition changes
# the regime. 1e8 is run and plotted but left out of the fit: its runs are 30-160 steps, every size
# sits near the unigram baseline (3.35 nats), and seed noise (~0.1) is as large as the U is deep.
BUDGETS = [1e8, 1e9, 3e9, 1e10, 3e10, 1e11]
FIT_FROM = 1e9                                   # rungs below this are shown, not fitted
WIDTHS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]   # N grows ~1.5x per step, depth fixed
N_LAYER, BLOCK = 2, 64
g = torch.Generator().manual_seed(1)
ix = torch.randint(len(val_data) - BLOCK - 1, (512,), generator=g)        # one fixed eval set
VAL_X = torch.stack([val_data[i: i + BLOCK] for i in ix])
VAL_Y = torch.stack([val_data[i + 1: i + BLOCK + 1] for i in ix])


def batch_size(C: float) -> int:
    return round(2 * (C / 1e8) ** 0.3)            # sequences per step, B ~ C^0.3 (DeepSeek: C^0.33)


def seeds(C: float) -> int:
    return 8 if C < 1e9 else 4 if C < 1e10 else 2  # small rungs are noisier and cheaper: more seeds


def plan(width: int, C: float):
    model = GPT(len(chars), BLOCK, N_LAYER, width, num_head=max(1, width // 8))
    N = sum(p.numel() for p in model.parameters())   # tied embedding counted once: it's the LM head
    flops_per_token = 6 * N + 12 * N_LAYER * BLOCK * width                # weights + attention
    B = batch_size(C)
    steps = int(C / (flops_per_token * B * BLOCK))
    return model, N, B, steps, steps * B * BLOCK


REFS = [dict(width=8, C=1e9), dict(width=8, C=1e10)]   # tune lr here, fit the rule, apply it
LR0, T_EXP = None, None


def lr_rule(width: int, B: int, T: int) -> float:
    # The peak lr is a function of the run, not a constant:
    #   sqrt(B)       Adam's square-root batch scaling
    #   T^-T_EXP      longer runs want a lower peak (Marin uses 0.3; here it is fitted)
    #   1/width       muP: with plain Adam, hidden-layer lr ~ 1/fan_in
    _, _, B0, _, T0 = plan(REFS[0]["width"], REFS[0]["C"])
    return LR0 * (REFS[0]["width"] / width) * (B / B0) ** 0.5 * (T0 / T) ** T_EXP


def train(width: int, C: float, lr: float = None, lr_mult: float = 1.0) -> dict:
    vals = []
    for seed in range(seeds(C)):                  # average seeds: the curve's noise is the fit's noise
        torch.manual_seed(seed)                   # same seeds for every size: comparisons share noise
        model, N, B, steps, T = plan(width, C)
        lr = (lr or lr_rule(width, B, T)) if lr_mult == 1.0 else lr_rule(width, B, T) * lr_mult
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        for step in range(steps):
            # warmup 10%, cosine to 10%, ending exactly at this run's budget (no mid-schedule reads)
            opt.param_groups[0]["lr"] = lr * min(1, (step + 1) / max(1, steps // 10)) * (
                0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
            i = torch.randint(len(train_data) - BLOCK - 1, (B,))
            x = torch.stack([train_data[j: j + BLOCK] for j in i])
            y = torch.stack([train_data[j + 1: j + BLOCK + 1] for j in i])
            loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            vals.append(F.cross_entropy(model(VAL_X).float().flatten(0, 1), VAL_Y.flatten()).item())
    return dict(C=C, width=width, N=N, D=T, steps=steps, lr=lr, val=float(np.mean(vals)),
                sem=float(np.std(vals, ddof=1) / np.sqrt(len(vals))))


def best_lr(width: int, C: float, lr: float = 0.04) -> float:
    # bracket the best lr on a sqrt(2) grid: step outward until the best value is interior
    r2 = 2 ** 0.5
    runs = {lr: train(width, C, lr)["val"] for lr in (lr / r2, lr, lr * r2)}
    while True:
        best = min(runs, key=runs.get)
        if best not in (min(runs), max(runs)):
            return best
        new = best * r2 if best == max(runs) else best / r2
        runs[new] = train(width, C, new)["val"]


def calibrate():
    # same width at two budgets: the lr change not explained by sqrt(B) gives the token exponent
    (w, C1), (_, C2) = [(r["width"], r["C"]) for r in REFS]
    lr1, lr2 = best_lr(w, C1), best_lr(w, C2)
    (_, _, B1, _, T1), (_, _, B2, _, T2) = plan(w, C1), plan(w, C2)
    t_exp = np.log((B2 / B1) ** 0.5 * lr1 / lr2) / np.log(T2 / T1)
    return lr1, float(np.clip(t_exp, 0.0, 0.6))


def isoflop_curve(C: float, start: int) -> dict:
    pts = {}                                       # width index -> result
    argmin = lambda: min(pts, key=lambda j: pts[j]["val"])
    i = start
    while i < len(WIDTHS):                         # walk right; stop 2 sizes past the minimum
        pts[i] = train(WIDTHS[i], C)
        if max(pts) - argmin() >= 2:
            break
        i += 1
    while min(pts) > 0 and argmin() - min(pts) < 2:   # and make sure 2 sizes sit left of it
        pts[min(pts) - 1] = train(WIDTHS[min(pts) - 1], C)
    k = argmin()
    near = [pts[j] for j in sorted(pts) if abs(j - k) <= 2]
    a, b, c = np.polyfit(np.log10([r["N"] for r in near]), [r["val"] for r in near], 2)
    n_opt = 10 ** (-b / (2 * a)) if a > 0 else pts[k]["N"]
    # validate the lr rule at the bottom of the U: half and double the rule's lr
    probes = [train(WIDTHS[k], C, lr_mult=m) for m in (0.5, 2.0)]
    return dict(C=C, points=[pts[j] for j in sorted(pts)], fit=[a, b, c], k=k,
                n_opt=n_opt, l_opt=float(np.polyval([a, b, c], np.log10(n_opt))), probes=probes)


def fit_floor(C, L):
    # L*(C) = E + A * C^-alpha with 0 <= E < min(L): an unconstrained E can go unphysical
    best = None
    for E in np.linspace(0, min(L) - 1e-3, 400):
        slope, icpt = np.polyfit(np.log(C), np.log(np.array(L) - E), 1)
        err = np.sum((E + np.exp(icpt) * np.array(C) ** slope - L) ** 2)
        if best is None or err < best[0]:
            best = (err, E, np.exp(icpt), -slope)
    return best[1:]


if __name__ == "__main__":
    t0 = time.time()
    LR0, T_EXP = calibrate()
    print(f"lr0 = {LR0:.3g}, token exponent {T_EXP:.2f}  [{time.time() - t0:.0f}s]")
    curves, start = [], 0
    for C in BUDGETS:
        cv = isoflop_curve(C, start)
        curves.append(cv)
        start = max(0, cv["k"] - 1)                 # N_opt grows with C: start the next rung near it
        p = cv["points"]
        print(f"C={C:.0e} B={batch_size(C)}: " + "  ".join(f"{r['N']}:{r['val']:.3f}±{r['sem']:.3f}" for r in p)
              + f"  ->  N_opt {cv['n_opt']:,.0f}, L* {cv['l_opt']:.3f}; rule lr x0.5/x2: "
              + "/".join(f"{q['val'] - min(r['val'] for r in p):+.3f}" for q in cv["probes"])
              + f"  [{time.time() - t0:.0f}s]")
    fitted = [cv for cv in curves if cv["C"] >= FIT_FROM]
    Cs = [cv["C"] for cv in fitted]
    a = np.polyfit(np.log10(Cs), np.log10([cv["n_opt"] for cv in fitted]), 1)[0]
    d_opt = [C / (6 * cv["n_opt"]) for C, cv in zip(Cs, fitted)]
    b = np.polyfit(np.log10(Cs), np.log10(d_opt), 1)[0]
    E, A, alpha = fit_floor(Cs, [cv["l_opt"] for cv in fitted])
    print(f"N_opt ~ C^{a:.2f}, D_opt ~ C^{b:.2f} (Chinchilla: 0.5, 0.5); L* = {E:.2f} + {A:.2f} C^-{alpha:.3f}")
    json.dump({"lr0": LR0, "t_exp": T_EXP, "fit_from": FIT_FROM, "curves": curves, "a": a, "b": b, "floor": [E, A, alpha],
               "seconds": time.time() - t0}, open("toy_isoflop.json", "w"))
