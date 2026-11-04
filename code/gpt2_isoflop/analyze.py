"""Plot and check the GPT-2 ladder:  python analyze.py <dir with summary.json | isoflop_gpu_full.json>
-> ladder.png next to it, and a report.

Runs on a laptop (numpy, scipy, matplotlib) next to ladder.py; no GPU. Refits everything from the
saved runs, so you can change a fit choice and see what moves.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ladder as L

LLMC_124M = (3.26, 3.29)          # llm.c's GPT-2 124M on FineWeb 10B: published final val loss, about 3.26-3.29


def main(path: str):
    if path.endswith(".json"):                       # isoflop_gpu_full.json, as `modal run` writes it locally
        s, d = json.load(open(path))["summary"], os.path.dirname(os.path.abspath(path))
    else:                                            # a folder with summary.json from the volume
        s, d = json.load(open(os.path.join(path, "summary.json"))), path
    curves = s["curves"]
    f2 = L.fit(curves)
    c = s.get("check", {})
    pred = L.predictions(curves, f2, check=(c["width"], c["D"]) if "D" in c else None)
    L.report(pred)

    p = pred["lnd"]
    pts = L.fit_points(curves)
    res = [q["val"] - L.L_of(p, q["N"], q["D"]) for q in pts]
    print(f"Approach 3 residuals on the {len(pts)} fitted points: rms {np.sqrt(np.mean(np.square(res))):.4f}, "
          f"max {np.max(np.abs(res)):.4f}")
    if f2:
        print(f"Approach 2: N_opt ~ C^{f2['a']:.3f} (Chinchilla ~0.5, Kaplan 0.73); D_opt ~ C^{1 - f2['a']:.3f}")
    for cv in curves:
        rep = cv.get("repeat")
        noise = f"seed 1 - seed 0 = {rep['diff']:+.4f}" if rep else (f"sem {cv['points'][0]['sem']:.4f}" if cv["points"][0].get("sem") else "no noise estimate")
        probes = ", ".join(f"{q['lr'] / cv['points'][[r['width'] for r in cv['points']].index(q['width'])]['lr']:.1f}x lr: "
                           f"{q['val'] - min(r['val'] for r in cv['points']):+.4f}" for q in cv.get("probes", []))
        print(f"  rung {cv['C']:.1e}: N_opt {cv['n_opt'] / 1e6:6.2f}M, bracketed={cv['bracketed']}, {noise}"
              + (f"; probes vs minimum: {probes}" if probes else ""))
    if "val" in c:
        print(f"check run (width {c['width']}, {c['D'] / 1e6:.0f}M tokens): actual {c['val']:.4f}, "
              f"pre-registered {c['predicted']:.4f}, refit now {pred['check']['predicted']:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(curves)))

    # 1. the IsoFLOP curves: walked points, parabola through the bottom, vertex (star), lr probes (x)
    for cv, col in zip(curves, cols):
        N, y = [q["N"] for q in cv["points"]], [q["val"] for q in cv["points"]]
        ax[0].plot(N, y, "o", color=col, ms=4, label=f"{cv['C']:.1e} FLOPs")
        if cv["fit"][0] > 0:
            xs = np.linspace(math.log10(min(N)), math.log10(max(N)), 60)
            ax[0].plot(10 ** xs, np.polyval(cv["fit"], xs), "-", color=col, lw=1)
        ax[0].plot(cv["n_opt"], cv["l_opt"], "*", color=col, ms=12, mec="k", mew=0.5)
        for q in cv.get("probes", []):
            ax[0].plot(q["N"], q["val"], "x", color=col, ms=6)
    ax[0].set(xscale="log", xlabel="parameters N (total)", ylabel="val loss (nats/token)", title="IsoFLOP curves")
    ax[0].legend(fontsize=7)

    # 2. compute-optimal size vs compute, extrapolated to the 124M run's compute
    C124 = pred["gpt2"]["124M"]["C"]
    ax[1].loglog([cv["C"] for cv in curves], [cv["n_opt"] for cv in curves], "o", color="k", label="ladder vertices")
    if f2:
        Cs = np.logspace(math.log10(min(cv["C"] for cv in curves)), math.log10(C124) + 0.2, 50)
        ax[1].loglog(Cs, 10 ** (f2["b0"] + f2["a"] * np.log10(Cs)), "-", color="C0", label=f"fit: N_opt ~ C^{f2['a']:.2f}")
    ax[1].loglog([C124], [pred["gpt2"]["124M"]["N"]], "s", color="C3", label="GPT-2 124M as trained (79 tok/param)")
    ax[1].set(xlabel="compute C (FLOPs)", ylabel="N_opt", title="Compute-optimal size")
    ax[1].legend(fontsize=7)

    # 3. the extrapolation that matters: the 124M shape's loss vs tokens
    N124 = pred["gpt2"]["124M"]["N"]
    D = np.logspace(7.5, 10.2, 80)
    ax[2].plot(D, [L.L_of(p, N124, x) for x in D], "-", color="C0", label="all-rung fit")
    if pred["lnd_loro"]:
        lo = [min(L.L_of(q, N124, x) for q in pred["lnd_loro"]) for x in D]
        hi = [max(L.L_of(q, N124, x) for q in pred["lnd_loro"]) for x in D]
        ax[2].fill_between(D, lo, hi, color="C0", alpha=0.15, label="leave-one-rung-out range")
    if pred["lnd_upper"]:
        ax[2].plot(D, [L.L_of(pred["lnd_upper"], N124, x) for x in D], "--", color="C1",
                   label=f"upper-rung fit (C >= {pred['upper_from']:.1e})")
    if "val" in c and c["width"] == 768:
        ax[2].plot(c["D"], c["val"], "o", color="C3", ms=8, label="check run (actual)")
    ax[2].fill_between([L.LLMC_TOKENS * 0.93, L.LLMC_TOKENS * 1.07], *LLMC_124M, color="C2", alpha=0.6,
                       label="llm.c GPT-2 124M (published)")
    ax[2].set(xscale="log", xlabel="training tokens D", ylabel="val loss (nats/token)",
              title="GPT-2 124M shape: predicted loss vs tokens")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(d, "ladder.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
