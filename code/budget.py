# Back-of-envelope memory and communication per GPU when training Llama-3-70B.
# Training state: bf16 weights + bf16 grads + fp32 master + fp32 Adam m, v = 2 + 2 + 4 + 4 + 4 = 16 B/param.
# Activations (Korthikanti et al. 2022, FlashAttention so no S^2 term): ~34 * s * b * h bytes per layer;
# with full recompute only each layer's input is kept: 2 * s * b * h bytes per layer.
GB = 1e9
L, h, ffn, H, H_kv, V = 80, 8192, 28672, 64, 8, 128256          # Llama-3-70B
hd = h // H
layer = 2 * h * h + 2 * h * H_kv * hd + 3 * h * ffn             # q, o + k, v (GQA) + SwiGLU (gate, up, down)
Psi = L * layer + 2 * V * h                                     # + embedding and LM head = 70.6B


def memory(s, b=1, dp=1, tp=1, cp=1, recompute=False):
    shard = dp * cp if dp > 1 else 1             # FSDP shards over every non-TP rank (dp and cp ranks)
    states = 16 * Psi / (tp * shard)             # TP splits each weight; FSDP splits what's left
    gathered = 2 * 2 * layer / tp if shard > 1 else 0         # FSDP: current + prefetched unit, bf16
    tokens = s * b / cp                                         # CP: each rank holds s / cp tokens
    per_layer = (2 if recompute else 34) * tokens * h / tp     # SP: the non-TP regions are split by tp too
    acts = L * per_layer + (34 * tokens * h / tp if recompute else 0)   # + one layer being recomputed
    return states / GB, (gathered + acts) / GB


def comm(s, b=1, dp=1, tp=1, cp=1):
    # bytes each GPU sends per optimizer step (one micro-batch); ring collectives send (n-1)/n of the data
    out = {}
    shard = dp * cp if dp > 1 else 1
    if shard > 1:   # FSDP: all-gather bf16 weights in forward and again in backward, reduce-scatter grads
        out["FSDP"] = 3 * (shard - 1) / shard * 2 * Psi / tp
    if tp > 1:      # per layer: 2 all-reduces forward + 2 backward (SP: RS + AG, same bytes) of b*s*h
        out["TP/SP"] = L * 4 * 2 * (tp - 1) / tp * 2 * b * (s / cp) * h
    if cp > 1:      # ring: K and V of my slice, cp - 1 hops forward; backward resends K, V plus dK, dV
        out["CP"] = L * 3 * 2 * (cp - 1) * 2 * b * (s / cp) * H_kv * hd / tp
    return {k: v / GB for k, v in out.items()}


def moe_ep(s=8192, ep=1, dp=1):
    # EP on a Mixtral-8x7B-shaped MoE (8 experts, top-2): experts are split over ep ranks, attention
    # isn't; FSDP shards the rest. Each MoE layer does 4 all-to-alls (dispatch, combine; fwd and bwd).
    Lm, hm, ffm, E, k, Vm = 32, 4096, 14336, 8, 2, 32000
    expert, attn = 3 * hm * ffm, 2 * hm * hm + 2 * hm * 8 * 128
    total = Lm * (attn + E * expert) + 2 * Vm * hm
    states = 16 * (Lm * (attn + E * expert / ep) + 2 * Vm * hm) / dp
    a2a = Lm * 4 * k * s * hm * 2 * (ep - 1) / ep                 # bf16 token rows that leave the rank
    return total / 1e9, states / GB, a2a / GB


if __name__ == "__main__":
    print(f"Llama-3-70B: {Psi / 1e9:.1f}B params -> {16 * Psi / GB:,.0f} GB of training state alone\n")
    ladder = [
        ("1 GPU, 8k context", dict(s=8192)),
        ("+ FSDP over 64 GPUs", dict(s=8192, dp=64)),
        ("+ TP/SP 8 (512 GPUs)", dict(s=8192, dp=64, tp=8)),
        ("128k context, same layout", dict(s=131072, dp=64, tp=8)),
        ("+ CP 16 (8192 GPUs)", dict(s=131072, dp=64, tp=8, cp=16)),
    ]
    print(f"{'setup':28s} {'states':>8s} {'activations':>12s} {'total':>8s}   comm per step")
    for name, cfg in ladder:
        st, act = memory(**cfg)
        c = comm(**cfg)
        cs = ", ".join(f"{k} {v:,.0f} GB" for k, v in c.items()) or "none"
        print(f"{name:28s} {st:7.1f}G {act:11.1f}G {st + act:7.1f}G   {cs}")

    print("\nMixtral-8x7B-shaped MoE, 8k tokens per GPU:")
    for ep, dp in [(1, 1), (8, 1), (8, 8)]:
        total, st, a2a = moe_ep(ep=ep, dp=dp)
        print(f"  EP {ep}, FSDP {dp}: {total:.1f}B params, states {st:6.1f} GB/GPU, all-to-all {a2a:5.1f} GB/step")
