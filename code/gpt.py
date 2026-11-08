import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from collectives import CUDA, init, launch, ranks
from cp import all_gather_kv
from tp import ColumnParallelLinear, RowParallelLinear, enter_sp, exit_sp


class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize over the feature dim (BatchNorm: over batch, InstanceNorm: over sequence)
        xf = x.float()                                                    # statistics in fp32, even for bf16 x
        mean = xf.mean(dim=-1, keepdim=True)
        var = xf.var(dim=-1, unbiased=False, keepdim=True)
        return (self.scale * (xf - mean) / torch.sqrt(var + self.eps) + self.shift).to(x.dtype)


def sinusoidal_pe(seq_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    # PE[p, 2i] = sin(p * w_i), PE[p, 2i+1] = cos(p * w_i), w_i = base^(-2i/dim): fast -> slow
    assert dim % 2 == 0
    w = base ** (-torch.arange(0, dim, 2).float() / dim)              # (dim/2,)
    angles = torch.outer(torch.arange(seq_len).float(), w)            # (S, dim/2): angles[p, i] = p * w_i
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe   # (S, dim), built once: GPT keeps it as a buffer (follows .to(device)), indexes it by
                # pos (packed rows, KV-cache offsets) and casts to x's dtype


def apply_rope(x: torch.Tensor, pos: torch.Tensor = None, base: float = 10000.0,
               scale: float = 1.0, method: str = "none", train_len: int = 2048) -> torch.Tensor:
    # x: (B, H, S, hd). Rotate each pair (x[2i], x[2i+1]) by angle pos * theta_i, theta_i = base^(-2i/hd)
    B, H, S, D = x.shape
    if pos is None:
        pos = torch.arange(S, device=x.device)                        # positions 0..S-1
    if method == "ntk":     # NTK-aware: raise the base; high freqs barely move, lowest slows by `scale`
        base = base * scale ** (D / (D - 2))
    theta = base ** (-torch.arange(0, D, 2, device=x.device).float() / D)   # (hd/2,): pair 0 spins fastest
    if method == "pi":      # position interpolation: slow down every frequency by `scale`
        theta = theta / scale
    elif method == "yarn":  # YaRN: slow down only dims that turn < 1 time in train_len, ramp up to 32
        turns = train_len * theta / (2 * math.pi)
        keep = ((turns - 1) / (32 - 1)).clamp(0, 1)      # 1 = high freq, keep; 0 = low freq, interpolate
        theta = keep * theta + (1 - keep) * theta / scale
    angles = pos[..., None].float() * theta                           # (S, hd/2)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    mscale = 0.1 * math.log(scale) + 1 if method == "yarn" else 1.0   # YaRN: sharper attention, logits x mscale^2
    return (out * mscale).to(x.dtype)                                 # angles are fp32


def linear_attention(q, k, v, state=None):
    # softmax(q k^T) v without the softmax is associative: q (k^T v). Causal, it becomes a recurrence on a
    # (hd, hd) state: S_t = S_{t-1} + phi(k_t)^T v_t,  o_t = phi(q_t) S_t / phi(q_t) z_t  (z: the normalizer)
    q, k = F.elu(q) + 1, F.elu(k) + 1                                 # phi > 0: o_t is a weighted average of v
    B, H, T, d = q.shape
    S, z = state if state is not None else (q.new_zeros(B, H, d, v.shape[-1]), q.new_zeros(B, H, d))
    out = []
    for t in range(T):                                                # constant memory per token, like an RNN
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]             # + k_t^T v_t (outer product)
        z = z + k[:, :, t]
        out.append((q[:, :, t, None] @ S).squeeze(2) / (q[:, :, t] * z).sum(-1, keepdim=True))
    return torch.stack(out, dim=2), (S, z)


def delta_rule(q, k, v, beta, state=None):
    # DeltaNet: write the error, not the value. Read what S stores for k_t and move it toward v_t:
    # S_t = S_{t-1} + beta_t k_t^T (v_t - k_t S_{t-1}). With |k_t| = 1 and beta_t = 1 it overwrites.
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    B, H, T, d = q.shape
    S = state if state is not None else q.new_zeros(B, H, d, v.shape[-1])
    out = []
    for t in range(T):
        kt = k[:, :, t, None]                                         # (B, H, 1, hd)
        err = v[:, :, t, None] - kt @ S                               # new value minus the stored one
        S = S + kt.transpose(-2, -1) @ (beta[:, :, t, None, None] * err)
        out.append((q[:, :, t, None] @ S).squeeze(2))
    return torch.stack(out, dim=2), S


class MHSA(nn.Module):

    def __init__(self, feat_dim: int, num_head: int, num_kv_head: int = None, tp_group=None, cp_group=None,
                 kind: str = "softmax"):
        super().__init__()
        assert feat_dim % num_head == 0
        self.cp_group = cp_group
        tp = ranks(tp_group)[1] if tp_group else 1
        num_kv_head = num_kv_head or num_head                             # GQA: H_kv < H, MQA: H_kv = 1
        assert num_head % num_kv_head == 0 and num_kv_head % tp == 0
        self.num_head, self.num_kv_head = num_head // tp, num_kv_head // tp   # TP: my heads only
        self.head_dim = feat_dim // num_head
        self.kind = kind                                                  # "softmax", "linear" or "delta"
        assert kind == "softmax" or (tp == 1 and not cp_group), "linear attention here is single-GPU"
        self.q_dim, self.kv_dim = self.num_head * self.head_dim, self.num_kv_head * self.head_dim
        # q: F, k and v: H_kv * hd each. TP: columns = my heads' q, k, v; rows of out = my heads
        self.qkv = ColumnParallelLinear(feat_dim, feat_dim + 2 * num_kv_head * self.head_dim, tp_group, True)
        self.beta = nn.Linear(feat_dim, self.num_head) if kind == "delta" else None   # per-head write strength
        self.out = RowParallelLinear(feat_dim, feat_dim, tp_group, True)

    def forward(self, x: torch.Tensor, is_causal: bool = False, rope=None, pos=None, attn_mask=None,
                cache=None) -> torch.Tensor:
        # x: (B, S, F), my slice of S under SP; rope: None or apply_rope settings; pos: positions; attn_mask: bool, True = may attend
        # cache: this layer's {"k", "v"} while decoding, or None
        q, k, v = self.qkv(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = rearrange(q, "b s (h f) -> b h s f", h=self.num_head)
        k = rearrange(k, "b s (h f) -> b h s f", h=self.num_kv_head)
        v = rearrange(v, "b s (h f) -> b h s f", h=self.num_kv_head)
        if self.kind != "softmax":                                        # no S x S scores: a recurrence
            assert attn_mask is None, "packed rows: reset the state at each document start instead"
            rep = self.num_head // self.num_kv_head
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            state = cache.get("state") if cache is not None else None
            if self.kind == "linear":
                o, state = linear_attention(q, k, v, state)
            else:
                o, state = delta_rule(q, k, v, torch.sigmoid(self.beta(x)).transpose(1, 2), state)
            if cache is not None:
                cache["state"] = state                                    # its whole cache: a fixed size
            return self.out(rearrange(o, "b h s f -> b s (h f)"))
        if rope is not None:                                              # rotate q and k, never v
            q, k = apply_rope(q, pos, **rope), apply_rope(k, pos, **rope)
        if cache is not None:                                             # decoding: past k, v + the new ones
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v                                 # H_kv heads: GQA's saving lands here
        if self.cp_group:                                                 # CP: every rank's keys and values
            k, v = (t.transpose(1, 2) for t in all_gather_kv(k.transpose(1, 2), v.transpose(1, 2), self.cp_group))
        # each K/V head serves H / H_kv query heads: same attention FLOPs, H / H_kv smaller KV cache
        k = k.repeat_interleave(self.num_head // self.num_kv_head, dim=1)
        v = v.repeat_interleave(self.num_head // self.num_kv_head, dim=1)
        # q.k sums head_dim terms, so its std grows like sqrt(head_dim): rescale to keep softmax soft
        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)       # (B, H, S_q, S_k)
        if attn_mask is None and is_causal:                               # the queries are the last S_q keys
            Sq, Sk = q.shape[2], k.shape[2]
            attn_mask = torch.ones(Sq, Sk, dtype=torch.bool, device=x.device).tril(diagonal=Sk - Sq)
        if attn_mask is not None:
            score = score.masked_fill(~attn_mask, float("-inf"))          # before softmax
        attn = torch.softmax(score.float(), dim=-1).to(v.dtype) @ v       # softmax in fp32; (B, H, S, hd)
        return self.out(rearrange(attn, "b h s f -> b s (h f)"))


class MoE(nn.Module):
    # token choice: every token picks its top-k experts; output = gate-weighted sum of their MLPs
    def __init__(self, feat_dim: int, num_expert: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(feat_dim, num_expert, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, 4 * feat_dim), nn.GELU(), nn.Linear(4 * feat_dim, feat_dim))
            for _ in range(num_expert)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])                                     # (T, F): routing is per token
        probs = torch.softmax(self.router(x), dim=-1)                    # (T, E)
        gate, idx = probs.topk(self.top_k, dim=-1)                       # (T, k)
        gate = gate / gate.sum(dim=-1, keepdim=True)                     # renormalize over chosen experts
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)                # tokens that chose expert e
            if len(tok):
                out = out.index_add(0, tok, gate[tok, slot, None] * expert(x[tok]))
        # Switch-style balance loss E * sum_e f_e * p_e (f: share of tokens routed, p: mean prob)
        f = torch.bincount(idx.flatten(), minlength=len(self.experts)).float() / idx.numel()
        self.aux_loss = len(self.experts) * (f * probs.mean(dim=0)).sum()
        return out.reshape(shape)


class TransformerBlock(nn.Module):

    def __init__(self, feat_dim: int, num_head: int, num_kv_head: int = None, use_rope: bool = True,
                 num_expert: int = 0, dropout: float = 0.0, tp_group=None, cp_group=None, attn: str = "softmax"):
        super().__init__()
        assert not (num_expert and tp_group), "MoE layers use expert parallelism, not TP"
        self.use_rope = use_rope                     # False = a NoPE layer: no position signal here
        self.drop = nn.Dropout(dropout)                                   # on each branch's output
        self.ln1 = LayerNorm(feat_dim)
        self.attention = MHSA(feat_dim, num_head, num_kv_head, tp_group, cp_group, kind=attn)
        self.ln2 = LayerNorm(feat_dim)
        self.mlp = MoE(feat_dim, num_expert) if num_expert else nn.Sequential(
            ColumnParallelLinear(feat_dim, feat_dim * 4, tp_group, True),
            nn.GELU(),
            RowParallelLinear(feat_dim * 4, feat_dim, tp_group, True),
        )
        for p in [*self.ln1.parameters(), *self.ln2.parameters()]:      # SP: norms see my slice of S
            p.sequence_parallel = tp_group is not None

    def forward(self, x: torch.Tensor, is_causal: bool = False, rope=None, pos=None, attn_mask=None,
                cache=None) -> torch.Tensor:
        # pre-norm: the residual stream x is never normalized, so gradients have an identity path
        if not self.use_rope:
            rope = None                                                   # NoPE layer
        x = x + self.drop(self.attention(self.ln1(x), is_causal, rope, pos, attn_mask, cache))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_layer: int, feat_dim: int, num_head: int,
                 pos_emb: str = "rope", rope_scale: float = 1.0, rope_method: str = "none",
                 dropout: float = 0.0, nope_every: int = 0, num_kv_head: int = None,
                 num_expert: int = 0, linear: str = None, full_every: int = 0, grad_ckpt: bool = False,
                 tp_group=None, cp_group=None):
        super().__init__()
        self.grad_ckpt = grad_ckpt                   # recompute each block in backward to save memory
        self.tp_group = tp_group                     # TP + SP inside this group; None = one GPU
        self.cp_group = cp_group                     # CP: each rank holds one chunk of the sequence
        self.pos_emb = pos_emb                       # "none" = NoPE everywhere (Kimi Linear's MLA layers)
        self.max_len = int(block_size * rope_scale)                    # context allowed at inference
        self.tok_emb = nn.Embedding(vocab_size, feat_dim)
        nope = lambda i: nope_every and (i + 1) % nope_every == 0      # 4: every 4th layer (Llama 4, SmolLM3)
        # linear="delta", full_every=4: three DeltaNet layers per softmax layer (Kimi Linear, Qwen3-Next)
        kind = lambda i: linear if linear and (not full_every or (i + 1) % full_every) else "softmax"
        self.blocks = nn.ModuleList([TransformerBlock(feat_dim, num_head, num_kv_head, not nope(i), num_expert,
                                                      dropout, tp_group, cp_group, attn=kind(i))
                                     for i in range(n_layer)])
        self.ln_f = LayerNorm(feat_dim)
        self.lm_head = nn.Linear(feat_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                       # weight tying
        if pos_emb == "sin":
            self.register_buffer("pe", sinusoidal_pe(self.max_len, feat_dim), persistent=False)
        # apply_rope settings for every layer; None = no RoPE
        self.rope = dict(scale=rope_scale, method=rope_method, train_len=block_size) if pos_emb == "rope" else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, seqlist=None, cache=None):
        S = idx.shape[1]                                               # my CP chunk (all of it without CP)
        c, cp = ranks(self.cp_group)[:2] if self.cp_group else (0, 1)
        past = max((kv["k"].shape[2] for kv in cache if "k" in kv), default=0) if cache else 0   # tokens cached
        x = self.tok_emb(idx)                                          # (B, S, F)
        q_pos = past + c * S + torch.arange(S, device=idx.device)      # global positions of my tokens
        pos = q_pos
        mask = None                                                    # plain causal
        if seqlist is not None:                                        # one packed row, from pack()
            mask, pos = packed_mask_pos(seqlist)
        if cp > 1:                                                     # CP: my queries vs every rank's keys
            assert seqlist is None and cache is None, "packing with CP needs each rank's rows of the mask"
            mask = torch.arange(S * cp, device=idx.device)[None, :] <= q_pos[:, None]   # causal by position
        if self.pos_emb == "sin":
            x = x + self.pe[pos].to(x.dtype)
        if self.tp_group:
            x = enter_sp(x, self.tp_group)                               # SP: my slice of the sequence
        for i, block in enumerate(self.blocks):
            if self.grad_ckpt and self.training:     # keep only the block input; rerun the block in backward
                x = checkpoint(block, x, True, self.rope, pos, mask, use_reentrant=False)
            else:
                x = block(x, is_causal=True, rope=self.rope, pos=pos, attn_mask=mask, cache=cache[i] if cache else None)
        if self.tp_group:
            x = exit_sp(x, self.tp_group)                                # full sequence for the head
        logits = self.lm_head(self.ln_f(x))                            # (B, S, vocab)
        loss = None
        if targets is not None:                                        # -100 = no loss; fp32 loss
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
            aux = [b.mlp.aux_loss for b in self.blocks if isinstance(b.mlp, MoE)]
            if aux:                                                    # keep the experts load-balanced
                loss = loss + 0.01 * sum(aux)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        cache, new = None, idx
        for _ in range(max_new_tokens):
            if cache is None or idx.shape[1] > self.max_len:           # start, or past the context: prefill
                cache, new = [{} for _ in self.blocks], idx[:, -self.max_len:]   # one {"k", "v"} per layer
            logits, _ = self(new, cache=cache)                         # else decode: only the newest token
            new = logits[:, -1].argmax(dim=-1, keepdim=True)           # greedy
            idx = torch.cat([idx, new], dim=1)
        return idx



def pack(tokens: torch.Tensor, pad_id: int = 0):
    # Pack (B, S) token IDs or (B, S, D) features in batch order.
    # For features, a position is padding only if all D values equal pad_id.
    # seqlist (B,) holds the number of retained positions in each sequence.
    if tokens.ndim not in (2, 3):
        raise ValueError("tokens must have shape (B, S) or (B, S, D)")

    valid = tokens != pad_id
    if tokens.ndim == 3:
        valid = valid.any(dim=-1)

    seqlist = valid.sum(dim=1)
    packed_tokens = tokens[valid].unsqueeze(0)

    return packed_tokens, seqlist


def packed_mask_pos(seqlist: torch.Tensor):
    # one packed row: attend only inside your own sequence (block-diagonal causal); positions restart at 0
    seq = torch.repeat_interleave(torch.arange(len(seqlist), device=seqlist.device), seqlist)   # (N,) which sequence
    pos = torch.arange(len(seq), device=seqlist.device) - (seqlist.cumsum(0) - seqlist)[seq]   # (N,) position in it
    mask = (seq[:, None] == seq[None, :]).tril()                                                # (N, N), True = may attend
    return mask, pos


# ---------------- check gradient checkpointing on one GPU: same gradients, less memory, more time
def saved_bytes(model: nn.Module, idx: torch.Tensor, targets: torch.Tensor) -> int:
    # activation memory: the tensors autograd keeps for backward, minus the weights, each storage once
    weights = {p.untyped_storage().data_ptr() for p in model.parameters()}
    kept = {}
    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in weights:
            kept[s.data_ptr()] = s.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = model(idx, targets)
    loss.backward()
    return sum(kept.values())


def _test_ckpt():
    import time
    cfg = dict(vocab_size=65, block_size=128, n_layer=6, feat_dim=384, num_head=6, dropout=0.2)   # train.py's
    idx, tgt = torch.randint(0, 65, (32, 128)), torch.randint(0, 65, (32, 128))                    # one batch
    out = {}
    for ckpt in (False, True):
        torch.manual_seed(0)                                           # same weights, same dropout masks
        model = GPT(**cfg, grad_ckpt=ckpt)
        mem = saved_bytes(model, idx, tgt)
        grads = [p.grad.clone() for p in model.parameters()]
        start = time.time()
        for _ in range(3):
            model(idx, tgt)[1].backward()
        out[ckpt] = mem, (time.time() - start) / 3, grads
    (mem0, sec0, g0), (mem1, sec1, g1) = out[False], out[True]
    diff = max((a - b).abs().max().item() for a, b in zip(g0, g1))
    assert diff < 1e-6                                                 # the recompute replays dropout's masks
    print(f"train.py's model (10.7M), one 32 x 128 batch, fp32: grad_ckpt=True vs False\n"
          f"  max gradient difference {diff:.1e}, with dropout 0.2\n"
          f"  kept for backward       {mem0 / 1e6:6.1f} MB -> {mem1 / 1e6:5.1f} MB\n"
          f"  forward + backward      {sec0:6.2f} s  -> {sec1:5.2f} s  ({sec1 / sec0:.2f}x)")


# ---------------- check the parallel GPT against one GPU, and measure memory per GPU
def mesh(dp: int, cp: int, tp: int) -> dict:
    # rank = (d * cp + c) * tp + t: TP innermost (NVLink), then CP, then DP.
    # new_group is collective: every rank creates every group, in the same order.
    import torch.distributed as dist
    r = dist.get_rank()
    m = dict(dp=dp, cp=cp, tp=tp, d=r // (cp * tp), c=r // tp % cp, t=r % tp)
    for dc in range(dp * cp):
        g = dist.new_group([dc * tp + t for t in range(tp)])
        if r // tp == dc:
            m["tp_group"] = g
    for d in range(dp):
        for t in range(tp):
            g = dist.new_group([(d * cp + c) * tp + t for c in range(cp)])
            if m["d"] == d and m["t"] == t:
                m["cp_group"] = g
    for t in range(tp):            # CP ranks hold the same weights: FSDP shards over DP x CP
        g = dist.new_group([dc * tp + t for dc in range(dp * cp)])
        if m["t"] == t:
            m["fsdp_group"] = g
    return m


def load_tp(model: GPT, ref: GPT, t: int, tp: int):
    # copy my TP slice of every single-GPU weight; q, k and v are sliced separately
    full, F_, kv = dict(ref.named_parameters()), ref.tok_emb.weight.shape[1], ref.blocks[0].attention.kv_dim
    mine = lambda w: w[t * len(w) // tp:(t + 1) * len(w) // tp]
    with torch.no_grad():
        for name, p in model.named_parameters():
            w = full[name]
            if "qkv" in name:
                w = torch.cat([mine(part) for part in w.split([F_, kv, kv])])
            elif "mlp.0" in name:                                      # column-parallel: my rows
                w = mine(w)
            elif name.endswith(("out.weight", "mlp.2.weight")):        # row-parallel: my columns
                w = mine(w.T).T
            p.copy_(w)


def activations(tokens: int, tp: int, cfg: dict) -> int:
    # bytes per GPU: 34 * s * b * h per layer with FlashAttention (Korthikanti et al.), divided by TP/SP
    return 34 * tokens * cfg["feat_dim"] * cfg["n_layer"] // tp


def states(model: nn.Module) -> dict:
    # bytes per GPU for mixed-precision AdamW: bf16 weights and grads, fp32 master, m and v
    n = sum(p.numel() for p in model.parameters())
    return {"bf16 weights": 2 * n, "bf16 grads": 2 * n, "fp32 master": 4 * n, "Adam m": 4 * n, "Adam v": 4 * n}


def _test(rank: int, world: int, dp: int, cp: int, tp: int):
    import torch.distributed as dist
    from collectives import all_reduce
    from fsdp import fully_shard
    from tp import sync_sp_grads
    init(rank, world, 29508)
    m = mesh(dp, cp, tp)
    cfg = dict(vocab_size=64, block_size=16, n_layer=2, feat_dim=32, num_head=4)
    torch.manual_seed(0)                                               # same data and weights everywhere
    ref = GPT(**cfg)
    idx, tgt = torch.randint(0, 64, (2 * dp, 16)), torch.randint(0, 64, (2 * dp, 16))
    ref(idx, tgt)[1].backward()

    model = GPT(**cfg, tp_group=m["tp_group"], cp_group=m["cp_group"])
    load_tp(model, ref, m["t"], tp)
    model = fully_shard(model, m["fsdp_group"])
    mine = lambda x: x.chunk(dp)[m["d"]].chunk(cp, dim=1)[m["c"]]      # DP: my micro-batch; CP: my chunk
    _, loss = model(mine(idx), mine(tgt))                              # the training step is unchanged
    loss.backward()
    sync_sp_grads(model, m["tp_group"])
    assert torch.allclose(all_reduce(loss.detach()) / world, ref(idx, tgt)[1], atol=1e-5)
    blk = model.blocks[0]
    g = ref.blocks[0].mlp[0].weight.grad
    for name, want in (("mlp.0.weight", g[m["t"] * len(g) // tp:(m["t"] + 1) * len(g) // tp]),
                       ("ln1.scale", ref.blocks[0].ln1.scale.grad)):  # TP-sharded, and SP-replicated
        n, i = dp * cp, m["d"] * cp + m["c"]                           # my FSDP shard, over DP x CP
        want = torch.cat([want, want.new_zeros((-len(want)) % n, *want.shape[1:])]).chunk(n)[i]
        assert torch.allclose(blk.shards[blk.names.index(name)].grad, want, atol=1e-5)
    if rank == 0:
        print(f"DP {dp} x CP {cp} x TP/SP {tp} GPT matches the single-GPU GPT (loss and gradients)")
    dist.destroy_process_group()


GPT2_SMALL = dict(vocab_size=50257, block_size=1024, n_layer=12, feat_dim=768, num_head=12)


def _report(rank: int, world: int, dp: int, cp: int, tp: int, seq: int):
    # GPT-2 small built on the meta device: real shapes and shards, no memory allocated
    import torch.distributed as dist
    from fsdp import fully_shard
    init(rank, world, 29510)
    m = mesh(dp, cp, tp)
    with torch.device("meta"):
        one = states(GPT(**GPT2_SMALL))
        mine = states(fully_shard(GPT(**GPT2_SMALL, tp_group=m["tp_group"], cp_group=m["cp_group"]), m["fsdp_group"]))
    if rank == 0:
        print(f"GPT-2 small, one {seq // 1024}k-token sequence per DP rank, GB per GPU: "
              f"1 GPU -> DP {dp} x CP {cp} x TP {tp} (FSDP over DP x CP)")
        print(f"  {'model states':12s} {sum(one.values()) / 1e9:6.3f} -> {sum(mine.values()) / 1e9:6.3f}")
        print(f"  {'activations':12s} {activations(seq, 1, GPT2_SMALL) / 1e9:6.2f} -> "
              f"{activations(seq // cp, tp, GPT2_SMALL) / 1e9:6.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    if not CUDA:
        _test_ckpt()                                                   # one device: CPU
    launch(_test, 8, 2, 2, 2)                                          # GPUs: torchrun --nproc-per-node 8 gpt.py device=cuda
    if not CUDA:                                                       # meta device: CPU
        for dp, cp in ((4, 1), (2, 2), (1, 4)):                        # 8 GPUs, TP 2 throughout
            torch.multiprocessing.spawn(_report, args=(8, dp, cp, 2, 131072), nprocs=8)
