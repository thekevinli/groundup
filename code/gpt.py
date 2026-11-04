import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from collectives import ranks
from cp import all_gather_kv
from tp import ColumnParallelLinear, GatherFromSeq, RowParallelLinear, ScatterToSeq


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
    pos = torch.arange(seq_len).float()[:, None]                      # (S, 1)
    w = base ** (-torch.arange(0, dim, 2).float() / dim)              # (dim/2,)
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos * w)
    pe[:, 1::2] = torch.cos(pos * w)
    return pe                                                         # (S, dim): added to token embeddings


def rope_freqs(head_dim: int, base: float = 10000.0, scale: float = 1.0,
               method: str = "none", train_len: int = 2048) -> torch.Tensor:
    # theta_i = base^(-2i/hd): dim pair 0 spins fastest, the last pair slowest
    freqs = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    if method == "pi":      # position interpolation: slow down every frequency by `scale`
        freqs = freqs / scale
    elif method == "ntk":   # NTK-aware: raise the base; high freqs barely move, lowest slows by `scale`
        base = base * scale ** (head_dim / (head_dim - 2))
        freqs = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    elif method == "yarn":  # YaRN: slow down only dims that turn < 1 time in train_len, ramp up to 32
        turns = train_len * freqs / (2 * math.pi)
        keep = ((turns - 1) / (32 - 1)).clamp(0, 1)      # 1 = high freq, keep; 0 = low freq, interpolate
        freqs = keep * freqs + (1 - keep) * freqs / scale
    return freqs


def rope_cos_sin(pos: torch.Tensor, freqs: torch.Tensor, mscale: float = 1.0):
    angles = pos[..., None].float() * freqs                           # (S, hd/2), or (B, S, hd/2) packed
    if angles.dim() == 3:
        angles = angles[:, None]                                      # (B, 1, S, hd/2): shared by heads
    return torch.cos(angles) * mscale, torch.sin(angles) * mscale    # YaRN: mscale sharpens attention


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # rotate each pair (x[2i], x[2i+1]) by angle pos * theta_i; x: (B, H, S, hd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(x.dtype)                                            # cos/sin are fp32


class MHSA(nn.Module):

    def __init__(self, feat_dim: int, num_head: int, num_kv_head: int = None, tp_group=None, cp_group=None):
        super().__init__()
        assert feat_dim % num_head == 0
        self.cp_group = cp_group
        tp = ranks(tp_group)[1] if tp_group else 1
        num_kv_head = num_kv_head or num_head                             # GQA: H_kv < H, MQA: H_kv = 1
        assert num_head % num_kv_head == 0 and num_kv_head % tp == 0
        self.num_head, self.num_kv_head = num_head // tp, num_kv_head // tp   # TP: my heads only
        self.head_dim = feat_dim // num_head
        self.q_dim, self.kv_dim = self.num_head * self.head_dim, self.num_kv_head * self.head_dim
        # q: F, k and v: H_kv * hd each. TP: columns = my heads' q, k, v; rows of out = my heads
        self.qkv = ColumnParallelLinear(feat_dim, feat_dim + 2 * num_kv_head * self.head_dim, tp_group, True)
        self.out = RowParallelLinear(feat_dim, feat_dim, tp_group, True)

    def forward(self, x: torch.Tensor, is_causal: bool = False, rope=None, attn_mask=None,
                cache=None) -> torch.Tensor:
        # x: (B, S, F), my slice of S under SP; rope: (cos, sin) or None; attn_mask: bool, True = may attend
        # cache: this layer's {"k", "v"} while decoding, or None
        q, k, v = self.qkv(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = rearrange(q, "b s (h f) -> b h s f", h=self.num_head)
        k = rearrange(k, "b s (h f) -> b h s f", h=self.num_kv_head)
        v = rearrange(v, "b s (h f) -> b h s f", h=self.num_kv_head)
        if rope is not None:                                              # rotate q and k, never v
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
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
                 num_expert: int = 0, dropout: float = 0.0, tp_group=None, cp_group=None):
        super().__init__()
        assert not (num_expert and tp_group), "MoE layers use expert parallelism, not TP"
        self.use_rope = use_rope                     # False = a NoPE layer: no position signal here
        self.drop = nn.Dropout(dropout)                                   # on each branch's output
        self.ln1 = LayerNorm(feat_dim)
        self.attention = MHSA(feat_dim, num_head, num_kv_head, tp_group, cp_group)
        self.ln2 = LayerNorm(feat_dim)
        self.mlp = MoE(feat_dim, num_expert) if num_expert else nn.Sequential(
            ColumnParallelLinear(feat_dim, feat_dim * 4, tp_group, True),
            nn.GELU(),
            RowParallelLinear(feat_dim * 4, feat_dim, tp_group, True),
        )
        for p in [*self.ln1.parameters(), *self.ln2.parameters()]:      # SP: norms see my slice of S
            p.sequence_parallel = tp_group is not None

    def forward(self, x: torch.Tensor, is_causal: bool = False, rope=None, attn_mask=None,
                cache=None) -> torch.Tensor:
        # pre-norm: the residual stream x is never normalized, so gradients have an identity path
        if not self.use_rope:
            rope = None                                                   # NoPE layer
        x = x + self.drop(self.attention(self.ln1(x), is_causal, rope, attn_mask, cache))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_layer: int, feat_dim: int, num_head: int,
                 pos_emb: str = "rope", rope_scale: float = 1.0, rope_method: str = "none",
                 dropout: float = 0.0, nope_every: int = 0, num_kv_head: int = None,
                 num_expert: int = 0, tp_group=None, cp_group=None, grad_ckpt: bool = False):
        super().__init__()
        self.grad_ckpt = grad_ckpt                   # recompute each block in backward to save memory
        self.tp_group = tp_group                     # TP + SP inside this group; None = one GPU
        self.cp_group = cp_group                     # CP: each rank holds one chunk of the sequence
        self.pos_emb = pos_emb                       # "none" = NoPE everywhere (Kimi Linear's MLA layers)
        self.max_len = int(block_size * rope_scale)                    # context allowed at inference
        self.tok_emb = nn.Embedding(vocab_size, feat_dim)
        nope = lambda i: nope_every and (i + 1) % nope_every == 0      # 4: every 4th layer (Llama 4, SmolLM3)
        self.blocks = nn.ModuleList([TransformerBlock(feat_dim, num_head, num_kv_head, not nope(i), num_expert,
                                                      dropout, tp_group, cp_group) for i in range(n_layer)])
        self.ln_f = LayerNorm(feat_dim)
        self.lm_head = nn.Linear(feat_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                       # weight tying
        if pos_emb == "sin":
            self.register_buffer("pe", sinusoidal_pe(self.max_len, feat_dim), persistent=False)
        if pos_emb == "rope":
            freqs = rope_freqs(feat_dim // num_head, scale=rope_scale, method=rope_method,
                               train_len=block_size)
            self.register_buffer("freqs", freqs, persistent=False)
            self.mscale = 0.1 * math.log(rope_scale) + 1 if rope_method == "yarn" else 1.0
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, doc_ids=None, pos_ids=None, cache=None):
        S = idx.shape[1]                                               # my CP chunk (all of it without CP)
        c, cp = ranks(self.cp_group)[:2] if self.cp_group else (0, 1)
        past = cache[0]["k"].shape[2] if cache and cache[0] else 0     # tokens already in the cache
        x = self.tok_emb(idx)                                          # (B, S, F)
        q_pos = past + c * S + torch.arange(S, device=idx.device)      # global positions of my tokens
        pos = pos_ids if pos_ids is not None else q_pos
        mask = doc_mask(doc_ids) if doc_ids is not None else None      # packed rows; else plain causal
        if cp > 1:                                                     # CP: my queries vs every rank's keys
            assert doc_ids is None and cache is None, "with CP, compare the gathered keys' doc ids too"
            mask = torch.arange(S * cp, device=idx.device)[None, :] <= q_pos[:, None]   # causal by position
        rope = None
        if self.pos_emb == "sin":
            x = x + self.pe[pos].to(x.dtype)
        if self.pos_emb == "rope":
            rope = rope_cos_sin(pos, self.freqs, self.mscale)
        if self.tp_group:
            x = ScatterToSeq.apply(x, self.tp_group)                   # SP: my slice of the sequence
        for i, block in enumerate(self.blocks):
            if self.grad_ckpt and self.training:     # keep only the block input; rerun the block in backward
                x = checkpoint(block, x, True, rope, mask, use_reentrant=False)
            else:
                x = block(x, is_causal=True, rope=rope, attn_mask=mask, cache=cache[i] if cache else None)
        if self.tp_group:
            x = GatherFromSeq.apply(x, self.tp_group)                  # full sequence for the head
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


def doc_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    # packed rows: causal AND same document, (B, 1, S, S), shared by heads
    S = doc_ids.shape[1]
    causal = torch.ones(S, S, dtype=torch.bool, device=doc_ids.device).tril()
    return (causal & (doc_ids[:, :, None] == doc_ids[:, None, :]))[:, None]


def pack(docs: list[list[int]], seq_len: int, pad_id: int = 0):
    # greedily put whole documents into rows of seq_len tokens (truncate long ones, pad the rest)
    rows = [[]]
    for d, doc in enumerate(docs):
        doc = doc[:seq_len]
        if len(rows[-1]) + len(doc) > seq_len:
            rows.append([])
        rows[-1] += [(tok, d, p) for p, tok in enumerate(doc)]      # (token, doc id, position in doc)
    tokens = torch.full((len(rows), seq_len), pad_id)
    doc_ids = torch.full((len(rows), seq_len), -1)                    # -1 = padding
    pos_ids = torch.zeros((len(rows), seq_len), dtype=torch.long)
    for b, row in enumerate(rows):
        for s, (tok, d, p) in enumerate(row):
            tokens[b, s], doc_ids[b, s], pos_ids[b, s] = tok, d, p
    # next-token targets, but never predict across a document boundary or into padding
    targets = torch.full_like(tokens, -100)
    same_doc = (doc_ids[:, 1:] == doc_ids[:, :-1]) & (doc_ids[:, 1:] >= 0)
    targets[:, :-1][same_doc] = tokens[:, 1:][same_doc]
    return tokens, targets, doc_ids, pos_ids


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
    # bytes per GPU: 34 * s * b * h per layer with FlashAttention (budget.py), divided by TP/SP
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
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29508", rank=rank, world_size=world)
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
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29510", rank=rank, world_size=world)
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
    torch.multiprocessing.spawn(_test, args=(8, 2, 2, 2), nprocs=8)
    for dp, cp in ((4, 1), (2, 2), (1, 4)):                            # 8 GPUs, TP 2 throughout
        torch.multiprocessing.spawn(_report, args=(8, dp, cp, 2, 131072), nprocs=8)
