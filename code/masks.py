import torch
from torch.nn import functional as F

# A mask is a rule on PAIRS of tokens: mask[b, i, j] = True if query i may attend to key j.
# Write the rule on token attributes (document, position), never on array indices, so it still
# holds when tokens arrive in any order. Shapes: ids (B, S) -> mask (B, S, S).


def same_doc(doc_ids):
    return doc_ids[:, :, None] == doc_ids[:, None, :]                # query i's doc == key j's doc


def causal_doc_mask(doc_ids, pos_ids):
    # same document, and key j is not later than query i in that document
    return same_doc(doc_ids) & (pos_ids[:, None, :] <= pos_ids[:, :, None])


def block_causal_mask(doc_ids, step_ids):
    # robotics / video: every token of timestep t sees all tokens of steps <= t (ties see each other)
    return causal_doc_mask(doc_ids, step_ids)


def prefix_lm_mask(doc_ids, pos_ids, prefix_len):
    # bidirectional inside each document's prefix (e.g. an instruction or image), causal after it
    in_prefix = pos_ids < prefix_len                                    # (B, S); prefix_len per token
    both_prefix = in_prefix[:, :, None] & in_prefix[:, None, :]
    return same_doc(doc_ids) & ((pos_ids[:, None, :] <= pos_ids[:, :, None]) | both_prefix)


def sliding_window_mask(doc_ids, pos_ids, window):
    dist = pos_ids[:, :, None] - pos_ids[:, None, :]                  # how far back key j is
    return same_doc(doc_ids) & (dist >= 0) & (dist < window)


# ---------------- tests ----------------
def brute_force(doc_ids, pos_ids):
    B, S = doc_ids.shape
    m = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        for i in range(S):
            for j in range(S):
                m[b, i, j] = doc_ids[b, i] == doc_ids[b, j] and pos_ids[b, j] <= pos_ids[b, i]
    return m


def random_packed(B=2, S=12, shuffle=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    doc, pos = [], []
    for _ in range(B):
        lens = torch.randint(1, 6, (S,), generator=g).tolist()
        d = [k for k, n in enumerate(lens) for _ in range(n)][:S]        # doc id per token
        p = [t - d.index(k) for t, k in enumerate(d)]                    # position within doc
        order = torch.randperm(S, generator=g) if shuffle else torch.arange(S)
        doc.append(torch.tensor(d)[order]); pos.append(torch.tensor(p)[order])
    return torch.stack(doc), torch.stack(pos)


def test_hand_example():
    doc = torch.tensor([[0, 0, 1, 0]])                                  # doc 1 sits between doc 0's tokens
    pos = torch.tensor([[1, 0, 0, 2]])                                  # and doc 0 is out of order
    expected = torch.tensor([[1, 1, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [1, 1, 0, 1]], dtype=torch.bool)
    assert torch.equal(causal_doc_mask(doc, pos)[0], expected)


def test_matches_brute_force():
    for seed in range(5):
        doc, pos = random_packed(seed=seed)
        assert torch.equal(causal_doc_mask(doc, pos), brute_force(doc, pos))


def test_permutation_equivariant():
    # shuffling the tokens must shuffle the mask's rows and columns the same way
    doc, pos = random_packed(shuffle=False)
    perm = torch.randperm(doc.shape[1])
    assert torch.equal(causal_doc_mask(doc[:, perm], pos[:, perm]), causal_doc_mask(doc, pos)[:, perm][:, :, perm])


def test_no_empty_rows():
    doc, pos = random_packed()
    for m in (causal_doc_mask(doc, pos), prefix_lm_mask(doc, pos, torch.full_like(pos, 2)),
              sliding_window_mask(doc, pos, 3)):
        assert m.any(dim=-1).all()                                      # every query sees itself: no NaN


def test_attention_equals_per_document():
    # packed, shuffled attention == ordinary causal attention on each document alone
    doc, pos = random_packed(B=1, S=16)
    q, k, v = torch.randn(3, 1, 16, 8)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_doc_mask(doc, pos))
    for d in doc.unique():
        idx = (doc[0] == d).nonzero().flatten()
        idx = idx[pos[0, idx].argsort()]                                # this doc's tokens in order
        ref = F.scaled_dot_product_attention(q[:, idx], k[:, idx], v[:, idx], is_causal=True)
        assert torch.allclose(out[:, idx], ref, atol=1e-6)


def test_variants():
    doc = torch.tensor([[0, 0, 0, 0]]); pos = torch.tensor([[0, 1, 2, 3]])
    step = torch.tensor([[0, 0, 1, 1]])                                 # two tokens per timestep
    assert torch.equal(block_causal_mask(doc, step)[0].int(), torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]))
    assert torch.equal(prefix_lm_mask(doc, pos, torch.full_like(pos, 2))[0].int(), torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]))
    assert torch.equal(sliding_window_mask(doc, pos, 2)[0].int(), torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
