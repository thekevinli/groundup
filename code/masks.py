import torch
from torch.nn import functional as F

# A mask is a rule on PAIRS of tokens: mask[b, i, j] = True if query i may attend to key j.
# Write the rule on token attributes (document, position), never on array indices, so it still
# holds when tokens arrive in any order. Shapes: ids (B, S) -> mask (B, S, S).


def causal_mask(S: int) -> torch.Tensor:
    # one sequence in order: query i sees keys 0..i
    return torch.ones(S, S, dtype=torch.bool).tril()


def causal_doc_mask(doc_ids, pos_ids):
    # packed documents, tokens in any order: same document, and key j is not later than query i
    q_doc, k_doc = doc_ids[:, :, None], doc_ids[:, None, :]            # (B, S, 1), (B, 1, S) -> (B, S, S)
    q_pos, k_pos = pos_ids[:, :, None], pos_ids[:, None, :]
    same_doc = q_doc == k_doc
    not_later = k_pos <= q_pos
    return same_doc & not_later


def block_causal_mask(doc_ids, step_ids):
    # robotics / video: a token of timestep t sees every token of steps <= t, its own step included
    q_doc, k_doc = doc_ids[:, :, None], doc_ids[:, None, :]
    q_step, k_step = step_ids[:, :, None], step_ids[:, None, :]
    same_doc = q_doc == k_doc
    not_later_step = k_step <= q_step
    return same_doc & not_later_step


def prefix_lm_mask(doc_ids, pos_ids, prefix_len: int):
    # bidirectional inside the first prefix_len tokens of each document (an instruction), causal after
    q_doc, k_doc = doc_ids[:, :, None], doc_ids[:, None, :]
    q_pos, k_pos = pos_ids[:, :, None], pos_ids[:, None, :]
    same_doc = q_doc == k_doc
    not_later = k_pos <= q_pos
    both_in_prefix = (q_pos < prefix_len) & (k_pos < prefix_len)
    return same_doc & (not_later | both_in_prefix)


def transfusion_mask(doc_ids, pos_ids, img_ids):
    # interleaved text and images (Transfusion, OmniGen): causal over the sequence, but the patches
    # of one image see each other. img_ids: which image a token belongs to, -1 for text
    q_doc, k_doc = doc_ids[:, :, None], doc_ids[:, None, :]
    q_pos, k_pos = pos_ids[:, :, None], pos_ids[:, None, :]
    q_img, k_img = img_ids[:, :, None], img_ids[:, None, :]
    same_doc = q_doc == k_doc
    not_later = k_pos <= q_pos
    same_image = (q_img == k_img) & (q_img >= 0)
    return same_doc & (not_later | same_image)


def sliding_window_mask(doc_ids, pos_ids, window: int):
    # causal, and only the last `window` tokens: memory per query is O(window), not O(S)
    q_doc, k_doc = doc_ids[:, :, None], doc_ids[:, None, :]
    q_pos, k_pos = pos_ids[:, :, None], pos_ids[:, None, :]
    same_doc = q_doc == k_doc
    dist = q_pos - k_pos                                               # how far back key j is
    return same_doc & (dist >= 0) & (dist < window)


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
    for m in (causal_doc_mask(doc, pos), prefix_lm_mask(doc, pos, 2),
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
    assert torch.equal(prefix_lm_mask(doc, pos, 2)[0].int(), torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]))
    assert torch.equal(sliding_window_mask(doc, pos, 2)[0].int(), torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))
    # interleaved "t i i t i i t": text causal, each image's two patches see each other, never the next image
    doc7, pos7 = torch.zeros(1, 7, dtype=torch.long), torch.arange(7)[None]
    img = torch.tensor([[-1, 0, 0, -1, 1, 1, -1]])
    assert torch.equal(transfusion_mask(doc7, pos7, img)[0].int(), torch.tensor([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]]))
    assert torch.equal(causal_doc_mask(doc7, pos7)[0], causal_mask(7))           # one document in order = tril

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
