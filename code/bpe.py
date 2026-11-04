"""Byte-level BPE from scratch: train, encode, decode -- and load GPT-2's real merges into it."""
from collections import Counter

import regex as re

# GPT-2's pre-tokenizer. Merges never cross these chunks, so " the" and "the." can't fuse into one token.
GPT2_SPLIT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge(ids: list, pair: tuple, new: int) -> list:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self, merges: dict, base: list = None, pattern: str = GPT2_SPLIT):
        self.base = base or list(range(256))          # token id of each single byte
        self.merges = merges                          # (id, id) -> new id; smaller id = learned earlier
        self.pattern = re.compile(pattern)
        self.vocab = {self.base[b]: bytes([b]) for b in range(256)}
        for (a, b), m in sorted(merges.items(), key=lambda kv: kv[1]):
            self.vocab[m] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, text: str, vocab_size: int, pattern: str = GPT2_SPLIT):
        # count each distinct chunk once, weighted by how often it occurs
        words = Counter(tuple(c.encode("utf-8")) for c in re.findall(pattern, text))
        merges = {}
        for new in range(256, vocab_size):
            pairs = Counter()
            for w, n in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += n
            if not pairs:
                break
            best = max(pairs, key=pairs.get)          # most frequent adjacent pair
            merges[best] = new
            words = Counter({tuple(merge(list(w), best, new)): n for w, n in words.items()})
        return cls(merges, pattern=pattern)

    def encode(self, text: str) -> list:
        out = []
        for chunk in self.pattern.findall(text):
            ids = [self.base[b] for b in chunk.encode("utf-8")]
            while len(ids) >= 2:
                # replay merges in the order they were learned: lowest id first, not most frequent now
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            out += ids
        return out

    def decode(self, ids: list) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    @classmethod
    def from_ranks(cls, ranks: dict, pattern: str = GPT2_SPLIT):
        """Recover the merge list from a tiktoken rank table {token bytes: id}. Each multi-byte token
        is the merge of the two pieces that the merges learned before it split it into."""
        base = [ranks[bytes([b])] for b in range(256)]
        tok = cls({}, base, pattern)
        for t, r in sorted(ranks.items(), key=lambda kv: kv[1]):
            if len(t) == 1:
                continue
            ids = [base[b] for b in t]
            while len(ids) > 2:
                pair = min(zip(ids, ids[1:]), key=lambda p: tok.merges.get(p, float("inf")))
                ids = merge(ids, pair, tok.merges[pair])
            tok.merges[tuple(ids)] = r
            tok.vocab[r] = t
        return tok


# ---------------- tests
if __name__ == "__main__":
    import base64
    import os
    import sys
    import time

    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else "the cat sat on the mat. " * 200
    t0 = time.time()
    tok = BPE.train(text[:300_000], 512)
    print(f"trained 256 merges in {time.time() - t0:.1f}s")

    # 1. first merge is the most frequent adjacent byte pair inside a chunk
    words = Counter(tuple(c.encode()) for c in re.findall(GPT2_SPLIT, text[:300_000]))
    pairs = Counter()
    for w, n in words.items():
        for p in zip(w, w[1:]):
            pairs[p] += n
    assert next(iter(tok.merges)) == max(pairs, key=pairs.get)

    # 2. lossless on anything, including text it never saw
    for s in [text[:2000], "héllo wörld 你好 🤖\n\n  tabs\tand  spaces", "", "a"]:
        assert tok.decode(tok.encode(s)) == s, s
    ids = tok.encode(text[300_000:310_000])
    print(f"compression on held-out text: {len(text[300_000:310_000].encode()) / len(ids):.2f} bytes/token")

    # 3. no token spans a chunk boundary: never a letter followed by a space then another letter
    assert not any(re.search(rb"\w \w", t) for t in tok.vocab.values())

    # 4. load GPT-2's real ranks; our encoder must match tiktoken token for token
    ranks, path = None, os.environ.get("GPT2_TIKTOKEN", "gpt2.tiktoken")   # e.g. whisper/assets/gpt2.tiktoken
    if os.path.exists(path):
        ranks = {base64.b64decode(t): int(r) for t, r in (line.split() for line in open(path) if line.strip())}
    else:
        try:
            import tiktoken
            ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks       # downloads once, then cached
        except Exception as e:
            print(f"skipping the GPT-2 check: {type(e).__name__}")
    if ranks:
        t0 = time.time()
        gpt2 = BPE.from_ranks(ranks)
        print(f"recovered {len(gpt2.merges):,} GPT-2 merges in {time.time() - t0:.1f}s")
        assert gpt2.encode("hello world") == [31373, 995]
        try:
            import tiktoken
            ref = tiktoken.Encoding("gpt2", pat_str=GPT2_SPLIT, mergeable_ranks=ranks, special_tokens={})
            sample = text[:20_000] + " héllo wörld 你好 🤖 1234567 don't  IT'S"
            assert gpt2.encode(sample) == ref.encode_ordinary(sample)
            print(f"matches tiktoken on {len(ref.encode_ordinary(sample)):,} tokens")
        except ImportError:
            pass
    print("all tests passed")
