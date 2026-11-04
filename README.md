# GPT interview guide

- `gpt-interview-guide.html`: the study guide. Open it in a browser; it works offline (MathJax is inlined; fonts fall back to system fonts offline). The "Files and imports" panel under the title shows this layout and the import graph.
- `code/`: the final version of every file the guide builds.
  - Main: `gpt.py` (model, 2D/3D parallel). It imports `collectives.py`, `tp.py`, `cp.py`; `fsdp.py` and `ep.py` are the other parallel helpers.
  - Side quests: `masks.py`, `backprop.py`, `budget.py`, `balance.py`.
  - Training: `train.py` (AdamW/Muon on Tiny Shakespeare).
  - Scaling, toy: `toy_isoflop.py`, an IsoFLOP sweep on CPU with the interview `gpt.py`.
  - Tokenizer: `bpe.py`, byte-level BPE from scratch, checked against GPT-2's real tokenizer.
  - Scaling, real: `gpt2_isoflop/` predicts GPT-2's loss. `ladder.py` does the work (its own GPT-2 copy, data, ladder, fits); `modal_app.py` runs it on a Modal H100 (≤ $10, ~1 h); `analyze.py` plots and checks the result on your laptop. Run all three from inside that folder.
  - Each file runs its own checks: `python <file>.py` (the parallel ones spawn CPU processes with gloo). Needs `torch`, `numpy`, `einops`; `regex`/`tiktoken` for bpe.py; `scipy`/`matplotlib` for analyze.py; `modal` for the ladder.
- `results/`: the training and toy IsoFLOP runs shown in the guide.
