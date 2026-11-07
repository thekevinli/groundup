# groundup

Models built from the ground up, one diff at a time. First: a GPT, from attention to 3D parallelism and scaling laws.

Open `gpt-interview-guide.html` in a browser (works offline). Its 40 interview-style steps build the files in `code/`, one change at a time.

![Where files live and who imports whom](files-and-imports.png)

## Run

- **Any file:** `cd code && python <file>.py` runs that file's checks. The parallel ones spawn CPU processes, so no GPU is needed.
- **GPT-2 ladder:** from `code/gpt2_isoflop/`:
  1. `modal run modal_app.py --smoke` (≤ 10 min)
  2. `modal run --detach modal_app.py` (H100, about 1 h, ≤ $10)
  3. `python analyze.py isoflop_gpu_full.json`
- **Needs:**
  - `torch`, `numpy` and `einops` for everything
  - `tiktoken` and `regex` for `bpe.py`
  - `scipy` and `matplotlib` for `analyze.py`
  - `modal` for the ladder

`results/` holds the runs plotted in the guide.
