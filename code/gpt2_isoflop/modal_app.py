"""Run ladder.py on a Modal GPU for <= $10: tokenize FineWeb once, run the ladder, bring the results home.

  pip install modal && modal setup                 # once
  modal run modal_app.py --smoke                   # <= 10 min, <= $0.80 on an H100: a 3-rung mini ladder, then a
                                                   #   benchmark that prints the full ladder's H100-hours and $
  modal run --detach modal_app.py                  # the full ladder: ~1-1.5 h, stops by 1.5 h, killed at 1.75 h
  modal run --detach modal_app.py --rungs 5        # if the smoke run projects more than 1.5 h
  modal run --detach modal_app.py --gpu A10 --max-hours 6     # slower, same budget
  modal volume get isoflop-ladder results/full/summary.json . # if the terminal disconnected
  python analyze.py isoflop_gpu_full.json          # plots and the report, on your laptop
"""
import json
import pathlib
import shutil

import modal

HERE = pathlib.Path(__file__).parent
app = modal.App("isoflop-ladder")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "numpy<2.3", "scipy>=1.11", "huggingface_hub>=0.25", "pyarrow>=15", "tiktoken>=0.7")
         .env({"TIKTOKEN_CACHE_DIR": "/root/tiktoken"})
         .run_commands("python -c \"import tiktoken; tiktoken.get_encoding('gpt2')\"")   # bake GPT-2's BPE in
         .add_local_file(HERE / "ladder.py", "/root/ladder.py"))
vol = modal.Volume.from_name("isoflop-ladder", create_if_missing=True)

PRICE = {"H100": 3.95, "A10": 1.10}       # $/GPU-hour on Modal
BUDGET = 10.0                             # $ for everything
RESERVE = 1.5                             # $ for tokenizing (~$0.30 of CPU) and the smoke run
GRACE = 0.25                              # h: the ladder plans to stop at max_hours; Modal kills it this much later
REPO = "HuggingFaceFW/fineweb"            # the dataset llm.c's GPT-2 124M run used (sample-10BT)
SMOKE = dict(rungs=[1e14, 2e14, 4e14], widths=[24, 32, 48, 64, 96], calib=(32, 1e14, 2e14), check=(96, 4e14))


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=16384, timeout=3 * 3600)
def prepare(out: str, n_val: int, n_train: int):
    import sys
    sys.path.insert(0, "/root")
    import ladder
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    if pathlib.Path(f"{out}/meta.json").exists():                # tokenize once, reuse on every later run
        return
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("sample/10BT/") and f.endswith(".parquet"))

    def docs():                                                  # dataset order: the val prefix is llm.c's
        for name in files:
            path = hf_hub_download(REPO, name, repo_type="dataset", local_dir="/tmp/hf")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["text"]):
                yield from batch.column(0).to_pylist()

    ladder.write_tokens(docs(), out, n_val, n_train)
    vol.commit()


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, memory=32768)   # main() sets the GPU and timeout
def run(data: str, tag: str, smoke: bool, max_hours: float, n_rungs: int, gpu: str):
    import sys
    sys.path.insert(0, "/root")
    import ladder
    shutil.copytree(data, "/tmp/data")                           # random reads from local disk, not the volume
    if smoke:
        ladder.seeds = lambda C: 1                               # smoke tests the code path, not the noise
    kw = SMOKE if smoke else dict(rungs=ladder.RUNGS[:n_rungs])
    res = dict(summary=ladder.run_all("/tmp/data", f"/vol/results/{tag}", "cuda", max_hours, commit=vol.commit, **kw))
    if smoke:                                                    # same container: no second cold start
        thr = ladder.bench("cuda", widths=[64, 256, 768], steps=5)
        res["estimates"] = {n: ladder.estimate_hours(thr, rungs=ladder.RUNGS[:n]) for n in (6, 5)}
        for n, e in res["estimates"].items():
            core = e["total"] - e["seed repeats"] - e["lr probes"]
            print(f"{n} rungs: ~{e['total']:.2f} {gpu}-hours (${e['total'] * PRICE[gpu]:.2f}), core ~{core:.2f} h")
    return res


@app.local_entrypoint()
def main(smoke: bool = False, gpu: str = "H100", max_hours: float = 1.5, rungs: int = 6):
    tag = "smoke" if smoke else "full"
    max_hours, cap = (0.1, 0.2) if smoke else (max_hours, max_hours + GRACE)   # smoke: stops by 6 min, killed at 12
    worst = cap * PRICE[gpu] + (0 if smoke else RESERVE)
    assert worst <= BUDGET, f"worst case ${worst:.2f} > ${BUDGET:.0f}: lower --max-hours"
    print(f"[{tag}] {gpu}: stops by {max_hours} h, killed at {cap} h; worst case ${worst:.2f}")
    # val: the first 10.5M tokens of the dataset, llm.c's val shard (we evaluate on its first 2M).
    # train: 2.5B tokens, so no ladder run ever repeats data.
    data, n_val, n_train = ("/vol/fineweb_smoke", 2_200_000, 60_000_000) if smoke else ("/vol/fineweb", 10_485_760, 2_500_000_000)
    prepare.remote(data, n_val, n_train)
    res = run.with_options(gpu=gpu, timeout=int(cap * 3600)).remote(data, tag, smoke, max_hours, rungs, gpu)
    path = HERE / f"isoflop_gpu_{tag}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")
