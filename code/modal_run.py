r"""Run any file in code/ on N T4 GPUs on Modal: torchrun --nproc-per-node N <file> <args>, NCCL instead of gloo.

  pip install modal && python3 -m modal setup                          # once
  modal run modal_run.py --gpus 4 --cmd "collectives.py device=cuda"    # a file's own checks on 4 x T4
  modal run modal_run.py --gpus 8 --ref "train.py parity=1 device=cuda out=1gpu.json" \
      --cmd "train.py parity=1 dp=2 cp=2 tp=2 device=cuda ref=1gpu.json"   # parity: the 1-GPU run first, same container

T4 on Modal: $0.59 per GPU-hour ($0.000164/s), up to 8 per container. Written for the guide, not yet run.
"""
import pathlib
import shlex
import subprocess

import modal

HERE = pathlib.Path(__file__).parent
REMOTE = "/root/code"
app = modal.App("groundup-t4")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "numpy<2.3", "einops")
         .add_local_dir(HERE, REMOTE, ignore=["gpt2_isoflop", "__pycache__", "*.json"]))   # input.txt too, if here


@app.function(image=image, gpu="T4", timeout=1800)            # main() sets the GPU count
def run(cmd: str, gpus: int, ref: str):
    if ref:                                                    # one plain process: train.py's 1-GPU reference
        print(f"$ python {ref}", flush=True)
        subprocess.run(["python", *shlex.split(ref)], cwd=REMOTE, check=True)
    print(f"$ torchrun --nproc-per-node {gpus} {cmd}", flush=True)
    subprocess.run(["torchrun", f"--nproc-per-node={gpus}", *shlex.split(cmd)], cwd=REMOTE, check=True)


@app.local_entrypoint()
def main(cmd: str, gpus: int = 1, ref: str = ""):
    assert 1 <= gpus <= 8, "a Modal container holds up to 8 T4s"
    run.with_options(gpu=f"T4:{gpus}").remote(cmd, gpus, ref)    # its output streams back here
