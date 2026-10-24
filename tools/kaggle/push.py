"""Train on a Kaggle GPU: publish the training notebook and start it.

    python tools/kaggle/push.py --user <kaggle-user> --name fly-1 --args "--core fly --seed 1"
    python tools/kaggle/push.py --user <kaggle-user> --name fly-1 --resume runs/fly-1   # next session

The notebook (train_notebook.py, converted to .ipynb) is public and clones the
code from GitHub at the current commit, so push your commits first. The
connectome is not public (neuPrint needs a login): data/tier_s.npz goes to your
private dataset <user>/fly-racer-connectome, uploaded once. With --resume, the
run folder (resume.pt and logs) goes to the private dataset <user>/fly-racer-runs.
Download the result with `kaggle kernels output <user>/fly-racer-fruit-fly-connectome-drives-a-race-car -p runs/kaggle`.
Needs the kaggle CLI (pip install kaggle, or set KAGGLE_CLI="uvx kaggle") and a
token in ~/.kaggle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = Path(__file__).with_name("train_notebook.py")
TITLE = "Fly-Racer: Fruit Fly Connectome Drives a Race Car"
SLUG = "fly-racer-fruit-fly-connectome-drives-a-race-car"  # Kaggle derives the URL from the title
KEYWORDS = ["neuroscience", "biology", "reinforcement learning", "pytorch", "simulations"]  # Kaggle allows at most 5 tags


def kaggle(*args: str) -> str:
    cmd = shlex.split(os.environ.get("KAGGLE_CLI", "kaggle")) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(out.splitlines()[-1] if out else "", flush=True)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def dataset_ready(ref: str) -> bool:
    return "ready" in kaggle("datasets", "status", ref).lower()


def upload_dataset(folder: Path, ref: str, title: str, message: str):
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": ref, "licenses": [{"name": "other"}]}))
    if dataset_ready(ref):
        kaggle("datasets", "version", "-p", str(folder), "-m", message, "-q")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-q")
    time.sleep(10)
    for _ in range(60):  # a new version must finish processing before a notebook can mount it
        if dataset_ready(ref):
            break
        time.sleep(10)
    time.sleep(20)


def to_ipynb(source: str) -> dict:
    """Convert a "# %%" percent-format script to a notebook; "# %% [markdown]" cells lose their "# "."""
    cells = []
    for chunk in re.split(r"^# %%", source, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        markdown = "[markdown]" in header
        lines = body.strip("\n").splitlines()
        if markdown:
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        text = [ln + "\n" for ln in lines]
        if text:
            text[-1] = text[-1].rstrip("\n")
        cell = {"cell_type": "markdown" if markdown else "code", "metadata": {}, "source": text}
        if not markdown:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}}}


def set_setting(source: str, name: str, value) -> str:
    text = json.dumps(value) if isinstance(value, str) else repr(value)
    return re.sub(rf"^{name} = .*?(  #.*)?$", lambda m: f"{name} = {text}{m.group(1) or ''}", source, count=1, flags=re.M)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True, help="Kaggle username")
    p.add_argument("--name", default="fly-1", help="run name (runs/<name>/)")
    p.add_argument("--args", default="--core fly --seed 1", help="extra flyracer.train arguments")
    p.add_argument("--max-hours", type=float, default=11.0, help="stop and save resume.pt before Kaggle's 12 h limit")
    p.add_argument("--resume", help="local run folder to continue (with resume.pt)")
    p.add_argument("--gpu", choices=["T4", "P100"], default="T4")
    p.add_argument("--private", action="store_true", help="keep the notebook private (default: public)")
    a = p.parse_args()

    rev = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", rev):
        sys.exit(f"commit {rev[:7]} is not on any remote branch; the notebook clones from GitHub, so git push first")
    repo = re.sub(r"^git@github\.com:", "https://github.com/", git("remote", "get-url", "origin")).removesuffix(".git")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sources = [f"{a.user}/fly-racer-connectome"]
        if not dataset_ready(sources[0]):
            connectome = ROOT / "data" / "tier_s.npz"
            assert connectome.exists(), "extract the connectome first (flyracer.connectome.extract)"
            (tmp / "connectome").mkdir()
            shutil.copy(connectome, tmp / "connectome")
            upload_dataset(tmp / "connectome", sources[0], "fly-racer-connectome", "tier_s")

        if a.resume:
            run_dir = Path(a.resume)
            assert (run_dir / "resume.pt").exists(), f"no {run_dir}/resume.pt"
            dst = tmp / "runs" / "runs" / a.name
            dst.mkdir(parents=True)
            for f in run_dir.iterdir():  # resume state, logs and best.pt; not every checkpoint
                if f.suffix in (".csv", ".json") or f.name in ("resume.pt", "best.pt", "latest.pt"):
                    shutil.copy(f, dst)
            sources.append(f"{a.user}/fly-racer-runs")
            upload_dataset(tmp / "runs", sources[1], "fly-racer-runs", f"resume {a.name}")

        code = NOTEBOOK.read_text()
        for name, value in [("REPO", repo), ("REF", rev), ("RUN_NAME", a.name), ("TRAIN_ARGS", a.args),
                            ("MAX_HOURS", a.max_hours), ("RESUME", bool(a.resume))]:
            code = set_setting(code, name, value)
        kdir = tmp / "kernel"
        kdir.mkdir()
        (kdir / f"{SLUG}.ipynb").write_text(json.dumps(to_ipynb(code), indent=1))
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{a.user}/{SLUG}", "title": TITLE, "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": a.private, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": sources, "keywords": KEYWORDS,
            "machine_shape": {"T4": "NvidiaTeslaT4", "P100": "NvidiaTeslaP100"}[a.gpu],
        }))
        kaggle("kernels", "push", "-p", str(kdir))
    print(f"progress: https://www.kaggle.com/code/{a.user}/{SLUG}")


if __name__ == "__main__":
    main()
