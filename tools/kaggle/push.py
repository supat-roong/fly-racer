"""Train on a Kaggle GPU: upload the code and connectome, then start the training notebook.

    python tools/kaggle/push.py --user <kaggle-user> --name fly-1 --args "--core fly --seed 1"
    python tools/kaggle/push.py --user <kaggle-user> --name fly-1 --resume runs/fly-1   # next session

The committed code (git HEAD) and data/tier_s.npz go to the private dataset
<user>/fly-racer-src. With --resume, the run folder (resume.pt and logs) goes
to <user>/fly-racer-runs and training continues from it. Download the result
with `kaggle kernels output <user>/fly-racer-train -p runs/kaggle`.
Needs the kaggle CLI (pip install kaggle, or set KAGGLE_CLI="uvx kaggle") and a
token in ~/.kaggle.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = Path(__file__).with_name("train_kernel.py")


def kaggle(*args: str) -> str:
    cmd = shlex.split(os.environ.get("KAGGLE_CLI", "kaggle")) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(out.splitlines()[-1] if out else "", flush=True)
    return out


def upload_dataset(folder: Path, ref: str, title: str, message: str):
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": ref, "licenses": [{"name": "other"}]}))
    exists = "ready" in kaggle("datasets", "status", ref).lower()
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", message, "-q", "-r", "tar")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-q", "-r", "tar")
    time.sleep(10)
    for _ in range(60):  # a new version must finish processing before a notebook can mount it
        if "ready" in kaggle("datasets", "status", ref).lower():
            break
        time.sleep(10)
    time.sleep(20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True, help="Kaggle username")
    p.add_argument("--name", default="fly-1", help="run name (runs/<name>/)")
    p.add_argument("--args", default="--core fly --seed 1", help="extra flyracer.train arguments")
    p.add_argument("--max-hours", type=float, default=11.0, help="stop and save resume.pt before Kaggle's 12 h limit")
    p.add_argument("--resume", help="local run folder to continue (with resume.pt)")
    p.add_argument("--gpu", choices=["T4", "P100"], default="T4")
    a = p.parse_args()

    connectome = ROOT / "data" / "tier_s.npz"
    assert connectome.exists(), "extract the connectome first (flyracer.connectome.extract)"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        src.mkdir()
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        subprocess.run(["git", "archive", "--format=tar.gz", "--prefix=repo/", "HEAD", "-o", str(src / "repo.tar.gz")],
                       cwd=ROOT, check=True)
        shutil.copy(connectome, src)
        sources = [f"{a.user}/fly-racer-src"]
        upload_dataset(src, sources[0], "fly-racer-src", f"code {rev}")

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

        kdir = tmp / "kernel"
        kdir.mkdir()
        code = KERNEL.read_text()
        start, end = code.index("RUN = {"), code.index("}\n") + 2
        run = {"name": a.name, "args": f"{a.args} --max-hours {a.max_hours}", "resume": bool(a.resume)}
        code = code[:start] + "RUN = " + json.dumps(run, indent=4).replace("true", "True").replace("false", "False") \
            + "\n" + code[end:]
        (kdir / "fly-racer-train.py").write_text(code)
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{a.user}/fly-racer-train", "title": "fly-racer-train", "code_file": "fly-racer-train.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": sources,
            "machine_shape": {"T4": "NvidiaTeslaT4", "P100": "NvidiaTeslaP100"}[a.gpu],
        }))
        kaggle("kernels", "push", "-p", str(kdir))
    print(f"progress: https://www.kaggle.com/code/{a.user}/fly-racer-train")


if __name__ == "__main__":
    main()
