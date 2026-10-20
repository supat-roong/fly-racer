# %% [markdown]
# # Fly-Racer: a fruit fly brain learns to drive
#
# This notebook trains a policy wired like a real fly's nervous system (MaleCNS v1.0 connectome)
# to drive Gymnasium `CarRacing-v3` with PPO. Code: https://github.com/supat-roong/fly-racer
#
# **To run it yourself**
#
# 1. The connectome comes from neuPrint, which needs a login, so it is not included here.
#    Extract it on your machine (README, "Get the connectome") and upload `data/tier_s.npz`
#    as a private Kaggle dataset.
# 2. Copy this notebook, then in the side panel: **Add Input** your connectome dataset,
#    set **Accelerator** to GPU T4 x2, and turn **Internet** on (phone-verified account needed).
# 3. Edit the settings below if you like, then **Save Version → Save & Run All**.
#
# The default run (1.5M decisions) takes about 3.2 hours on Kaggle's GPU T4 x2. Results land in the
# notebook output under `runs/<name>/`, with the best policy in `best.pt`.

# %%
REPO = "https://github.com/supat-roong/fly-racer"
REF = "main"  # branch, tag or commit to train
RUN_NAME = "fly-1"
TRAIN_ARGS = "--core fly --seed 1"  # any flyracer.train arguments; cores: fly, random, signs, mlp
MAX_HOURS = 11.0  # save resume.pt and stop before Kaggle's 12 hour limit
RESUME = False  # continue runs/<RUN_NAME>/resume.pt from an attached dataset

# %% [markdown]
# ## 1. Get the code

# %%
import glob
import os
import shutil
import subprocess
import sys

PY = sys.executable
CODE = "/kaggle/tmp/fly-racer"  # outside /kaggle/working, so only runs/ is saved as output
RUNS = "/kaggle/working/runs"


def run(cmd, cwd=CODE):
    """Run a shell command, stream its output, and stop the notebook if it fails."""
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})


shutil.rmtree(CODE, ignore_errors=True)
run(f"git clone -q {REPO} {CODE} && git -C {CODE} checkout -q {REF} && git -C {CODE} log -1 --oneline", cwd="/kaggle")
run("nvidia-smi --query-gpu=name --format=csv,noheader; nproc")

# %% [markdown]
# ## 2. Find your connectome
#
# Any attached dataset with a `tier_s.npz` works.

# %%
found = glob.glob("/kaggle/input/**/tier_s.npz", recursive=True)
if not found:
    raise FileNotFoundError(
        "No tier_s.npz under /kaggle/input. Extract the connectome with flyracer.connectome.extract, "
        "upload it as a private dataset, and attach it with Add Input."
    )
CONNECTOME = found[0]
print("connectome:", CONNECTOME)

# %% [markdown]
# ## 3. Install
#
# Kaggle's PyTorch already has CUDA. Recent builds dropped the P100 (Pascal), so on that GPU
# we install a cu126 build. `box2d-py` builds from source and needs `swig`.

# %%
cuda_ok = "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.ones(2, device='cuda').sum().item())"
if subprocess.run([PY, "-c", cuda_ok]).returncode != 0:
    run(f"{PY} -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126")
run(f"{PY} -m pip install -q swig pytest")
run(f"{PY} -m pip install -q 'gymnasium[box2d]>=1.0'")
run(f"{PY} -m pip install -q --no-deps -e .")

# %% [markdown]
# ## 4. Check the fast renderer
#
# Drawing frames is the bottleneck on Kaggle's 4 CPUs, so the repo draws them with a faster
# renderer. This test checks its frames, rewards and episode ends against stock `CarRacing-v3`
# with the libraries installed here.

# %%
run(f"{PY} -m pytest -q tests/test_envs.py")

# %% [markdown]
# ## 5. Train
#
# Progress prints every update. Every 25 updates the policy plays 16 held-out tracks and the
# best one is kept as `best.pt`.

# %%
os.makedirs(RUNS, exist_ok=True)
if not os.path.islink(f"{CODE}/runs"):
    os.symlink(RUNS, f"{CODE}/runs")

resume = ""
if RESUME:
    prev = glob.glob(f"/kaggle/input/**/runs/{RUN_NAME}/resume.pt", recursive=True)
    if not prev:
        raise FileNotFoundError(f"RESUME is on but no runs/{RUN_NAME}/resume.pt is attached")
    shutil.copytree(os.path.dirname(prev[0]), f"{RUNS}/{RUN_NAME}", dirs_exist_ok=True)
    resume = "--resume"

run(f"{PY} -m flyracer.train --name {RUN_NAME} --connectome {CONNECTOME} {TRAIN_ARGS} --max-hours {MAX_HOURS} {resume}")

# %% [markdown]
# ## 6. Results
#
# Mean score on 16 held-out tracks at each evaluation. 900 counts as solving CarRacing.

# %%
import matplotlib.pyplot as plt
import pandas as pd

evals = pd.read_csv(f"{RUNS}/{RUN_NAME}/eval.csv")
best = evals.loc[evals["score_mean"].idxmax()]
print(f"best: {best.score_mean:.1f} +/- {best.score_std:.1f} at step {int(best.step):,}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(evals["step"], evals["score_mean"])
ax.fill_between(evals["step"], evals["score_mean"] - evals["score_std"], evals["score_mean"] + evals["score_std"], alpha=0.2)
ax.axhline(900, color="gray", linestyle="--", linewidth=1)
ax.set(xlabel="agent decisions", ylabel="held-out score", title=RUN_NAME)
plt.show()
