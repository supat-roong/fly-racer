# %% [markdown]
# # Fly-Racer: Fruit Fly Connectome Drives a Race Car
#
# Can the wiring diagram of a real fruit fly brain learn to drive a car? This notebook trains a
# driving policy whose "brain" is a recurrent network wired exactly like part of a male fruit fly's
# nervous system, then lets it race Gymnasium `CarRacing-v3`.
#
# ![Fly racer demo](https://raw.githubusercontent.com/supat-roong/fly-racer/main/docs/demo.gif)
#
# Code and full write-up: https://github.com/supat-roong/fly-racer
#
# ## The brain
#
# A **connectome** is a complete map of which neurons connect to which, and how strongly. This project
# uses the **MaleCNS v1.0** connectome (Janelia, Google Research, Cambridge): more than 166,000 neurons
# across the brain and nerve cord, reconstructed from electron microscopy.
#
# Driving does not need the whole brain, so we extract a steering pathway (Tier S):
#
# | Group | Neurons | Cell types | Role |
# |---|---|---|---|
# | Input | 800 | Visual projection neurons (LC, LPLC2), ring neurons (ER) | Looming, objects, optic flow, visual landmarks |
# | Core | 2,430 | Central complex (EPG, PEN, PFL, PFN, hDeltaB) and the strongest bridging neurons | The fly's heading compass and steering circuit |
# | Output | 120 | Descending neurons (DNa, DNb, MDN, ...) | Commands for turning, walking forward and backing up |
#
# That is 3,350 neurons and 164,773 real connections.
#
# ## From pixels to steering
#
# ```
# game frame (96x96 gray, 4 stacked)
#   -> CNN "retina" (256 visual features)
#   -> input neurons -> central complex core (recurrent, connectome wiring) -> descending neurons
#   -> steer, gas, brake
# ```
#
# Each neuron is a leaky rate unit: `v <- v + (dt/tau)(-v + W r + b + I)`, `r = tanh(relu(v))`.
# Only real connectome edges exist, and each keeps its presynaptic neuron's sign from its
# neurotransmitter (Dale's law: GABA and glutamate inhibit, the rest excite). Training adjusts
# synapse strengths, time constants, biases and the visual input; it never adds connections.
# A separate small critic estimates the value of each state for PPO.
#
# ## Does the fly's wiring matter?
#
# The same pipeline trains control "brains" you can pick with `--core`:
#
# | Core | Wiring |
# |---|---|
# | `fly` | The real connectome subgraph |
# | `random` | Same neurons and degrees, connections shuffled |
# | `signs` | Real wiring, excitatory/inhibitory labels shuffled |
# | `mlp` | Plain neural network, no connectome |
#
# ## Run it yourself
#
# 1. The connectome comes from [neuPrint](https://neuprint.janelia.org), which needs a free login,
#    so it is not included here. Extract it on your machine (see "Get the connectome" in the README)
#    and upload `data/tier_s.npz` as a private Kaggle dataset.
# 2. Copy this notebook, then in the side panel: **Add Input** your connectome dataset,
#    set **Accelerator** to GPU T4 x2, and turn **Internet** on (phone-verified account needed).
# 3. Edit the settings below if you like, then **Save Version → Save & Run All**.
#
# The default run (1.5M decisions, 3M game frames) takes about 3.2 hours on Kaggle's GPU T4 x2.
# Results land in the notebook output under `runs/<name>/`, with the best policy in `best.pt`.
#
# ## Settings
#
# - `REF`: the commit this run uses, so the result is reproducible.
# - `TRAIN_ARGS`: switch brains with `--core random`, `--core signs` or `--core mlp`, or change the seed.
#   `python -m flyracer.train --help` lists every option.
# - `RESUME`: Kaggle stops a session after 12 hours. Training saves `resume.pt` before that; attach the
#   previous output as a dataset and set this to `True` to continue.

# %%
REPO = "https://github.com/supat-roong/fly-racer"
REF = "main"  # branch, tag or commit to train
RUN_NAME = "fly-1"
TRAIN_ARGS = "--core fly --seed 1"  # any flyracer.train arguments; cores: fly, random, signs, mlp
MAX_HOURS = 11.0  # save resume.pt and stop before Kaggle's 12 hour limit
RESUME = False  # continue runs/<RUN_NAME>/resume.pt from an attached dataset

# %% [markdown]
# ## 1. Get the code
#
# Clone the repository at `REF` and show the hardware. Kaggle's GPU T4 x2 machine has 4 CPU cores,
# which run the 8 game copies in parallel.

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
# Any attached dataset with a `tier_s.npz` works. It holds the neuron list (type, group, side,
# neurotransmitter sign) and the weighted connections. It is copied to `data/tier_s.npz` so the
# checkpoints refer to the same path as a local run and load on any machine.

# %%
found = glob.glob("/kaggle/input/**/tier_s.npz", recursive=True)
if not found:
    raise FileNotFoundError(
        "No tier_s.npz under /kaggle/input. Extract the connectome with flyracer.connectome.extract, "
        "upload it as a private dataset, and attach it with Add Input."
    )
os.makedirs(f"{CODE}/data", exist_ok=True)
shutil.copy(found[0], f"{CODE}/data/tier_s.npz")
CONNECTOME = "data/tier_s.npz"  # relative, as in a local run, so checkpoints load on any machine
print("connectome:", found[0])

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
# Most of a CarRacing step is drawing the frame, so on 4 CPUs the game, not the GPU, sets the pace.
# The repo draws the same frames faster: it skips off-screen track polygons in numpy, caches the font,
# and skips the in-between frames the agent never sees. This test checks that frames, rewards and
# episode ends still match stock `CarRacing-v3` exactly with the libraries installed here.

# %%
run(f"{PY} -m pytest -q tests/test_envs.py")

# %% [markdown]
# ## 5. Train
#
# PPO with these choices (all defaults in `flyracer/train.py`):
#
# | Choice | Setting | Why |
# |---|---|---|
# | Actions | Beta distribution per control in [0, 1] | Bounded like steer, gas and brake, so no probability is wasted on clipped values |
# | Augmentation | DrAC: frames shifted by up to 4 px must give the same policy and value | The biggest single gain on unseen tracks |
# | Control rate | A decision every 2 frames (25 per second) | Fine enough steering to hold speed through corners |
# | Horizon | Discount 0.9975 per decision, GAE 0.95 | Values finishing the lap, not just the next tile |
# | Update size | 10 epochs, clip 0.2, stop early at KL 0.02 | Keeps updates from undoing earlier learning |
# | Episodes | End after 100 decisions without a new tile | No time wasted on a stalled car |
#
# Each log line is one update of 1,024 decisions (8 games x 128 steps):
#
# - `return(last20)`: mean score of the last 20 training episodes, `max` the best so far
# - `eps`: training episodes finished, `sps`: decisions per second, `epochs`: epochs before the KL stop
#
# Every 25 updates the policy drives 16 tracks it never trains on (`eval step ...`), and the best
# policy so far is saved as `best.pt`. Checkpoints for a progress video are saved along the way.

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
# CarRacing gives +1000/N for each of the N track tiles visited and -0.1 per frame, so a perfect lap
# scores a bit over 900; 900 is the usual bar for solving it. The chart shows the mean score on 16
# held-out tracks at each evaluation, with one standard deviation shaded. Scores on 16 tracks are noisy,
# so judge a run by the trend, not one point.

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

# %% [markdown]
# ## What to expect, and what next
#
# Reference results, each core's `best.pt` scored on 100 full episodes on unseen tracks:
#
# | Core | Mean score | Worst episode | Episodes at 900+ |
# |---|---|---|---|
# | `fly` | **903.2 +/- 23.2** | **796.3** | 66 / 100 |
# | `signs` | 896.9 +/- 28.2 | 745.0 | 57 / 100 |
# | `mlp` | 894.0 +/- 75.9 | 351.2 | 63 / 100 |
# | `random` | 883.7 +/- 82.0 | 430.9 | 63 / 100 |
#
# ![Training curves of all four cores](https://raw.githubusercontent.com/supat-roong/fly-racer/main/docs/core_comparison.png)
#
# The real fly wiring learned fastest and drove most consistently: its worst track is far better than
# the shuffled or wiring-free brains. These are one seed each, so treat the ranking as a hint, not proof.
# Runs on a GPU are not bit-for-bit repeatable, so your scores will differ a little.
#
# To score and watch your run, download the output (`runs/<name>/`) next to a clone of the repo with
# your connectome in `data/`, then:
#
# ```bash
# python -m flyracer.evaluate runs/fly-1/best.pt --episodes 100 --seed 30000
# python -m flyracer.watch --ckpt runs/fly-1/best.pt --live
# python -m flyracer.watch --ckpt runs/fly-1/best.pt --record fly.mp4 --seed 26 --max-seconds 60
# ```
