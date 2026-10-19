# fly-racer PPO training on a Kaggle GPU; tools/kaggle/push.py fills in RUN.
RUN = {
    "name": "fly-1",
    "args": "--core fly --seed 1 --max-hours 11.0",
    "resume": False,  # continue runs/<name>/resume.pt from an attached earlier output
}
import glob
import os
import shutil
import subprocess
import sys
import time


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "Warn" not in line and "warn(" not in line:
            print(line, end="", flush=True)
    return p.wait()


py = sys.executable
work = "/kaggle/working"
repo = "/kaggle/tmp/repo"  # outside working/, so only runs/ is saved as output
# Kaggle unpacks the uploaded repo.tar.gz itself; find the project root by its pyproject.toml.
src = [p for p in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
       if os.path.isdir(os.path.dirname(p) + "/flyracer")]
shutil.copytree(os.path.dirname(src[0]), repo)
os.makedirs(f"{repo}/data", exist_ok=True)
shutil.copy(glob.glob("/kaggle/input/**/tier_s.npz", recursive=True)[0], f"{repo}/data/tier_s.npz")
sh("nvidia-smi --query-gpu=name --format=csv,noheader; nproc")

# Recent cu128 torch wheels dropped Pascal (P100, sm_60); fall back to a cu126 build there.
cuda_ok = "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.ones(2, device='cuda').sum().item())"
if subprocess.run([py, "-c", cuda_ok]).returncode != 0:
    sh(f"{py} -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126")
# box2d-py builds from source and needs swig.
sh(f"{py} -m pip install -q swig pytest && {py} -m pip install -q 'gymnasium[box2d]>=1.0' && "
   f"cd {repo} && {py} -m pip install -q --no-deps -e .")
# The fast renderer must give stock CarRacing's frames with the gymnasium and pygame installed here.
assert subprocess.run([py, "-m", "pytest", "-q", "tests/test_envs.py"], cwd=repo).returncode == 0, \
    "fast env does not match stock CarRacing here"

runs = f"{work}/runs"
os.makedirs(runs, exist_ok=True)
os.symlink(runs, f"{repo}/runs")
extra = ""
if RUN["resume"]:
    prev = glob.glob(f"/kaggle/input/**/runs/{RUN['name']}/resume.pt", recursive=True)
    assert prev, "no resume.pt to continue"
    shutil.copytree(os.path.dirname(prev[0]), f"{runs}/{RUN['name']}", dirs_exist_ok=True)
    extra = " --resume"
t0 = time.time()
rc = sh(f"cd {repo} && {py} -m flyracer.train --name {RUN['name']} --connectome data/tier_s.npz {RUN['args']}{extra}")
print(f"trainer exit {rc} after {time.time() - t0:.0f} s", flush=True)
