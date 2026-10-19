"""PPO training on CarRacing-v3 (single file, CleanRL style).

    python -m flyracer.train --core fly --connectome data/tier_s.npz
    python -m flyracer.train --core random --connectome data/tier_s.npz   # control
    python -m flyracer.train --core mlp                                    # baseline

With no --connectome and a connectome core, a synthetic Tier S graph is used.

Recipe: a decision every 2 frames, Beta actions, DrAC random-shift
regularization, KL early stop, a long discount, a learning rate that decays to a
floor, and periodic evaluation on held-out tracks that keeps the best checkpoint
as best.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

from .augment import random_shift
from .checkpoint import load_connectome, pick_device, save_checkpoint
from .envs import episode_returns, make_vec_env
from .evaluate import evaluate
from .models import CORES, Agent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", choices=CORES, default="fly")
    p.add_argument("--connectome", help=".npz from flyracer.connectome.extract (default: synthetic)")
    p.add_argument("--name", help="run name (default: <core>-<seed>)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--total-steps", type=int, default=1_500_000, help="agent decisions (x action-repeat frames)")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--action-repeat", type=int, default=2,
                   help="frames per decision; 2 (25 decisions/s) steers finely enough to hold speed in corners")
    p.add_argument("--patience", type=int, default=100,
                   help="end episode after N decisions (200 frames at repeat 2) without progress (0=off)")
    p.add_argument("--rnn-steps", type=int, default=6, help="FlyRNN Euler steps per frame")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-floor", type=float, default=0.1, help="LR decays linearly to this fraction of --lr")
    p.add_argument("--gamma", type=float, default=0.9975, help="per decision; ~0.995 per 4 frames")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--target-kl", type=float, default=0.02, help="stop an update's epochs past this KL (0=off)")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--drac-coef", type=float, default=0.1,
                   help="DrAC: weight of policy/value consistency under random frame shifts (0=off)")
    p.add_argument("--aug-pad", type=int, default=4, help="max random shift in pixels for DrAC")
    p.add_argument("--save-every", type=int, default=20, help="updates between checkpoints")
    p.add_argument("--eval-every", type=int, default=25, help="updates between evaluations for best.pt (0=off)")
    p.add_argument("--eval-episodes", type=int, default=16, help="full episodes per evaluation")
    p.add_argument("--resume", action="store_true",
                   help="continue runs/<name>/resume.pt (agent, optimizer, counters, best score)")
    p.add_argument("--max-hours", type=float, default=0,
                   help="save resume.pt and stop after this many hours (0=off), e.g. under a session limit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    name = args.name or f"{args.core}-{args.seed}"
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    state = None
    if args.resume and (run_dir / "resume.pt").exists():
        state = torch.load(run_dir / "resume.pt", map_location="cpu", weights_only=False)
    else:
        (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    conn = load_connectome(args.connectome, args.core)
    if conn is not None:
        print(conn.summary())

    envs = make_vec_env(args.num_envs, args.seed, args.action_repeat, patience=args.patience or None)
    agent = Agent(args.core, conn, steps=args.rnn_steps, seed=args.seed).to(device)
    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"core={args.core} device={device} trainable params={n_params:,}")
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    first_update, step, recent, best_score = 1, 0, [], -float("inf")
    if state is not None:
        agent.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["opt"])
        first_update, step, recent, best_score = state["update"] + 1, state["step"], state["recent"], state["best"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
        print(f"resumed {run_dir} at update {first_update} step {step}")

    T, N = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, N, 4, 96, 96), dtype=torch.uint8, device=device)
    act_buf = torch.zeros((T, N, 3), device=device)
    logp_buf, rew_buf, done_buf, val_buf = (torch.zeros((T, N), device=device) for _ in range(4))
    # Value of the true final observation for truncated episodes (time limit, not failure).
    boot_buf = torch.zeros((T, N), device=device)

    batch = T * N
    mb_size = batch // args.minibatches
    n_updates = args.total_steps // batch
    mode = "a" if state is not None else "w"
    log_f = open(run_dir / "progress.csv", mode, newline="")
    log = csv.writer(log_f)
    ep_f = open(run_dir / "episodes.csv", mode, newline="")
    ep_log = csv.writer(ep_f)
    # Periodic evaluation on held-out tracks keeps the best checkpoint even if PPO later drifts.
    eval_f = open(run_dir / "eval.csv", mode, newline="")
    eval_log = csv.writer(eval_f)
    if state is None:
        log.writerow(["update", "step", "frames", "return_mean", "return_max", "episodes", "sps",
                      "pg_loss", "v_loss", "entropy", "approx_kl"])
        ep_log.writerow(["episode", "step", "score"])
        eval_log.writerow(["update", "step", "score_mean", "score_std"])

    # A resumed run starts fresh episodes (env state is not saved), on new tracks.
    obs, _ = envs.reset(seed=args.seed + step)
    obs = torch.as_tensor(obs, device=device)
    start, start_step = time.time(), step
    for update in range(first_update, n_updates + 1):
        for g in opt.param_groups:
            frac = 1 - (update - 1) / n_updates
            g["lr"] = args.lr * (args.lr_floor + (1 - args.lr_floor) * frac)

        agent.eval()
        for t in range(T):
            obs_buf[t] = obs
            with torch.no_grad():
                action, logp, _, value, _ = agent(obs)
            act_buf[t], logp_buf[t], val_buf[t] = action, logp, value
            nxt, reward, term, trunc, infos = envs.step(agent.env_action(action).cpu().numpy())
            step += N
            rew_buf[t] = torch.as_tensor(reward, device=device, dtype=torch.float32)
            done_buf[t] = torch.as_tensor(term | trunc, device=device, dtype=torch.float32)
            boot_buf[t] = 0
            if trunc.any() and "final_obs" in infos:
                idx = np.flatnonzero(trunc & ~term)
                if len(idx):
                    final = torch.as_tensor(np.stack(infos["final_obs"][idx]), device=device)
                    with torch.no_grad():
                        boot_buf[t, idx] = agent.value(final)
            for score in episode_returns(infos.get("final_info", infos)):
                recent.append(score)
                ep_log.writerow([len(recent), step, f"{score:.1f}"])
            obs = torch.as_tensor(nxt, device=device)

        with torch.no_grad():
            next_value = agent.value(obs)
            adv = torch.zeros_like(rew_buf)
            gae = 0
            for t in reversed(range(T)):
                nv = next_value if t == T - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                target = rew_buf[t] + args.gamma * (nv * nonterm + boot_buf[t])
                delta = target - val_buf[t]
                gae = delta + args.gamma * args.gae_lambda * nonterm * gae
                adv[t] = gae
            ret = adv + val_buf

        b_obs, b_act = obs_buf.reshape(batch, 4, 96, 96), act_buf.reshape(batch, 3)
        b_logp, b_adv, b_ret, b_val = (x.reshape(-1) for x in (logp_buf, adv, ret, val_buf))
        agent.train()
        epochs_run = 0
        for _ in range(args.epochs):
            epochs_run += 1
            for mb in torch.randperm(batch, device=device).split(mb_size):
                m = len(mb)
                obs_mb = b_obs[mb]
                if args.drac_coef:
                    # Real and shifted frames in one pass (DrAC terms below).
                    obs_mb = torch.cat([obs_mb.float(), random_shift(obs_mb, args.aug_pad)])
                alpha, beta, value_all, _ = agent.dist_value(obs_mb)
                dist = Beta(alpha[:m], beta[:m])
                value = value_all[:m]
                logp = dist.log_prob(b_act[mb].clamp(1e-6, 1 - 1e-6)).sum(-1)
                ent = dist.entropy().sum(-1)
                ratio = (logp - b_logp[mb]).exp()
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_clipped = b_val[mb] + (value - b_val[mb]).clamp(-args.clip, args.clip)
                v_loss = 0.5 * torch.max((value - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * ent.mean() + args.vf_coef * v_loss
                if args.drac_coef:
                    # DrAC (Raileanu et al. 2021): the PPO loss stays on real frames; these terms
                    # make the policy and value agree on randomly shifted copies of them. The
                    # target actions are sampled from the (detached) policy on the real frames.
                    with torch.no_grad():
                        target_act = Beta(alpha[:m].detach(), beta[:m].detach()).sample()
                    logp_aug = Beta(alpha[m:], beta[m:]).log_prob(target_act.clamp(1e-6, 1 - 1e-6)).sum(-1)
                    drac = -logp_aug.mean() + ((value_all[m:] - value.detach()) ** 2).mean()
                    loss = loss + args.drac_coef * drac
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - (logp - b_logp[mb])).mean().item()
            if args.target_kl and approx_kl > args.target_kl:
                break  # policy moved far enough from the one that collected this data

        sps = (step - start_step) / (time.time() - start)
        r_mean = float(np.mean(recent[-20:])) if recent else float("nan")
        r_max = float(np.max(recent)) if recent else float("nan")
        log.writerow([update, step, step * args.action_repeat, f"{r_mean:.1f}", f"{r_max:.1f}", len(recent),
                      f"{sps:.0f}", f"{pg_loss.item():.4f}", f"{v_loss.item():.4f}",
                      f"{ent.mean().item():.3f}", f"{approx_kl:.4f}"])
        log_f.flush()
        ep_f.flush()
        print(f"update {update}/{n_updates} step {step} return(last20) {r_mean:7.1f} "
              f"max {r_max:7.1f} eps {len(recent)} sps {sps:.0f} epochs {epochs_run}")
        if update % args.save_every == 0 or update == n_updates:
            save_checkpoint(run_dir / f"ckpt_{step:08d}.pt", agent, args, step, args.connectome)
            save_checkpoint(run_dir / "latest.pt", agent, args, step, args.connectome)
        if args.eval_every and (update % args.eval_every == 0 or update == n_updates):
            # Keep every evaluated model on disk, not only the current best, so a
            # later re-evaluation on more tracks can pick among them.
            ckpt = run_dir / f"ckpt_{step:08d}.pt"
            if not ckpt.exists():
                save_checkpoint(ckpt, agent, args, step, args.connectome)
            agent.eval()
            scores = evaluate(agent, args.eval_episodes, action_repeat=args.action_repeat)
            score = float(np.mean(scores))
            eval_log.writerow([update, step, f"{score:.1f}", f"{np.std(scores):.1f}"])
            eval_f.flush()
            tag = ""
            if score > best_score:
                best_score, tag = score, "  new best -> best.pt"
                save_checkpoint(run_dir / "best.pt", agent, args, step, args.connectome)
            print(f"eval step {step}: {score:.1f} +/- {np.std(scores):.1f}{tag}")
        if args.max_hours and time.time() - start > args.max_hours * 3600 and update < n_updates:
            torch.save({"state_dict": agent.state_dict(), "opt": opt.state_dict(), "update": update, "step": step,
                        "recent": recent, "best": best_score, "torch_rng": torch.get_rng_state(),
                        "np_rng": np.random.get_state()}, run_dir / "resume.pt")
            print(f"--max-hours reached: saved {run_dir / 'resume.pt'}, rerun with --resume")
            break

    envs.close()
    log_f.close()
    ep_f.close()
    eval_f.close()
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
