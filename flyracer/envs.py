"""CarRacing-v3 with the standard preprocessing for pixel-based driving."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    """Repeat each action `k` frames and sum rewards (the car barely moves in one frame).

    With an rgb_array render mode, every intermediate frame is kept in `frames`
    so a viewer can show smooth 50 fps video instead of every k-th frame.
    """

    def __init__(self, env, k: int):
        super().__init__(env)
        self.k = k
        self.frames: list[np.ndarray] = []

    def step(self, action):
        total = 0.0
        self.frames = []
        for _ in range(self.k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            if self.env.render_mode == "rgb_array":
                self.frames.append(self.env.render())
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


class NoProgressTruncation(gym.Wrapper):
    """End the episode after `patience` decisions without a positive reward (no new track tile).

    Training only: an agent parked on the grass otherwise burns the rest of the
    1000-frame episode learning nothing. Ends as truncation, so PPO bootstraps.
    """

    def __init__(self, env, patience: int):
        super().__init__(env)
        self.patience = patience
        self.stalled = 0

    def reset(self, **kwargs):
        self.stalled = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.stalled = 0 if reward > 0 else self.stalled + 1
        if self.stalled >= self.patience:
            truncated = True
        return obs, reward, terminated, truncated, info


def make_env(seed: int = 0, action_repeat: int = 2, render_mode: str | None = None,
             patience: int | None = None, normalize_reward: bool = False):
    def thunk():
        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        env = ActionRepeat(env, action_repeat)
        if patience:
            env = NoProgressTruncation(env, patience)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # true game score, before normalization
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
            env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk


def make_vec_env(n: int, seed: int = 0, action_repeat: int = 2, patience: int | None = 100,
                 normalize_reward: bool = True, asynchronous: bool = True):
    fns = [make_env(seed + i, action_repeat, patience=patience, normalize_reward=normalize_reward)
           for i in range(n)]
    cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return cls(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def episode_returns(infos: dict) -> list[float]:
    """Episodic game scores for the envs that just finished (vector-env info format)."""
    ep = infos.get("episode")
    if ep is None:
        return []
    done = infos.get("_episode", np.ones(len(ep["r"]), dtype=bool))
    return [float(r) for r in np.asarray(ep["r"])[done]]
