import numpy as np

from flyracer.envs import make_env

NOOP = np.zeros(3, dtype=np.float32)


def test_observation_is_four_stacked_gray_frames():
    env = make_env(0, action_repeat=4)()
    obs, _ = env.reset(seed=0)
    assert np.asarray(obs).shape == (4, 96, 96)


def test_stalled_car_is_truncated_after_patience():
    env = make_env(0, action_repeat=4, patience=10)()
    env.reset(seed=0)
    steps, done = 0, False
    while not done:  # standing still never reaches a new tile
        _, _, term, trunc, _ = env.step(NOOP)
        steps += 1
        done = term or trunc
    assert trunc and not term and steps <= 12  # a tile or two may count at the start


def test_render_mode_keeps_every_repeated_frame():
    env = make_env(0, action_repeat=4, render_mode="rgb_array")()
    env.reset(seed=0)
    env.step(NOOP)
    frames = env.get_wrapper_attr("frames")
    assert len(frames) == 4 and frames[0].shape == (400, 600, 3)


def test_fast_env_matches_stock_car_racing():
    # Same track and actions: FastCarRacing (vectorized culling, skipped hidden
    # frames) must give the stock env's observations, rewards and episode ends.
    # Repeat 3 also ends on a frame that was not drawn (1000 frames % 3 != 0).
    for k in (2, 3):
        stock, fast = (make_env(0, action_repeat=k, fast=f)() for f in (False, True))
        a, _ = stock.reset(seed=5)
        b, _ = fast.reset(seed=5)
        assert np.array_equal(a, b)
        rng = np.random.default_rng(0)
        for t in range(600):
            act = np.array([0.6 * np.sin(t / 12), 0.3 + 0.4 * rng.random(), 0.05 * rng.random()], np.float32)
            a, ra, ta, ua, _ = stock.step(act)
            b, rb, tb, ub, _ = fast.step(act)
            assert np.array_equal(a, b) and ra == rb and (ta, ua) == (tb, ub)
            if ta or ua:
                break
        assert ua  # reached the time limit
