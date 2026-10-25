import argparse

import pytest
import torch

from flyracer.checkpoint import WIRING, load_agent, save_checkpoint, strip_wiring
from flyracer.connectome import make_synthetic
from flyracer.models import Agent


def _save(tmp_path, core, conn_path):
    args = argparse.Namespace(core=core, rnn_steps=6, seed=0)
    conn = make_synthetic(n_input=40, n_core=150, n_output=12, density=0.05, seed=0)
    conn.save(conn_path)
    agent = Agent(core, conn, steps=6, seed=0)
    with torch.no_grad():
        agent.core.bias.add_(1.0)  # differs from a fresh agent
    save_checkpoint(tmp_path / "full.pt", agent, args, 0, str(conn_path))
    return agent


@pytest.mark.parametrize("core", ["fly", "random", "signs"])
def test_stripped_checkpoint_loads_the_same_agent(tmp_path, core):
    agent = _save(tmp_path, core, tmp_path / "conn.npz")
    strip_wiring(tmp_path / "full.pt", tmp_path / "stripped.pt")
    stripped = torch.load(tmp_path / "stripped.pt", weights_only=False)
    assert not set(WIRING) & set(stripped["state_dict"])
    loaded, *_ = load_agent(tmp_path / "stripped.pt")
    for k, v in agent.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)


def test_stripped_checkpoint_rejects_another_connectome(tmp_path):
    conn_path = tmp_path / "conn.npz"
    _save(tmp_path, "fly", conn_path)
    strip_wiring(tmp_path / "full.pt", tmp_path / "stripped.pt")
    other = make_synthetic(n_input=40, n_core=150, n_output=12, density=0.05, seed=0)
    other.sign = -other.sign
    other.save(conn_path)
    with pytest.raises(ValueError, match="different connectome"):
        load_agent(tmp_path / "stripped.pt")
