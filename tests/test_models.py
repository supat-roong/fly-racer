import pytest
import torch

from flyracer.connectome import make_synthetic
from flyracer.models import CORES, Agent, FlyRNN


@pytest.fixture(scope="module")
def conn():
    return make_synthetic(n_input=40, n_core=150, n_output=12, density=0.05, seed=0)


def test_flyrnn_respects_topology_and_dale(conn):
    net = FlyRNN(conn, n_features=8)
    W = net.weight_matrix().detach()
    mask = torch.zeros_like(W, dtype=torch.bool)
    mask[torch.as_tensor(conn.post).long(), torch.as_tensor(conn.pre).long()] = True
    assert (W[~mask] == 0).all()
    sign = torch.as_tensor(conn.sign, dtype=torch.float32)
    assert ((W * sign[None, :]) >= 0).all()  # column j carries neuron j's sign


def test_flyrnn_training_keeps_constraints(conn):
    net = FlyRNN(conn, n_features=8)
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    for _ in range(3):
        out, _ = net(torch.randn(4, 8))
        opt.zero_grad(); (-out.sum()).backward(); opt.step()
    W = net.weight_matrix().detach()
    sign = torch.as_tensor(conn.sign, dtype=torch.float32)
    assert ((W * sign[None, :]) >= 0).all()


@pytest.mark.parametrize("core", CORES)
def test_agent_shapes(conn, core):
    agent = Agent(core, conn)
    obs = torch.randint(0, 255, (5, 4, 96, 96), dtype=torch.uint8)
    action, logp, ent, value, rates = agent(obs)
    assert action.shape == (5, 3) and logp.shape == (5,) and value.shape == (5,)
    _, logp2, *_ = agent(obs, action=action)
    torch.testing.assert_close(logp, logp2)
    if core != "mlp":
        assert rates.shape == (5, conn.n_neurons) and (rates >= 0).all() and (rates < 1).all()


def test_env_actions_in_bounds(conn):
    agent = Agent("fly", conn)
    obs = torch.randint(0, 255, (64, 4, 96, 96), dtype=torch.uint8)
    u, *_ = agent(obs)
    a = agent.env_action(u)
    assert ((u >= 0) & (u <= 1)).all()
    assert (a[:, 0].abs() <= 1).all() and ((a[:, 1:] >= 0) & (a[:, 1:] <= 1)).all()
    mean, *_ = agent(obs, deterministic=True)
    assert mean[:, 2].mean() < 0.3  # starts light on the brake
