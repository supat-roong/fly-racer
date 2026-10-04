import numpy as np

from flyracer.connectome import Connectome, degree_preserving_shuffle, make_synthetic, shuffle_signs


def small():
    return make_synthetic(n_input=50, n_core=200, n_output=20, density=0.05, seed=1)


def test_roundtrip(tmp_path):
    c = small()
    c.save(tmp_path / "c.npz")
    d = Connectome.load(tmp_path / "c.npz")
    assert d.n_neurons == c.n_neurons and d.n_edges == c.n_edges
    np.testing.assert_array_equal(d.pre, c.pre)
    assert d.meta == c.meta


def test_shuffle_preserves_degrees_and_synapses():
    c = small()
    s = degree_preserving_shuffle(c, seed=3)
    n = c.n_neurons
    strength = lambda idx, w: np.bincount(idx, weights=w, minlength=n)
    # Counts travel with the presynaptic neuron, so out-strength is kept except
    # for dropped self-loops. In-degree is kept up to self-loops and merged duplicates.
    assert (strength(c.pre, c.count) - strength(s.pre, s.count) >= -1e-4).all()
    in_c, in_s = np.bincount(c.post, minlength=n), np.bincount(s.post, minlength=n)
    assert (in_s <= in_c).all() and in_s.sum() > 0.9 * in_c.sum()
    assert c.count.sum() - s.count.sum() < 0.01 * c.count.sum()
    assert (s.pre != s.post).all()
    assert not np.array_equal(np.sort(s.pre * n + s.post), np.sort(c.pre * n + c.post))


def test_shuffle_signs_keeps_ratio():
    c = small()
    s = shuffle_signs(c, seed=0)
    assert (s.sign < 0).sum() == (c.sign < 0).sum()
