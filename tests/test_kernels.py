import numpy as np
import pytest

from llmbo.kernels import KernelParseError, ard_spec, canonical, parse_kernel
from llmbo.gp import GP


@pytest.mark.parametrize("spec", ["RBF()", "RBF(0)*RBF(1)", "PER(0) + RBF(1)",
                                  "(PER(0)*RBF(0)) + LIN(1)", "rq(0,1) * mat32()"])
def test_parse_and_psd(spec):
    k = parse_kernel(spec, 2)
    X = np.random.default_rng(0).uniform(0, 1, (15, 2))
    K = k(X)
    assert K.shape == (15, 15)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_bad_specs():
    with pytest.raises(KernelParseError):
        parse_kernel("FOO(0)", 2)
    with pytest.raises(KernelParseError):
        parse_kernel("RBF(3)", 2)
    with pytest.raises(KernelParseError):
        parse_kernel("RBF(0) +", 2)


def test_product_variance_not_duplicated():
    k = parse_kernel("RBF(0)*RBF(1)*RBF(2)", 3)
    assert k.n_params() == 1 + 3  # one variance + three length-scales


def test_theta_roundtrip():
    k = parse_kernel("PER(0)*RBF(1) + LIN(0)", 2)
    th = np.linspace(-1, 1, k.n_params())
    k.set_theta(th)
    assert np.allclose(k.get_theta(), th)
    assert canonical("PER(0)*RBF(1) + LIN(0)") == repr(k)


def test_gp_recovers_periodic_signal():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (30, 1))
    y = np.sin(2 * np.pi * X[:, 0] / 0.5) + 0.01 * rng.normal(size=30)
    gp_per = GP(parse_kernel("PER(0)", 1))
    fr_per = gp_per.fit(X, y, n_restarts=4, rng=rng)
    gp_rbf = GP(parse_kernel("RBF(0)", 1))
    fr_rbf = gp_rbf.fit(X, y, n_restarts=4, rng=rng)
    assert fr_per.bic < fr_rbf.bic
    Xs = np.linspace(0, 1, 50)[:, None]
    mu, sd = gp_per.predict(Xs)
    assert np.max(np.abs(mu - np.sin(2 * np.pi * Xs[:, 0] / 0.5))) < 0.2
