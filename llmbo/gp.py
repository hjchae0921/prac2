"""Minimal Gaussian-process regression with marginal-likelihood hyper-parameter fitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .kernels import Kernel

_LOG_NOISE_BOUNDS = (np.log(1e-6), np.log(1.0))


@dataclass
class FitResult:
    log_marginal_likelihood: float
    n_params: int
    bic: float
    theta: np.ndarray


class GP:
    """Zero-mean GP on standardised targets.  `kernel` is any `Kernel` from kernels.py."""

    def __init__(self, kernel: Kernel, noise: float = 1e-3, jitter: float = 1e-8):
        self.kernel = kernel
        self.log_noise = float(np.log(noise))
        self.jitter = jitter
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._L = None
        self._alpha = None

    # ---- parameter vector = [kernel theta..., log_noise] ---------------------
    def _get_params(self) -> np.ndarray:
        return np.concatenate([self.kernel.get_theta(), [self.log_noise]])

    def _set_params(self, p: np.ndarray) -> None:
        self.kernel.set_theta(p[:-1])
        self.log_noise = float(p[-1])

    def _bounds(self):
        return self.kernel.bounds() + [_LOG_NOISE_BOUNDS]

    def n_params(self) -> int:
        return self.kernel.n_params() + 1

    # ---- likelihood ----------------------------------------------------------
    def _nll(self, p: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        self._set_params(p)
        n = len(y)
        K = self.kernel(X) + (np.exp(self.log_noise) + self.jitter) * np.eye(n)
        try:
            L, low = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve((L, low), y, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
        return float(nll) if np.isfinite(nll) else 1e10

    def fit(self, X: np.ndarray, y: np.ndarray, n_restarts: int = 2,
            rng: Optional[np.random.Generator] = None, warm_start: bool = True,
            maxiter: int = 200) -> FitResult:
        """Type-II ML with multi-restart L-BFGS-B (numerical gradients)."""
        rng = np.random.default_rng(0) if rng is None else rng
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) if len(y) > 1 and np.std(y) > 1e-12 else 1.0
        ys = (y - self._y_mean) / self._y_std

        bounds = self._bounds()
        starts = []
        if warm_start:
            starts.append(self._get_params())
        for _ in range(n_restarts):
            kt = self.kernel.random_theta(rng)
            ln = rng.uniform(np.log(1e-5), np.log(1e-1))
            starts.append(np.concatenate([kt, [ln]]))

        best_p, best_f = None, np.inf
        for p0 in starts:
            p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
            try:
                res = minimize(self._nll, p0, args=(X, ys), method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": maxiter})
                f, p = res.fun, res.x
            except Exception:
                f, p = self._nll(p0, X, ys), p0
            if f < best_f:
                best_f, best_p = f, p
        self._set_params(best_p)
        self.X, self.y = X, ys
        self._precompute()
        lml = -best_f
        k = self.n_params()
        return FitResult(lml, k, -2.0 * lml + k * np.log(len(y)), best_p.copy())

    def _precompute(self):
        n = len(self.y)
        K = self.kernel(self.X) + (np.exp(self.log_noise) + self.jitter) * np.eye(n)
        self._L = cho_factor(K, lower=True, check_finite=False)
        self._alpha = cho_solve(self._L, self.y, check_finite=False)

    def predict(self, Xs: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self.kernel(Xs, self.X)
        mu = Ks @ self._alpha
        mu = mu * self._y_std + self._y_mean
        if not return_std:
            return mu, None
        v = cho_solve(self._L, Ks.T, check_finite=False)
        kss = np.einsum("ij,ij->i", Ks, v.T)
        diag = np.array([self.kernel(x[None, :])[0, 0] for x in Xs])
        var = np.maximum(diag - kss, 1e-12)
        return mu, np.sqrt(var) * self._y_std
