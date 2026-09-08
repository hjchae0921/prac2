"""Bayesian optimisation loop with a pluggable kernel-selection strategy."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from .gp import GP, FitResult
from .kernels import Kernel, KernelParseError, parse_kernel
from .problems import Problem


@dataclass
class KernelContext:
    """Everything a strategy is allowed to see when proposing kernels."""
    problem: Problem
    X: np.ndarray
    y: np.ndarray
    iteration: int
    history: List[Dict] = field(default_factory=list)  # [{"spec", "bic", "lml", "iteration"}]
    current_spec: Optional[str] = None


class KernelStrategy:
    name = "base"

    def candidates(self, ctx: KernelContext) -> List[str]:
        raise NotImplementedError

    def refresh_every(self) -> int:
        return 10**9  # by default only called once (after the initial design)


@dataclass
class BOResult:
    problem: str
    strategy: str
    seed: int
    X: np.ndarray
    y: np.ndarray
    best_y: np.ndarray            # running best, length n_init + n_iter
    regret: np.ndarray            # best_y - f_min
    kernel_trace: List[Dict]      # which kernel was active when
    wall_time: float


def expected_improvement(mu, sigma, y_best, xi=0.0):
    z = (y_best - xi - mu) / np.maximum(sigma, 1e-12)
    return (y_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)


class BayesOpt:
    def __init__(self, problem: Problem, strategy: KernelStrategy, n_init: int = 5,
                 n_iter: int = 30, seed: int = 0, n_restarts: int = 2,
                 n_candidates: int = 2000, verbose: bool = False):
        self.problem = problem
        self.strategy = strategy
        self.n_init = n_init
        self.n_iter = n_iter
        self.seed = seed
        self.n_restarts = n_restarts
        self.n_candidates = n_candidates
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    # ---- kernel selection ----------------------------------------------------
    def _fit_candidates(self, specs: List[str], X, y, history) -> tuple[GP, FitResult, str]:
        best = None
        for spec in specs:
            try:
                k = parse_kernel(spec, self.problem.dim)
            except KernelParseError as e:
                if self.verbose:
                    print(f"    [skip] {spec!r}: {e}")
                continue
            gp = GP(k)
            fr = gp.fit(X, y, n_restarts=self.n_restarts, rng=self.rng, warm_start=False)
            history.append({"spec": repr(k), "bic": fr.bic, "lml": fr.log_marginal_likelihood,
                            "n_params": fr.n_params, "iteration": len(y)})
            if self.verbose:
                print(f"    cand {repr(k):40s} lml={fr.log_marginal_likelihood:8.2f} bic={fr.bic:8.2f}")
            if best is None or fr.bic < best[1].bic:
                best = (gp, fr, repr(k))
        if best is None:  # all invalid -> safe fallback
            from .kernels import ard_spec
            k = parse_kernel(ard_spec("RBF", self.problem.dim))
            gp = GP(k)
            fr = gp.fit(X, y, n_restarts=self.n_restarts, rng=self.rng, warm_start=False)
            best = (gp, fr, repr(k))
        return best

    # ---- acquisition ---------------------------------------------------------
    def _next_point(self, gp: GP, y_best: float) -> np.ndarray:
        d = self.problem.dim
        cands = self.rng.uniform(0, 1, size=(self.n_candidates, d))
        mu, sd = gp.predict(cands)
        ei = expected_improvement(mu, sd, y_best)
        top = np.argsort(-ei)[:5]

        def neg_ei(u):
            m, s = gp.predict(u[None, :])
            return -expected_improvement(m, s, y_best)[0]

        best_x, best_v = cands[top[0]], -ei[top[0]]
        for i in top:
            r = minimize(neg_ei, cands[i], method="L-BFGS-B", bounds=[(0, 1)] * d,
                         options={"maxiter": 50})
            if r.fun < best_v:
                best_x, best_v = np.clip(r.x, 0, 1), r.fun
        if best_v > -1e-12:  # EI numerically zero everywhere -> explore
            best_x = self.rng.uniform(0, 1, d)
        return best_x

    # ---- main loop -----------------------------------------------------------
    def run(self) -> BOResult:
        t0 = time.time()
        d = self.problem.dim
        f_min = self.problem.estimate_min()
        sampler = qmc.LatinHypercube(d=d, seed=self.seed)
        X = sampler.random(self.n_init)
        y = np.array([self.problem(x) for x in X])
        history: List[Dict] = []
        kernel_trace: List[Dict] = []

        gp, fr, spec = None, None, None
        refresh = self.strategy.refresh_every()
        for it in range(self.n_iter):
            need_select = (gp is None) or (it % refresh == 0)
            if need_select:
                ctx = KernelContext(self.problem, X.copy(), y.copy(), it, history, spec)
                specs = self.strategy.candidates(ctx)
                gp, fr, spec = self._fit_candidates(specs, X, y, history)
                kernel_trace.append({"iteration": it, "n_obs": len(y), "spec": spec,
                                     "bic": fr.bic, "lml": fr.log_marginal_likelihood,
                                     "hyper": gp.kernel.describe()})
                if self.verbose:
                    print(f"  it={it:3d} selected {spec}  (bic={fr.bic:.2f})")
            else:
                gp.fit(X, y, n_restarts=1, rng=self.rng, warm_start=True)
            x_new = self._next_point(gp, float(np.min(y)))
            y_new = self.problem(x_new)
            X = np.vstack([X, x_new])
            y = np.append(y, y_new)
            if self.verbose:
                print(f"  it={it:3d}  y={y_new:9.4f}  best={np.min(y):9.4f}")

        best_y = np.minimum.accumulate(y)
        return BOResult(self.problem.name, self.strategy.name, self.seed, X, y, best_y,
                        best_y - f_min, kernel_trace, time.time() - t0)
