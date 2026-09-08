"""Benchmark objectives with *natural-language* descriptions (the knowledge an LLM would receive).

Each problem is minimised.  `structure` is a machine-readable summary used only by the
offline MockLLMStrategy; the real LLM only ever sees `variables` / `objective_description`
/ `domain_knowledge` (see llm_client.build_prompt)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class Problem:
    name: str
    f: Callable[[np.ndarray], float]
    bounds: np.ndarray                      # (d, 2)
    variables: List[Dict[str, str]]         # [{"name", "description"}]
    objective_description: str
    domain_knowledge: str                   # qualitative hints, what a domain expert would tell the LLM
    structure: Dict[str, List[int]] = field(default_factory=dict)  # mock-only
    f_min: Optional[float] = None

    @property
    def dim(self) -> int:
        return self.bounds.shape[0]

    def __call__(self, x_unit: np.ndarray) -> float:
        """Evaluate at a point in the unit cube."""
        x = self.bounds[:, 0] + np.asarray(x_unit) * (self.bounds[:, 1] - self.bounds[:, 0])
        return float(self.f(x))

    def estimate_min(self, n_starts: int = 200, seed: int = 0) -> float:
        if self.f_min is not None:
            return self.f_min
        rng = np.random.default_rng(seed)
        best = np.inf
        for _ in range(n_starts):
            x0 = rng.uniform(0, 1, self.dim)
            r = minimize(lambda u: self(np.clip(u, 0, 1)), x0, method="L-BFGS-B",
                         bounds=[(0, 1)] * self.dim)
            best = min(best, float(r.fun))
        self.f_min = best
        return best


# ---------------------------------------------------------------------------
def _sinlin(x):
    x1, x2 = x
    return 2.0 * np.sin(4 * np.pi * x1) + 5.0 * (x2 - 0.6) ** 2 + 0.3 * x1


def _branin(x):
    x1, x2 = x
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def _ackley(x):
    x = np.asarray(x)
    d = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e)


_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
_H6_alpha = np.array([1.0, 1.2, 3.0, 3.2])


def _hartmann6(x):
    x = np.asarray(x)
    inner = np.sum(_H6_A * (x[None, :] - _H6_P) ** 2, axis=1)
    return -np.sum(_H6_alpha * np.exp(-inner))


def _additive4(x):
    x1, x2, x3, x4 = x
    return (np.sin(6 * np.pi * x1)                # periodic
            + 4.0 * (x2 - 0.7) ** 2               # smooth quadratic
            + 1.5 * x3                            # linear
            - 2.0 * np.exp(-((x4 - 0.25) / 0.1) ** 2))  # narrow bump (short length-scale)


def _gear_like(x):
    """A smooth 3-D response with a periodic term in an angle variable and an interaction."""
    theta, load, ratio = x
    return (0.6 * np.cos(3 * theta) * (1 + 0.5 * load)
            + (load - 0.4) ** 2 * 3.0
            + 0.8 * (ratio - 0.5) ** 2 * (1 + 2 * load))


PROBLEMS: Dict[str, Problem] = {
    "sinlin2d": Problem(
        name="sinlin2d", f=_sinlin, bounds=np.array([[0, 1], [0, 1]], float),
        variables=[
            {"name": "x1", "description": "phase-like control parameter; the response is known to oscillate as x1 sweeps its range (about 2 full cycles)"},
            {"name": "x2", "description": "a tuning parameter with a single well-defined optimum; response grows smoothly away from it"},
        ],
        objective_description="A cost to be minimised.",
        domain_knowledge="The x1 effect is periodic with a slight drift; the x2 effect is roughly quadratic. The two effects are believed to be approximately additive.",
        structure={"periodic": [0], "smooth": [1], "additive": [0, 1]},
    ),
    "branin": Problem(
        name="branin", f=_branin, bounds=np.array([[-5, 10], [0, 15]], float),
        variables=[
            {"name": "x1", "description": "first design variable in [-5, 10]; the response contains a cosine term in x1 on top of a quadratic-type interaction"},
            {"name": "x2", "description": "second design variable in [0, 15]; enters through a smooth quadratic interaction with x1"},
        ],
        objective_description="Branin-type smooth cost with three global minima; to be minimised.",
        domain_knowledge="Smooth, multimodal, with a low-amplitude oscillatory component along x1 and a strong smooth interaction between x1 and x2.",
        structure={"periodic": [0], "smooth": [0, 1], "interaction": [0, 1]},
        f_min=0.397887,
    ),
    "ackley2d": Problem(
        name="ackley2d", f=_ackley, bounds=np.array([[-5, 5], [-5, 5]], float),
        variables=[
            {"name": "x1", "description": "design variable in [-5, 5]"},
            {"name": "x2", "description": "design variable in [-5, 5]"},
        ],
        objective_description="Rugged cost: a broad bowl overlaid with a regular grid of local minima (spacing 1.0 in each variable); minimum at the origin.",
        domain_knowledge="Global trend is a radially symmetric bowl; superimposed oscillations are periodic with period 1 in every variable.",
        structure={"periodic": [0, 1], "smooth": [0, 1], "additive": [0, 1]},
        f_min=0.0,
    ),
    "hartmann6": Problem(
        name="hartmann6", f=_hartmann6, bounds=np.array([[0, 1]] * 6, float),
        variables=[{"name": f"x{i+1}", "description": f"design variable {i+1} in [0, 1]"} for i in range(6)],
        objective_description="Smooth 6-D negative sum of Gaussians (Hartmann-6); minimised.",
        domain_knowledge="Smooth everywhere; some variables matter much more than others (anisotropic); no periodicity.",
        structure={"smooth": [0, 1, 2, 3, 4, 5], "anisotropic": [0, 1, 2, 3, 4, 5]},
        f_min=-3.32237,
    ),
    "additive4d": Problem(
        name="additive4d", f=_additive4, bounds=np.array([[0, 1]] * 4, float),
        variables=[
            {"name": "x1", "description": "an angle-like parameter; the response oscillates about 3 times over its range"},
            {"name": "x2", "description": "a parameter with a smooth bowl-shaped effect"},
            {"name": "x3", "description": "a parameter with an approximately linear (monotone) effect"},
            {"name": "x4", "description": "a parameter whose effect is a narrow dip (sharp feature) somewhere in its range"},
        ],
        objective_description="Cost that is a sum of four independent one-dimensional effects; minimised.",
        domain_knowledge="The effects are additive across variables: periodic in x1, quadratic in x2, linear in x3, and a localised sharp dip in x4.",
        structure={"periodic": [0], "smooth": [1], "linear": [2], "rough": [3], "additive": [0, 1, 2, 3]},
    ),
    "gear3d": Problem(
        name="gear3d", f=_gear_like, bounds=np.array([[0, 2 * np.pi], [0, 1], [0, 1]], float),
        variables=[
            {"name": "theta", "description": "mounting angle in radians [0, 2pi]; the response is periodic in theta with three lobes"},
            {"name": "load", "description": "normalised load in [0, 1]; smooth effect with an interior optimum, also modulates the amplitude of the theta effect"},
            {"name": "ratio", "description": "normalised gear ratio in [0, 1]; smooth bowl whose curvature increases with load"},
        ],
        objective_description="Vibration-like cost to be minimised.",
        domain_knowledge="Periodic in theta (period 2pi/3); smooth in load and ratio; the theta and load effects interact multiplicatively.",
        structure={"periodic": [0], "smooth": [1, 2], "interaction": [0, 1]},
    ),
}


def get_problem(name: str) -> Problem:
    p = PROBLEMS[name]
    p.estimate_min()
    return p
