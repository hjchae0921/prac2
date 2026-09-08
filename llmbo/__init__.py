"""LLM-guided kernel design for Bayesian optimisation: pilot simulation package."""
from .kernels import parse_kernel, Kernel
from .gp import GP
from .bo import BayesOpt
from .problems import PROBLEMS, Problem
from .strategies import (
    FixedKernelStrategy,
    GreedySearchStrategy,
    LLMKernelStrategy,
    MockLLMStrategy,
    make_strategy,
)

__all__ = [
    "parse_kernel", "Kernel", "GP", "BayesOpt", "PROBLEMS", "Problem",
    "FixedKernelStrategy", "GreedySearchStrategy", "LLMKernelStrategy",
    "MockLLMStrategy", "make_strategy",
]
