"""Kernel-selection strategies compared in the pilot study.

  fixed_rbf / fixed_mat52 : plain GP-BO with a fixed ARD kernel (the baseline)
  greedy                  : algorithmic compositional search (Automatic-Statistician style,
                            greedy expansion by '+'/'*' scored by BIC) - no language knowledge
  mock_llm                : OFFLINE stand-in for the LLM.  It reads the problem's
                            `structure` tags (which encode the same qualitative knowledge the
                            real LLM is given in words) and emits the kernels a knowledgeable
                            expert would.  It is an *upper bound* on "LLM with good domain
                            knowledge", NOT a measurement of any LLM.
  llm                     : real Claude call (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .bo import KernelContext, KernelStrategy
from .kernels import KernelParseError, ard_spec, canonical, parse_kernel


class FixedKernelStrategy(KernelStrategy):
    def __init__(self, base: str = "RBF"):
        self.base = base
        self.name = f"fixed_{base.lower()}"

    def candidates(self, ctx: KernelContext) -> List[str]:
        return [ard_spec(self.base, ctx.problem.dim)]


class GreedySearchStrategy(KernelStrategy):
    """Greedy compositional kernel search (Duvenaud et al. 2013 style), re-run every
    `refresh` BO iterations.  Depth-limited to keep the run time reasonable."""

    BASES = ["RBF", "MAT52", "PER", "LIN"]

    def __init__(self, refresh: int = 10, depth: int = 2, beam: int = 2):
        self.refresh = refresh
        self.depth = depth
        self.beam = beam
        self.name = "greedy_search"

    def refresh_every(self) -> int:
        return self.refresh

    def _expand(self, spec: str, d: int) -> List[str]:
        out = []
        for b in self.BASES:
            for dim in range(d):
                out.append(f"({spec}) + {b}({dim})")
                out.append(f"({spec}) * {b}({dim})")
        return out

    def candidates(self, ctx: KernelContext) -> List[str]:
        from .gp import GP
        d = ctx.problem.dim
        rng = np.random.default_rng(len(ctx.y))
        seeds = [f"{b}({dim})" for b in self.BASES for dim in range(d)] + [ard_spec("RBF", d)]
        scored: Dict[str, float] = {}

        def score(spec):
            c = canonical(spec, d)
            if c in scored:
                return scored[c]
            gp = GP(parse_kernel(spec, d))
            fr = gp.fit(ctx.X, ctx.y, n_restarts=1, rng=rng, warm_start=False, maxiter=100)
            scored[c] = fr.bic
            return fr.bic

        frontier = seeds
        for s in frontier:
            score(s)
        for _ in range(self.depth):
            frontier = sorted(set(canonical(s, d) for s in frontier), key=score)[:self.beam]
            children = [c for s in frontier for c in self._expand(s, d)]
            for c in children:
                score(c)
            frontier = frontier + children
        best = sorted(scored, key=scored.get)[:4]
        return best


class MockLLMStrategy(KernelStrategy):
    """Deterministic, rule-based proposer used when no API key is available.

    It maps the qualitative structure tags of a problem to kernel expressions in the
    same way a domain-aware expert would.  This isolates the question
    'does injecting correct structural knowledge into the kernel help BO?' from
    'can a particular LLM produce that knowledge?'."""

    def __init__(self, refresh: int = 10):
        self.refresh = refresh
        self.name = "mock_llm"

    def refresh_every(self) -> int:
        return self.refresh

    def candidates(self, ctx: KernelContext) -> List[str]:
        d = ctx.problem.dim
        st = ctx.problem.structure
        per = st.get("periodic", [])
        rough = st.get("rough", [])
        lin = st.get("linear", [])
        additive = st.get("additive", [])
        inter = st.get("interaction", [])

        def per_dim_kernel(i, local=True):
            if i in per:
                return f"PER({i})*RBF({i})" if local else f"PER({i})"
            if i in rough:
                return f"MAT32({i})"
            if i in lin:
                return f"LIN({i}) + RBF({i})"
            return f"RBF({i})"

        props: List[str] = []
        # 1) fully additive structured model
        if additive:
            props.append(" + ".join(f"({per_dim_kernel(i)})" for i in range(d)))
            props.append(" + ".join(f"({per_dim_kernel(i, local=False)})" for i in range(d)))
        # 2) product (interaction) structured model
        props.append("*".join(f"({per_dim_kernel(i, local=False)})" for i in range(d)))
        # 3) periodic-in-some-dims x smooth-in-others, plus smooth ARD remainder
        if per:
            rest = [i for i in range(d) if i not in per]
            if rest:
                props.append("(" + "*".join(f"PER({i})" for i in per) + ")*("
                             + "*".join(f"RBF({i})" for i in rest) + ") + " + ard_spec("RBF", d))
            else:
                props.append("(" + "*".join(f"PER({i})" for i in per) + ") + " + ard_spec("RBF", d))
        if inter and len(inter) >= 2:
            a, b = inter[:2]
            props.append(f"({per_dim_kernel(a, local=False)})*RBF({b}) + " + ard_spec("RBF", d))
        # 4) always include the plain ARD baseline so the mock can fall back to it
        props.append(ard_spec("RBF", d))
        if st.get("anisotropic"):
            props.append(ard_spec("MAT52", d))
        # de-duplicate, keep order
        out, seen = [], set()
        for p in props:
            try:
                c = canonical(p, d)
            except KernelParseError:
                continue
            if c not in seen:
                seen.add(c)
                out.append(p)
        return out[:5]


class LLMKernelStrategy(KernelStrategy):
    """Real LLM-in-the-loop strategy.  Every `refresh` iterations the LLM sees the problem
    description, the data, and the BIC of previously tried kernels, and proposes new ones."""

    def __init__(self, proposer=None, refresh: int = 10, model: Optional[str] = None,
                 backend: str = "claude", verbose: bool = False):
        from .llm_client import make_proposer
        self.proposer = proposer or make_proposer(backend, model)
        self.refresh = refresh
        self.verbose = verbose
        self.name = "llm"
        self.transcript: List[Dict] = []

    def refresh_every(self) -> int:
        return self.refresh

    def candidates(self, ctx: KernelContext) -> List[str]:
        from .llm_client import build_prompt, parse_proposals
        prompt = build_prompt(ctx.problem, ctx.X, ctx.y, ctx.iteration, ctx.history, ctx.current_spec)
        text = self.proposer(prompt)
        try:
            props = parse_proposals(text)
        except Exception as e:  # malformed answer -> baseline
            if self.verbose:
                print(f"    [llm] unparsable response ({e}); falling back to RBF-ARD")
            props = [{"kernel": ard_spec("RBF", ctx.problem.dim), "rationale": "fallback"}]
        valid = []
        for p in props:
            try:
                parse_kernel(p["kernel"], ctx.problem.dim)
                valid.append(p["kernel"])
            except KernelParseError as e:
                if self.verbose:
                    print(f"    [llm] rejected {p['kernel']!r}: {e}")
        self.transcript.append({"problem": ctx.problem.name, "iteration": ctx.iteration,
                                "prompt": prompt, "response": text, "valid": valid})
        if not valid:
            valid = [ard_spec("RBF", ctx.problem.dim)]
        return valid


def make_strategy(name: str, **kw) -> KernelStrategy:
    if name == "fixed_rbf":
        return FixedKernelStrategy("RBF")
    if name == "fixed_mat52":
        return FixedKernelStrategy("MAT52")
    if name == "greedy":
        return GreedySearchStrategy(refresh=kw.get("refresh", 10))
    if name == "mock_llm":
        return MockLLMStrategy(refresh=kw.get("refresh", 10))
    if name == "llm":
        return LLMKernelStrategy(refresh=kw.get("refresh", 10), model=kw.get("model"),
                                 backend=kw.get("backend", "claude"), verbose=kw.get("verbose", False))
    raise ValueError(f"unknown strategy {name!r}")
