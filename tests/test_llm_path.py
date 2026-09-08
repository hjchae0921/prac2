"""End-to-end check of the LLM strategy using a stub proposer (no API call)."""
import numpy as np

from llmbo import BayesOpt
from llmbo.llm_client import build_prompt, parse_proposals
from llmbo.problems import get_problem
from llmbo.strategies import LLMKernelStrategy


class StubProposer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return ('Here you go:\n```json\n{"proposals": [{"kernel": "PER(0)*RBF(0) + RBF(1)", "rationale": "periodic x0"},'
                ' {"kernel": "FOO(0)", "rationale": "invalid on purpose"},'
                ' {"kernel": "RBF(0)*RBF(1)", "rationale": "ard"}]}\n```')


def test_parse_proposals_tolerates_fences_and_prose():
    props = parse_proposals(StubProposer()(""))
    assert [p["kernel"] for p in props] == ["PER(0)*RBF(0) + RBF(1)", "FOO(0)", "RBF(0)*RBF(1)"]


def test_prompt_contains_knowledge_and_history():
    prob = get_problem("sinlin2d")
    X = np.random.default_rng(0).uniform(0, 1, (5, 2))
    y = np.array([prob(x) for x in X])
    hist = [{"spec": "(RBF(0) * RBF(1))", "bic": 12.3, "lml": -4.1, "n_params": 4, "iteration": 5}]
    p = build_prompt(prob, X, y, 3, hist, "(RBF(0) * RBF(1))")
    assert prob.domain_knowledge in p and "BIC=12.3" in p and "Currently active kernel" in p


def test_llm_strategy_runs_bo_and_rejects_invalid():
    stub = StubProposer()
    strat = LLMKernelStrategy(proposer=stub, refresh=3)
    prob = get_problem("sinlin2d")
    r = BayesOpt(prob, strat, n_init=4, n_iter=6, seed=0, n_restarts=1).run()
    assert len(r.y) == 10
    assert len(stub.prompts) == 2                      # iterations 0 and 3
    assert all("FOO" not in v for t in strat.transcript for v in t["valid"])
    assert all("PER(0)" in t["valid"][0] for t in strat.transcript)
