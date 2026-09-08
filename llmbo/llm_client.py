"""Claude-backed kernel proposer: prompt construction, JSON parsing, on-disk response cache."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_MODEL = os.environ.get("LLMBO_MODEL", "claude-opus-5")
DEFAULT_OPENAI_MODEL = os.environ.get("LLMBO_OPENAI_MODEL", "gpt-5.4-mini")

SYSTEM_PROMPT = """You are an expert in Gaussian-process modelling and Bayesian optimisation.
Your job: given a description of the design variables, the objective, qualitative domain
knowledge, and the data observed so far, propose covariance (kernel) functions for a GP
surrogate that will be used inside Bayesian optimisation (minimisation, Expected Improvement).

Kernel expression grammar (inputs are scaled to the unit cube [0,1]^d; dims are 0-based):
  expr   := term ('+' term)*
  term   := factor ('*' factor)*
  factor := BASE '(' dims ')' | '(' expr ')'
  BASE   := RBF | MAT32 | MAT52 | PER | LIN | RQ
  dims   := comma-separated 0-based input dimensions; empty parentheses = all dimensions
Base kernels: RBF (smooth), MAT32 / MAT52 (rougher), PER (exactly periodic; the period is
learned), LIN (linear trend), RQ (multi-scale smoothness).
Guidelines: '+' encodes additive effects, '*' encodes interactions / modulation
(e.g. PER(0)*RBF(0) = locally periodic, PER(0)*RBF(1) = periodic in x0 with amplitude
varying in x1). A product over every dimension, e.g. RBF(0)*RBF(1)*RBF(2), is an ARD
kernel. Keep expressions compact: more hyper-parameters need more data, and candidates are
ranked by BIC on the observed data.

Respond with ONLY a JSON object of the form
{"proposals": [{"kernel": "<expression>", "rationale": "<one sentence>"}, ...]}
with 3 to 5 distinct proposals ordered from most to least promising."""


def _fmt_obs(X: np.ndarray, y: np.ndarray, max_rows: int = 40) -> str:
    n = len(y)
    idx = np.arange(n) if n <= max_rows else np.sort(np.random.default_rng(0).choice(n, max_rows, replace=False))
    lines = ["  " + "  ".join(f"x{d}" for d in range(X.shape[1])) + "     y"]
    for i in idx:
        lines.append("  " + "  ".join(f"{v:5.3f}" for v in X[i]) + f"  {y[i]:9.4f}")
    return "\n".join(lines)


def build_prompt(problem, X: np.ndarray, y: np.ndarray, iteration: int,
                 history: List[Dict], current_spec: Optional[str]) -> str:
    var_lines = "\n".join(f"  - x{i} ({v['name']}): {v['description']}" for i, v in enumerate(problem.variables))
    hist = ""
    if history:
        rows = sorted(history, key=lambda h: h["bic"])[:8]
        hist = "\nKernels already evaluated on this data (lower BIC is better):\n" + "\n".join(
            f"  - {h['spec']}: BIC={h['bic']:.1f}, log-lik={h['lml']:.1f}, n_params={h['n_params']} (fit at n={h['iteration']} points)"
            for h in rows)
        if current_spec:
            hist += f"\nCurrently active kernel: {current_spec}"
    return f"""Problem: {problem.name} ({problem.dim} input dimensions)
Design variables:
{var_lines}
Objective: {problem.objective_description}
Domain knowledge: {problem.domain_knowledge}

Observations so far (n={len(y)}, inputs scaled to [0,1], BO iteration {iteration}):
{_fmt_obs(X, y)}
Best observed y = {np.min(y):.4f} at x = {np.round(X[np.argmin(y)], 3).tolist()}
{hist}

Propose kernel expressions for the GP surrogate."""


def parse_proposals(text: str) -> List[Dict[str, str]]:
    """Extract {"proposals": [...]} from a text block, tolerating surrounding prose / fences."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in LLM response")
    obj = json.loads(m.group(0))
    props = obj.get("proposals", [])
    out = []
    for p in props:
        if isinstance(p, dict) and "kernel" in p:
            out.append({"kernel": str(p["kernel"]).strip(), "rationale": str(p.get("rationale", ""))})
        elif isinstance(p, str):
            out.append({"kernel": p.strip(), "rationale": ""})
    if not out:
        raise ValueError("LLM response contained no proposals")
    return out


class ClaudeProposer:
    """Calls the Anthropic API (official SDK).  Responses are cached on disk keyed by
    a hash of (model, system, prompt) so re-running an experiment is free and deterministic."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str = "results/llm_cache",
                 log_path: Optional[str] = "results/llm_log.jsonl", max_tokens: int = 4000,
                 effort: Optional[str] = None):
        import anthropic  # imported lazily so the offline mock never needs it
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else None
        self.n_calls = 0
        self.n_cache_hits = 0

    def _cache_key(self, prompt: str) -> Path:
        h = hashlib.sha256((self.model + "\n" + SYSTEM_PROMPT + "\n" + prompt).encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def __call__(self, prompt: str) -> str:
        path = self._cache_key(prompt)
        if path.exists():
            self.n_cache_hits += 1
            return json.loads(path.read_text())["text"]
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT,
                      messages=[{"role": "user", "content": prompt}])
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.messages.create(**kwargs)
        self.n_calls += 1
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        rec = {"model": self.model, "prompt": prompt, "text": text,
               "usage": resp.usage.to_dict() if hasattr(resp.usage, "to_dict") else str(resp.usage)}
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return text


class OpenAIProposer:
    """Same interface as ClaudeProposer but calls the OpenAI Chat Completions API
    (official `openai` SDK).  Requires OPENAI_API_KEY.  Responses are cached on disk
    exactly like the Claude backend, keyed by (model, system prompt, user prompt)."""

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, cache_dir: str = "results/llm_cache",
                 log_path: Optional[str] = "results/llm_log.jsonl", max_tokens: int = 4000,
                 reasoning_effort: Optional[str] = None):
        import openai  # lazy import
        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else None
        self.n_calls = 0
        self.n_cache_hits = 0

    def _cache_key(self, prompt: str) -> Path:
        h = hashlib.sha256((self.model + "\n" + SYSTEM_PROMPT + "\n" + prompt).encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def __call__(self, prompt: str) -> str:
        path = self._cache_key(prompt)
        if path.exists():
            self.n_cache_hits += 1
            return json.loads(path.read_text())["text"]
        kwargs = dict(model=self.model,
                      messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}],
                      max_completion_tokens=self.max_tokens)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(**kwargs)
        self.n_calls += 1
        text = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage is not None else None
        rec = {"model": self.model, "prompt": prompt, "text": text, "usage": usage}
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return text


def make_proposer(backend: str = "claude", model: Optional[str] = None, **kw):
    if backend == "claude":
        return ClaudeProposer(model=model or DEFAULT_MODEL, **kw)
    if backend == "openai":
        return OpenAIProposer(model=model or DEFAULT_OPENAI_MODEL, **kw)
    raise ValueError(f"unknown LLM backend {backend!r} (use 'claude' or 'openai')")
