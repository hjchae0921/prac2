"""Composable GP kernels with a tiny expression grammar.

Grammar (what the LLM / search strategy writes):

    expr    := term ('+' term)*
    term    := factor ('*' factor)*
    factor  := BASE '(' dims? ')' | '(' expr ')'
    BASE    := RBF | MAT32 | MAT52 | PER | LIN | RQ
    dims    := int (',' int)*        # active input dimensions, empty = all

Examples:  "RBF()", "RBF(0)*RBF(1)", "PER(0) + RBF(1)", "(PER(0)*RBF(0)) + LIN(1)"

All hyper-parameters live in log space in a flat vector `theta`.  Inputs are
assumed to be scaled to the unit cube [0, 1]^d; the default bounds below rely
on that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_LOG_VAR_BOUNDS = (np.log(1e-3), np.log(1e3))
_LOG_LS_BOUNDS = (np.log(0.02), np.log(5.0))
_LOG_PERIOD_BOUNDS = (np.log(0.05), np.log(3.0))
_LOG_ALPHA_BOUNDS = (np.log(0.05), np.log(50.0))


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    d = a2 + b2 - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


class Kernel:
    """Base class.  Sub-classes implement `_k(X1, X2)` using `self.theta`."""

    name = "K"

    def __init__(self, active_dims: Optional[Sequence[int]] = None, fix_variance: bool = False):
        self.active_dims = None if active_dims is None else tuple(int(d) for d in active_dims)
        self.fix_variance = fix_variance  # used inside products: only one factor keeps a variance
        self.theta = self.init_theta()

    # ---- parameter interface -------------------------------------------------
    def init_theta(self) -> np.ndarray:  # override
        return np.zeros(0)

    def bounds(self) -> List[Tuple[float, float]]:  # override
        return []

    def n_params(self) -> int:
        return len(self.theta)

    def get_theta(self) -> np.ndarray:
        return self.theta.copy()

    def set_theta(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, dtype=float).copy()

    def random_theta(self, rng: np.random.Generator) -> np.ndarray:
        b = np.array(self.bounds())
        if len(b) == 0:
            return np.zeros(0)
        return rng.uniform(b[:, 0], b[:, 1])

    def param_names(self) -> List[str]:
        return []

    # ---- evaluation ----------------------------------------------------------
    def _slice(self, X: np.ndarray) -> np.ndarray:
        if self.active_dims is None:
            return X
        return X[:, list(self.active_dims)]

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        return self._k(self._slice(np.atleast_2d(X1)), self._slice(np.atleast_2d(X2)))

    def _k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _dims_str(self) -> str:
        return "" if self.active_dims is None else ",".join(str(d) for d in self.active_dims)

    def __repr__(self) -> str:
        return f"{self.name}({self._dims_str()})"

    def describe(self) -> str:
        """Human-readable summary including fitted hyper-parameters (natural scale)."""
        names = self.param_names()
        vals = np.exp(self.theta)
        inner = ", ".join(f"{n}={v:.3g}" for n, v in zip(names, vals))
        return f"{self.name}({self._dims_str()})[{inner}]"

    # ---- algebra -------------------------------------------------------------
    def __add__(self, other: "Kernel") -> "Kernel":
        return Sum([self, other])

    def __mul__(self, other: "Kernel") -> "Kernel":
        return Product([self, other])


class _Stationary(Kernel):
    """Shared plumbing for kernels with (variance, lengthscale)."""

    def init_theta(self) -> np.ndarray:
        return np.array([0.0, np.log(0.3)]) if not self.fix_variance else np.array([np.log(0.3)])

    def bounds(self):
        return ([_LOG_VAR_BOUNDS] if not self.fix_variance else []) + [_LOG_LS_BOUNDS]

    def param_names(self):
        return (["var"] if not self.fix_variance else []) + ["ls"]

    def _var_ls(self):
        if self.fix_variance:
            return 1.0, np.exp(self.theta[0])
        return np.exp(self.theta[0]), np.exp(self.theta[1])


class RBF(_Stationary):
    name = "RBF"

    def _k(self, A, B):
        v, ls = self._var_ls()
        return v * np.exp(-0.5 * _sqdist(A, B) / ls**2)


class Matern32(_Stationary):
    name = "MAT32"

    def _k(self, A, B):
        v, ls = self._var_ls()
        r = np.sqrt(_sqdist(A, B)) / ls
        s = np.sqrt(3.0) * r
        return v * (1.0 + s) * np.exp(-s)


class Matern52(_Stationary):
    name = "MAT52"

    def _k(self, A, B):
        v, ls = self._var_ls()
        r = np.sqrt(_sqdist(A, B)) / ls
        s = np.sqrt(5.0) * r
        return v * (1.0 + s + s * s / 3.0) * np.exp(-s)


class RQ(Kernel):
    """Rational quadratic: scale mixture of RBFs (multi-scale smoothness)."""

    name = "RQ"

    def init_theta(self):
        base = [np.log(0.3), np.log(1.0)]
        return np.array(([0.0] if not self.fix_variance else []) + base)

    def bounds(self):
        return ([_LOG_VAR_BOUNDS] if not self.fix_variance else []) + [_LOG_LS_BOUNDS, _LOG_ALPHA_BOUNDS]

    def param_names(self):
        return (["var"] if not self.fix_variance else []) + ["ls", "alpha"]

    def _k(self, A, B):
        t = np.exp(self.theta)
        if self.fix_variance:
            v, ls, al = 1.0, t[0], t[1]
        else:
            v, ls, al = t[0], t[1], t[2]
        return v * (1.0 + _sqdist(A, B) / (2.0 * al * ls**2)) ** (-al)


class Periodic(Kernel):
    """Exactly periodic kernel (MacKay): exp(-2 sum_d sin^2(pi (x_d - x'_d)/p) / ls^2)."""

    name = "PER"

    def init_theta(self):
        base = [np.log(0.5), np.log(0.5)]  # ls, period
        return np.array(([0.0] if not self.fix_variance else []) + base)

    def bounds(self):
        return ([_LOG_VAR_BOUNDS] if not self.fix_variance else []) + [_LOG_LS_BOUNDS, _LOG_PERIOD_BOUNDS]

    def param_names(self):
        return (["var"] if not self.fix_variance else []) + ["ls", "period"]

    def _k(self, A, B):
        t = np.exp(self.theta)
        if self.fix_variance:
            v, ls, p = 1.0, t[0], t[1]
        else:
            v, ls, p = t[0], t[1], t[2]
        diff = A[:, None, :] - B[None, :, :]
        s = np.sin(np.pi * diff / p) ** 2
        return v * np.exp(-2.0 * np.sum(s, axis=2) / ls**2)


class Linear(Kernel):
    """Linear kernel with offset: var * (x - c)(x' - c)^T + bias.  c is fixed at 0.5 (centre of unit cube)."""

    name = "LIN"

    def init_theta(self):
        return np.array(([0.0] if not self.fix_variance else []) + [np.log(0.1)])

    def bounds(self):
        return ([_LOG_VAR_BOUNDS] if not self.fix_variance else []) + [_LOG_VAR_BOUNDS]

    def param_names(self):
        return (["var"] if not self.fix_variance else []) + ["bias"]

    def _k(self, A, B):
        t = np.exp(self.theta)
        v, b = (1.0, t[0]) if self.fix_variance else (t[0], t[1])
        return v * ((A - 0.5) @ (B - 0.5).T) + b


class _Composite(Kernel):
    op = "?"

    def __init__(self, parts: Sequence[Kernel]):
        self.parts = list(parts)
        self.active_dims = None
        self.fix_variance = False
        self.theta = self.init_theta()

    def init_theta(self):
        return np.concatenate([p.get_theta() for p in self.parts]) if self.parts else np.zeros(0)

    def get_theta(self):
        return np.concatenate([p.get_theta() for p in self.parts])

    def set_theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        i = 0
        for p in self.parts:
            n = p.n_params()
            p.set_theta(theta[i:i + n])
            i += n
        self.theta = theta.copy()

    def n_params(self):
        return sum(p.n_params() for p in self.parts)

    def bounds(self):
        return [b for p in self.parts for b in p.bounds()]

    def param_names(self):
        return [f"{i}.{n}" for i, p in enumerate(self.parts) for n in p.param_names()]

    def random_theta(self, rng):
        return np.concatenate([p.random_theta(rng) for p in self.parts])

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return self._combine([p(X1, X2) for p in self.parts])

    def _combine(self, mats):
        raise NotImplementedError

    def __repr__(self):
        return "(" + f" {self.op} ".join(repr(p) for p in self.parts) + ")"

    def describe(self):
        return "(" + f" {self.op} ".join(p.describe() for p in self.parts) + ")"


class Sum(_Composite):
    op = "+"

    def _combine(self, mats):
        out = mats[0].copy()
        for m in mats[1:]:
            out += m
        return out


class Product(_Composite):
    op = "*"

    def __init__(self, parts):
        parts = list(parts)
        # Only the first factor keeps a free variance; the rest are fixed to 1
        # so the product is not over-parameterised.
        for p in parts[1:]:
            _fix_variance(p)
        super().__init__(parts)

    def _combine(self, mats):
        out = mats[0].copy()
        for m in mats[1:]:
            out *= m
        return out


def _fix_variance(k: Kernel) -> None:
    if isinstance(k, Sum):
        return  # a sum inside a product keeps its own variances (needed for identifiability of summands)
    if isinstance(k, Product):
        _fix_variance(k.parts[0])
        k.theta = k.init_theta()
        return
    if not k.fix_variance:
        k.fix_variance = True
        k.theta = k.init_theta()


BASE_KERNELS = {
    "RBF": RBF, "SE": RBF,
    "MAT32": Matern32, "MATERN32": Matern32,
    "MAT52": Matern52, "MATERN52": Matern52,
    "PER": Periodic, "PERIODIC": Periodic,
    "LIN": Linear, "LINEAR": Linear,
    "RQ": RQ,
}

_TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z0-9_]*)|(.))")


class KernelParseError(ValueError):
    pass


def _tokenize(s: str) -> List[str]:
    out = []
    pos = 0
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise KernelParseError(f"cannot tokenize at {pos!r} in {s!r}")
        pos = m.end()
        tok = m.group(1) or m.group(2) or m.group(3)
        if tok is None:
            continue
        if tok.strip() == "":
            continue
        out.append(tok)
    return out


class _Parser:
    def __init__(self, s: str, n_dims: Optional[int]):
        self.toks = _tokenize(s)
        self.i = 0
        self.n_dims = n_dims
        self.src = s

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expect: Optional[str] = None):
        t = self.peek()
        if t is None or (expect is not None and t != expect):
            raise KernelParseError(f"expected {expect!r}, got {t!r} in {self.src!r}")
        self.i += 1
        return t

    def expr(self) -> Kernel:
        parts = [self.term()]
        while self.peek() == "+":
            self.take("+")
            parts.append(self.term())
        return parts[0] if len(parts) == 1 else Sum(parts)

    def term(self) -> Kernel:
        parts = [self.factor()]
        while self.peek() == "*":
            self.take("*")
            parts.append(self.factor())
        return parts[0] if len(parts) == 1 else Product(parts)

    def factor(self) -> Kernel:
        t = self.peek()
        if t == "(":
            self.take("(")
            k = self.expr()
            self.take(")")
            return k
        name = self.take().upper()
        if name not in BASE_KERNELS:
            raise KernelParseError(f"unknown kernel {name!r} in {self.src!r}")
        dims: Optional[List[int]] = None
        if self.peek() == "(":
            self.take("(")
            dims = []
            while self.peek() != ")":
                d = self.take()
                if not d.isdigit():
                    raise KernelParseError(f"bad dim {d!r} in {self.src!r}")
                dims.append(int(d))
                if self.peek() == ",":
                    self.take(",")
            self.take(")")
            if len(dims) == 0:
                dims = None
        if dims is not None and self.n_dims is not None:
            bad = [d for d in dims if d >= self.n_dims]
            if bad:
                raise KernelParseError(f"dims {bad} out of range for {self.n_dims}-D input in {self.src!r}")
        return BASE_KERNELS[name](active_dims=dims)


def parse_kernel(spec: str, n_dims: Optional[int] = None) -> Kernel:
    """Parse a kernel expression string into a Kernel object (fresh hyper-parameters)."""
    p = _Parser(spec, n_dims)
    k = p.expr()
    if p.peek() is not None:
        raise KernelParseError(f"trailing tokens {p.toks[p.i:]} in {spec!r}")
    return k


def canonical(spec: str, n_dims: Optional[int] = None) -> str:
    """Round-trip through the parser to get a normalised string (for de-duplication)."""
    return repr(parse_kernel(spec, n_dims))


def ard_spec(base: str, n_dims: int) -> str:
    """Per-dimension product, i.e. an ARD kernel: 'RBF(0)*RBF(1)*...'."""
    return "*".join(f"{base}({d})" for d in range(n_dims))
