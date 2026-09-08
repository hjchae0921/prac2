#!/usr/bin/env python3
"""Pilot study: plain GP-BO vs. kernel-search vs. LLM-proposed kernels.

Examples
  python run_experiment.py                                   # offline (mock LLM), all problems
  python run_experiment.py --strategies fixed_rbf llm --problems sinlin2d --seeds 3   # real Claude
  python run_experiment.py --dry-run-prompt sinlin2d         # print the prompt the LLM would get
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from llmbo import BayesOpt, PROBLEMS, make_strategy
from llmbo.problems import get_problem

ALL_STRATEGIES = ["fixed_rbf", "fixed_mat52", "greedy", "mock_llm", "llm"]


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pname in args.problems:
        problem = get_problem(pname)
        print(f"\n=== {pname} (d={problem.dim}, f*={problem.f_min:.4f}) ===")
        for sname in args.strategies:
            for seed in range(args.seeds):
                strat = make_strategy(sname, refresh=args.refresh, model=args.model, verbose=args.verbose)
                bo = BayesOpt(problem, strat, n_init=args.n_init, n_iter=args.iters, seed=seed,
                              n_restarts=args.restarts, verbose=args.verbose)
                r = bo.run()
                results.append({
                    "problem": pname, "strategy": sname, "seed": seed,
                    "regret": r.regret.tolist(), "best_y": r.best_y.tolist(),
                    "kernel_trace": r.kernel_trace, "wall_time": r.wall_time,
                })
                last_k = r.kernel_trace[-1]["spec"] if r.kernel_trace else "-"
                print(f"  {sname:12s} seed={seed}  final regret={r.regret[-1]:.4g}  "
                      f"({r.wall_time:5.1f}s)  kernel={last_k}")
                if sname == "llm" and getattr(strat, "transcript", None):
                    with open(out_dir / "llm_transcript.jsonl", "a") as f:
                        for t in strat.transcript:
                            f.write(json.dumps({"seed": seed, **t}, ensure_ascii=False) + "\n")
    path = out_dir / f"results_{args.tag}.json"
    path.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    print(f"\nsaved {path}")
    return path


def summarise(path: Path):
    data = json.loads(Path(path).read_text())
    res = data["results"]
    problems = sorted({r["problem"] for r in res}, key=list(PROBLEMS).index)
    strategies = [s for s in ALL_STRATEGIES if any(r["strategy"] == s for r in res)]
    print("\nFinal simple regret, median [IQR] over seeds  (lower is better)")
    print(f"{'problem':12s}" + "".join(f"{s:>24s}" for s in strategies))
    for p in problems:
        row = f"{p:12s}"
        for s in strategies:
            fr = np.array([r["regret"][-1] for r in res if r["problem"] == p and r["strategy"] == s])
            if len(fr) == 0:
                row += f"{'-':>24s}"
            else:
                q = np.percentile(fr, [25, 50, 75])
                row += f"{q[1]:12.3g} [{q[0]:.2g},{q[2]:.2g}]".rjust(24)
        print(row)
    return res, problems, strategies


def plot(path: Path, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, problems, strategies = summarise(path)
    n = len(problems)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)
    colors = {"fixed_rbf": "#7f7f7f", "fixed_mat52": "#bcbd22", "greedy": "#1f77b4",
              "mock_llm": "#d62728", "llm": "#9467bd"}
    for ax, p in zip(axes.ravel(), problems):
        for s in strategies:
            R = np.array([r["regret"] for r in res if r["problem"] == p and r["strategy"] == s])
            if len(R) == 0:
                continue
            R = np.maximum(R, 1e-8)
            med = np.median(R, axis=0)
            lo, hi = np.percentile(R, [25, 75], axis=0)
            it = np.arange(len(med))
            ax.plot(it, med, label=s, color=colors.get(s))
            ax.fill_between(it, lo, hi, alpha=0.15, color=colors.get(s))
        ax.set_yscale("log")
        ax.set_title(p)
        ax.set_xlabel("evaluations")
        ax.set_ylabel("simple regret")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--strategies", nargs="+", default=["fixed_rbf", "fixed_mat52", "greedy", "mock_llm"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--refresh", type=int, default=10, help="re-propose kernels every k iterations")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--model", default=None, help="Claude model id for --strategies llm")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plot-only", default=None, help="path to a results json; only summarise + plot")
    ap.add_argument("--dry-run-prompt", default=None, metavar="PROBLEM",
                    help="print the prompt the LLM would receive after the initial design and exit")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.dry_run_prompt:
        from scipy.stats import qmc
        from llmbo.llm_client import SYSTEM_PROMPT, build_prompt
        prob = get_problem(args.dry_run_prompt)
        X = qmc.LatinHypercube(d=prob.dim, seed=0).random(args.n_init)
        y = np.array([prob(x) for x in X])
        print("----- SYSTEM -----\n" + SYSTEM_PROMPT + "\n----- USER -----\n" + build_prompt(prob, X, y, 0, [], None))
        return

    if args.plot_only:
        path = Path(args.plot_only)
    else:
        if "llm" in args.strategies and not os.environ.get("ANTHROPIC_API_KEY"):
            print("warning: 'llm' strategy requested but ANTHROPIC_API_KEY is not set; "
                  "the SDK will try other credential sources.", file=sys.stderr)
        args.tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
        path = run(args)
    plot(path, path.with_suffix(".png"))


if __name__ == "__main__":
    main()
