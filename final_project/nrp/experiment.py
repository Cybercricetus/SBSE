"""Experiment harness: run each algorithm multiple times, aggregate, run
Wilcoxon signed-rank tests vs the random-search baseline.

Output is JSON-serialisable so we can dump per-run history for plotting later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

from .algorithms import RunResult
from .problem import NRPProblem


# ---------------------------------------------------------------------------
# Multi-run experiment
# ---------------------------------------------------------------------------

def run_experiment(
    problem: NRPProblem,
    algorithms: Dict[str, Callable],
    n_runs: int = 30,
    max_evals: int = 50_000,
    base_seed: int = 0,
    verbose: bool = True,
    algo_kwargs: Dict[str, dict] | None = None,
) -> Dict[str, List[RunResult]]:
    """Run every algorithm `n_runs` times with paired seeds.

    Paired seeds (same seed for all algorithms in run-i) make the Wilcoxon
    signed-rank test more powerful, since shared randomness in the random
    initial solution removes some variance.
    """
    algo_kwargs = algo_kwargs or {}
    results: Dict[str, List[RunResult]] = {name: [] for name in algorithms}

    for run in range(n_runs):
        seed = base_seed + run
        for name, algo in algorithms.items():
            rng = np.random.default_rng(seed)
            kwargs = dict(algo_kwargs.get(name, {}))
            kwargs.setdefault("seed", seed)
            res = algo(problem, max_evals=max_evals, rng=rng, **kwargs)
            results[name].append(res)
            if verbose:
                print(
                    f"  [run {run + 1}/{n_runs}] {name:>20s}: "
                    f"profit={res.best.profit:>8d}  "
                    f"cost={res.best.cost:>8d}/{problem.budget}  "
                    f"time={res.runtime_seconds:.2f}s"
                )

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, List[RunResult]]) -> Dict[str, dict]:
    """Per-algorithm summary: mean / std / median / min / max of best profit."""
    summary = {}
    for name, runs in results.items():
        profits = np.array([r.best.profit for r in runs], dtype=float)
        runtimes = np.array([r.runtime_seconds for r in runs], dtype=float)
        summary[name] = {
            "n_runs": len(runs),
            "profit_mean": float(profits.mean()),
            "profit_std": float(profits.std(ddof=1)) if len(profits) > 1 else 0.0,
            "profit_median": float(np.median(profits)),
            "profit_min": int(profits.min()),
            "profit_max": int(profits.max()),
            "runtime_mean": float(runtimes.mean()),
        }
    return summary


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank tests
# ---------------------------------------------------------------------------

def wilcoxon_vs_baseline(
    results: Dict[str, List[RunResult]],
    baseline: str = "random",
    alternative: str = "greater",
) -> Dict[str, dict]:
    """Paired Wilcoxon signed-rank test of every algorithm vs the baseline.

    Each algorithm's i-th run shares its base seed with the baseline's i-th run,
    so we can compare paired (best_profit) values.
    """
    if baseline not in results:
        raise KeyError(f"baseline '{baseline}' not in results")

    base_profits = np.array([r.best.profit for r in results[baseline]], dtype=float)
    out = {}
    for name, runs in results.items():
        if name == baseline:
            continue
        algo_profits = np.array([r.best.profit for r in runs], dtype=float)
        if len(algo_profits) != len(base_profits):
            raise ValueError("paired test requires equal number of runs")
        diffs = algo_profits - base_profits
        if np.all(diffs == 0):
            stat, p = float("nan"), 1.0
        else:
            res = wilcoxon(algo_profits, base_profits, alternative=alternative,
                           zero_method="wilcox")
            stat, p = float(res.statistic), float(res.pvalue)
        out[name] = {
            "vs": baseline,
            "alternative": alternative,
            "statistic": stat,
            "p_value": p,
            "median_diff": float(np.median(diffs)),
            "wins": int((diffs > 0).sum()),
            "losses": int((diffs < 0).sum()),
            "ties": int((diffs == 0).sum()),
        }
    return out


def vargha_delaney_a12(
    results: Dict[str, List[RunResult]],
    baseline: str = "random",
) -> Dict[str, dict]:
    if baseline not in results:
        raise KeyError(f"baseline '{baseline}' not in results")

    base_profits = np.array([r.best.profit for r in results[baseline]], dtype=float)
    out = {}
    for name, runs in results.items():
        if name == baseline:
            continue
        algo_profits = np.array([r.best.profit for r in runs], dtype=float)
        a12 = _a12(algo_profits, base_profits)
        out[name] = {
            "vs": baseline,
            "a12": float(a12),
            "magnitude": _a12_magnitude(a12),
        }
    return out


def _a12(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Vargha-Delaney A12 via pairwise comparison.

    A12 = (#{x > y} + 0.5 * #{x == y}) / (|x| * |y|)
    """
    # Broadcasting gives an (|x|, |y|) comparison matrix
    diff = x[:, None] - y[None, :]
    greater = (diff > 0).sum()
    ties = (diff == 0).sum()
    return (greater + 0.5 * ties) / (len(x) * len(y))


def _a12_magnitude(a12: float) -> str:
    """Qualitative effect-size label per Vargha & Delaney (2000)."""
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
#Persistence storage
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, List[RunResult]], path: str | Path) -> None:
    """Dump per-run records (excluding bit-strings) as JSON."""
    payload = {
        name: [
            {
                "algorithm": r.algorithm,
                "seed": r.seed,
                "n_evaluations": r.n_evaluations,
                "runtime_seconds": r.runtime_seconds,
                "best_profit": r.best.profit,
                "best_cost": r.best.cost,
                "history": r.history,
                "meta": r.meta,
            }
            for r in runs
        ]
        for name, runs in results.items()
    }
    Path(path).write_text(json.dumps(payload))