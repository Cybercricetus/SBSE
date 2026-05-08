"""Experiments; using Xuan's data...
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



def run_experiment(
    problem: NRPProblem,
    algorithms: Dict[str, Callable],
    n_runs: int = 30,
    max_evals: int = 50_000,
    base_seed: int = 0,
    verbose: bool = True,
    algo_kwargs: Dict[str, dict] | None = None,
) -> Dict[str, List[RunResult]]:
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
#Stats
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
#Wilcoxon signed-rank tests
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


# ---------------------------------------------------------------------------
#SAVE to disk
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