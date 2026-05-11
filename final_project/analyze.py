"""Re-analyze already-saved experiment results without re-running anything.

Reads every results/*.json produced by main.py (which contains per-run
``best_profit`` lists) and prints, for each instance:
  - Per-algorithm mean/std/median
  - Wilcoxon signed-rank vs RandomSearch
  - Vargha-Delaney A12 effect size vs RandomSearch (and optionally vs SA)

Usage:
    python analyze.py results/*.json
    python analyze.py results/e1_r03.json --baseline sa
    python analyze.py results/*.json --csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import wilcoxon

from nrp.experiment import _a12, _a12_magnitude


def load_results(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load a results JSON written by save_results."""
    return json.loads(path.read_text())


def analyze_instance(
    payload: Dict[str, List[Dict[str, Any]]],
    baseline: str = "random",
    alternative: str = "greater",
) -> Dict[str, dict]:
    """Compute summary, Wilcoxon, and A12 for one instance JSON.

    Returns a dict keyed by algorithm name with all stats nested per algorithm.
    """
    profits = {name: np.array([r["best_profit"] for r in runs], dtype=float)
               for name, runs in payload.items()}
    if baseline not in profits:
        raise KeyError(f"baseline '{baseline}' not in payload (have {list(profits)})")

    base = profits[baseline]
    out: Dict[str, dict] = {}
    for name, vals in profits.items():
        row = {
            "n_runs": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
        if name != baseline:
            diffs = vals - base
            if np.all(diffs == 0):
                stat, p = float("nan"), 1.0
            else:
                w = wilcoxon(vals, base, alternative=alternative, zero_method="wilcox")
                stat, p = float(w.statistic), float(w.pvalue)
            a12 = _a12(vals, base)
            row.update({
                "wilcoxon_W": stat,
                "wilcoxon_p": p,
                "a12": float(a12),
                "a12_magnitude": _a12_magnitude(a12),
                "wins_vs_baseline": int((diffs > 0).sum()),
                "losses_vs_baseline": int((diffs < 0).sum()),
                "ties_vs_baseline": int((diffs == 0).sum()),
            })
        out[name] = row
    return out


def print_instance(name: str, stats: Dict[str, dict], baseline: str) -> None:
    """Pretty-print one instance's stats to stdout."""
    print(f"\n{'=' * 70}")
    print(f"Instance: {name}    baseline = {baseline}")
    print(f"{'=' * 70}")

    # Per-algorithm summary
    hdr = (f"{'algorithm':<10} {'mean':>10} {'std':>8} {'median':>10} "
           f"{'min':>8} {'max':>8}")
    print(hdr)
    for algo, s in stats.items():
        print(f"{algo:<10} {s['mean']:>10.1f} {s['std']:>8.1f} "
              f"{s['median']:>10.1f} {s['min']:>8.0f} {s['max']:>8.0f}")

    # Significance + effect size
    print(f"\n  vs {baseline}:")
    for algo, s in stats.items():
        if algo == baseline:
            continue
        sig = "*" if s["wilcoxon_p"] < 0.05 else " "
        print(f"    {algo:<8} p={s['wilcoxon_p']:.4g} {sig}  "
              f"A12={s['a12']:.3f} ({s['a12_magnitude']:>10s})  "
              f"W/L/T={s['wins_vs_baseline']}/{s['losses_vs_baseline']}/{s['ties_vs_baseline']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="results/*.json files to analyze")
    p.add_argument("--baseline", default="random",
                   help="Algorithm to compare everything against (default: random)")
    p.add_argument("--alternative", default="greater",
                   choices=["greater", "less", "two-sided"],
                   help="Wilcoxon alternative hypothesis")
    p.add_argument("--csv", default=None,
                   help="If given, also write a tidy CSV summary to this path")
    args = p.parse_args()

    all_rows = []
    for path_str in args.files:
        path = Path(path_str)
        payload = load_results(path)
        try:
            stats = analyze_instance(payload, baseline=args.baseline,
                                     alternative=args.alternative)
        except KeyError as e:
            print(f"[skip] {path.name}: {e}")
            continue
        print_instance(path.stem, stats, args.baseline)

        if args.csv:
            for algo, s in stats.items():
                row = {"instance": path.stem, "algorithm": algo, **s}
                all_rows.append(row)

    if args.csv and all_rows:
        # Union of all keys across rows (some rows lack wilcoxon for baseline)
        keys = []
        seen = set()
        for r in all_rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote tidy CSV to {args.csv}")


if __name__ == "__main__":
    main()
