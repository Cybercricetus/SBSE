"""CLI entry point for Track A experiments.

Examples
--------
# Smoke test on the small example file
python main.py --instance data/example.txt --runs 3 --evals 5000

# Full run on classic instance
python main.py --instance data/nrp-e1.txt --runs 30 --evals 50000 \\
    --cost-ratio 0.3 --output results/e1_r03.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nrp import (
    parse_nrp_file,
    random_search,
    hill_climbing,
    simulated_annealing,
    genetic_algorithm,
    run_experiment,
    summarize,
    wilcoxon_vs_baseline,
)
from nrp.experiment import save_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", type=str, required=True,
                   help="Path to NRP instance file (e.g. data/nrp-e1.txt)")
    p.add_argument("--cost-ratio", type=float, default=0.3,
                   help="Budget ratio in (0,1). 0.3 or 0.5 typical (0.7 trivial).")
    p.add_argument("--runs", type=int, default=30,
                   help="Independent runs per algorithm.")
    p.add_argument("--evals", type=int, default=50_000,
                   help="Fitness-evaluation budget per run.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; run-i uses seed = base + i.")
    p.add_argument("--algorithms", nargs="+",
                   default=["random", "hc", "sa", "ga"],
                   choices=["random", "hc", "sa", "ga"],
                   help="Subset of algorithms to run.")
    p.add_argument("--output", type=str, default=None,
                   help="JSON path to dump full per-run results (history etc.)")
    p.add_argument("--plot", type=str, default=None,
                   help="Path stem for saving convergence + box plots (e.g. results/e1_r03)")
    return p.parse_args()


ALGO_REGISTRY = {
    "random": random_search,
    "hc": hill_climbing,
    "sa": simulated_annealing,
    "ga": genetic_algorithm,
}


def main():
    args = parse_args()

    problem = parse_nrp_file(args.instance, cost_ratio=args.cost_ratio)
    print(problem.info())
    print()

    algos = {name: ALGO_REGISTRY[name] for name in args.algorithms}
    print(f"Running {len(algos)} algorithms x {args.runs} runs x {args.evals} evals each\n")

    results = run_experiment(
        problem=problem,
        algorithms=algos,
        n_runs=args.runs,
        max_evals=args.evals,
        base_seed=args.seed,
    )

    print("\n=== Summary ===")
    summary = summarize(results)
    name_w = max(len(n) for n in summary)
    print(f"{'algorithm':<{name_w}}  {'mean':>10}  {'std':>8}  {'median':>10}  {'min':>8}  {'max':>8}  {'time(s)':>8}")
    for name, s in summary.items():
        print(f"{name:<{name_w}}  {s['profit_mean']:>10.1f}  {s['profit_std']:>8.1f}  "
              f"{s['profit_median']:>10.1f}  {s['profit_min']:>8d}  {s['profit_max']:>8d}  "
              f"{s['runtime_mean']:>8.2f}")

    if "random" in results and len(args.algorithms) > 1:
        print("\n=== Wilcoxon signed-rank vs RandomSearch (alt='greater') ===")
        tests = wilcoxon_vs_baseline(results, baseline="random", alternative="greater")
        for name, t in tests.items():
            sig = "*" if t["p_value"] < 0.05 else " "
            print(f"  {name:>4}  W={t['statistic']:>10.1f}  "
                  f"p={t['p_value']:.4g} {sig}  "
                  f"wins/losses/ties = {t['wins']}/{t['losses']}/{t['ties']}  "
                  f"median diff = {t['median_diff']:+.1f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, out)
        print(f"\nSaved per-run results to {out}")

    if args.plot:
        from nrp.plotting import plot_convergence, plot_box
        stem = Path(args.plot)
        stem.parent.mkdir(parents=True, exist_ok=True)
        title = f"{problem.instance_name}  ratio={problem.cost_ratio}  runs={args.runs}  evals={args.evals}"
        plot_convergence(results, stem.with_name(stem.stem + "_convergence.png"), title=title)
        plot_box(results, stem.with_name(stem.stem + "_box.png"), title=title)
        print(f"Saved plots: {stem.stem}_convergence.png, {stem.stem}_box.png")


if __name__ == "__main__":
    main()