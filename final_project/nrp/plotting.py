"""Plotting utilities: convergence curves and best-profit box plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .algorithms import RunResult


def plot_convergence(
    results: Dict[str, List[RunResult]],
    out_path: str | Path,
    title: str = "",
    log_x: bool = False,
):
    """Mean (+/- std) best-so-far profit vs evaluations, one line per algorithm."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, runs in results.items():
        # Pad histories to the longest length (they should already match)
        max_len = max(len(r.history) for r in runs)
        H = np.full((len(runs), max_len), np.nan, dtype=float)
        for i, r in enumerate(runs):
            H[i, : len(r.history)] = r.history
            # Forward-fill with last value if shorter
            if len(r.history) < max_len:
                H[i, len(r.history):] = r.history[-1]
        mean = np.nanmean(H, axis=0)
        std = np.nanstd(H, axis=0)
        x = np.arange(1, max_len + 1)
        ax.plot(x, mean, label=name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Best-so-far profit")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_box(
    results: Dict[str, List[RunResult]],
    out_path: str | Path,
    title: str = "",
):
    """Box plot of best profit across runs for each algorithm."""
    import matplotlib.pyplot as plt

    names = list(results.keys())
    data = [[r.best.profit for r in results[n]] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, labels=names, showmeans=True)
    ax.set_ylabel("Best profit")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mut_trajectory(
    results: Dict[str, List[RunResult]],
    out_path: str | Path,
    title: str = "",
    n_reqs: int | None = None,
):
    """Plot mut_prob trajectory, monitor its change
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for name, runs in results.items():
        for r in runs:
            traj = r.meta.get("mut_trajectory") if r.meta else None
            if not traj:
                continue
            evals = [t[0] for t in traj]
            probs = [t[1] for t in traj]
            if n_reqs is not None:
                probs = [p * n_reqs for p in probs]
            ax.plot(evals, probs, color="C0", alpha=0.25, linewidth=1)
            plotted = True

    if not plotted:
        plt.close(fig)
        return  # no adaptive runs - skip silently

    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Mutation rate (units of 1/n)" if n_reqs else "Mutation rate (per bit)")
    ax.set_yscale("log")
    ax.axhline(1.0 if n_reqs else 1.0 / (n_reqs or 1), color="gray",
               linestyle="--", linewidth=1, label="default 1/n")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)