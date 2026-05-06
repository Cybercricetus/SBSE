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
