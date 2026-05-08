"""This is the file that we implement the algorithms to test and modify.
Random, GA, SA, HC, and Adaptive GA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from .problem import NRPProblem


# ---------------------------------------------------------------------------
# Make it standard in form...
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    bits: np.ndarray
    profit: int
    cost: int

    def copy(self) -> "Solution":
        return Solution(self.bits.copy(), self.profit, self.cost)


@dataclass
class RunResult:
    algorithm: str
    best: Solution
    history: List[int]            #Best-so-far profit at each evaluation milestone
    runtime_seconds: float
    n_evaluations: int
    seed: int
    meta: dict = field(default_factory=dict)  # Algorithm-specific extras (e.g. mut_prob trajectory)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _eval(problem: NRPProblem, bits: np.ndarray) -> Solution:
    """Repair to feasibility (if needed) then evaluate."""
    repaired = problem.repair(bits) if problem.total_cost(bits) > problem.budget else bits.astype(bool)
    prof = problem.profit(repaired)
    cost = problem.total_cost(repaired)
    return Solution(repaired, prof, cost)


# ---------------------------------------------------------------------------
# random: baseline
# ---------------------------------------------------------------------------

def random_search(
    problem: NRPProblem,
    max_evals: int,
    rng: np.random.Generator,
    *,
    sample_p: float | None = None,
    seed: int = 0,
) -> RunResult:
    """Sample random feasible bit-strings, keep the best."""
    t0 = time.perf_counter()
    p = sample_p if sample_p is not None else problem.cost_ratio

    best = _eval(problem, rng.random(problem.n_reqs) < p)
    history = [best.profit]

    for _ in range(1, max_evals):
        sol = _eval(problem, rng.random(problem.n_reqs) < p)
        if sol.profit > best.profit:
            best = sol
        history.append(best.profit)

    return RunResult(
        algorithm="RandomSearch",
        best=best,
        history=history,
        runtime_seconds=time.perf_counter() - t0,
        n_evaluations=max_evals,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Hill Climbing
# ---------------------------------------------------------------------------

def hill_climbing(
    problem: NRPProblem,
    max_evals: int,
    rng: np.random.Generator,
    *,
    restart_after: int = 2000,
    seed: int = 0,
) -> RunResult:
    t0 = time.perf_counter()

    current = _eval(problem, rng.random(problem.n_reqs) < problem.cost_ratio)
    best = current.copy()
    history = [best.profit]
    stagnation = 0
    evals = 1

    while evals < max_evals:
        # Random bit-flip
        j = rng.integers(problem.n_reqs)
        candidate_bits = current.bits.copy()
        candidate_bits[j] = not candidate_bits[j]
        candidate = _eval(problem, candidate_bits)
        evals += 1

        if candidate.profit > current.profit:
            current = candidate
            stagnation = 0
            if current.profit > best.profit:
                best = current.copy()
        else:
            stagnation += 1

        history.append(best.profit)

        if stagnation >= restart_after and evals < max_evals:
            current = _eval(problem, rng.random(problem.n_reqs) < problem.cost_ratio)
            evals += 1
            history.append(best.profit)
            if current.profit > best.profit:
                best = current.copy()
                history[-1] = best.profit
            stagnation = 0

    return RunResult(
        algorithm="HillClimbing",
        best=best,
        history=history[:max_evals],
        runtime_seconds=time.perf_counter() - t0,
        n_evaluations=evals,
        seed=seed,
    )


# ---------------------------------------------------------------------------
#Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    problem: NRPProblem,
    max_evals: int,
    rng: np.random.Generator,
    *,
    t0: float | None = None,
    alpha: float = 0.9995,
    t_min: float = 1e-3,
    seed: int = 0,
) -> RunResult:
    timer_start = time.perf_counter()

    current = _eval(problem, rng.random(problem.n_reqs) < problem.cost_ratio)
    best = current.copy()
    history = [best.profit]

    # Auto-tune initial temperature: probe ~50 random bit flips, set T so that
    # the average worsening |delta| is accepted with prob ~0.5.
    if t0 is None:
        deltas = []
        for _ in range(50):
            j = rng.integers(problem.n_reqs)
            b = current.bits.copy()
            b[j] = not b[j]
            cand = _eval(problem, b)
            d = cand.profit - current.profit
            if d < 0:
                deltas.append(-d)
        t0 = float(np.mean(deltas)) / np.log(2) if deltas else 1.0
        t0 = max(t0, 1.0)

    T = t0
    evals = 1  # already evaluated initial

    while evals < max_evals:
        j = rng.integers(problem.n_reqs)
        cand_bits = current.bits.copy()
        cand_bits[j] = not cand_bits[j]
        cand = _eval(problem, cand_bits)
        evals += 1

        delta = cand.profit - current.profit
        if delta >= 0 or rng.random() < np.exp(delta / T):
            current = cand
            if current.profit > best.profit:
                best = current.copy()

        history.append(best.profit)

        T = max(T * alpha, t_min)

    return RunResult(
        algorithm="SimulatedAnnealing",
        best=best,
        history=history[:max_evals],
        runtime_seconds=time.perf_counter() - timer_start,
        n_evaluations=evals,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Adaptive/GA, can select; use np to optimize
# ---------------------------------------------------------------------------

def genetic_algorithm(
    problem: NRPProblem,
    max_evals: int,
    rng: np.random.Generator,
    *,
    pop_size: int = 100,
    cx_prob: float = 0.8,
    mut_prob: float | None = None, 
    tournsize: int = 3,
    elitism: int = 2,
    seed: int = 42,
    # ---- here are our adaptive version... ----
    adaptive_mut: bool = False,
    mut_window: int = 5,         # Adapt every mut_window generations
    mut_target: float = 0.2,     # Target success rate (1/5 rule)
    mut_factor: float = 1.2,     # Multiplicative step
    mut_min_mult: float = 1.0,   # Min mut_prob = mut_min_mult / n_reqs
    mut_max: float = 0.5,        # abs upper bound on per-bit rate
) -> RunResult:
    import random as _stdrandom
    from deap import base, creator, tools

    timer_start = time.perf_counter()
    _stdrandom.seed(seed)
    np_rng = rng

    if mut_prob is None:
        mut_prob = 1.0 / problem.n_reqs
    mut_state = {"prob": float(mut_prob)}
    mut_min = mut_min_mult / problem.n_reqs

    if not hasattr(creator, "_NRP_FitnessMax"):
        creator.create("_NRP_FitnessMax", base.Fitness, weights=(1.0,))
    Fitness = creator._NRP_FitnessMax

    class NRPInd(np.ndarray):
        def __array_finalize__(self, obj):
            pass

    def _attach_fitness(ind: "NRPInd") -> "NRPInd":
        ind.fitness = Fitness()
        return ind

    def make_individual(bits: np.ndarray) -> NRPInd:
        return _attach_fitness(bits.astype(np.int8).view(NRPInd))

    def init_individual() -> NRPInd:
        bits = (np_rng.random(problem.n_reqs) < problem.cost_ratio).astype(np.int8)
        return make_individual(bits)

    def evaluate_ind(ind: NRPInd):
        """Repair-on-eval. Writes repaired bits back into ind (in place)."""
        sol = _eval(problem, ind)
        # In-place writeback: numpy array assignment, ~1us for n=3500
        ind[:] = sol.bits.astype(np.int8)
        return (sol.profit,)

    def cx_uniform_np(ind1: NRPInd, ind2: NRPInd, indpb: float = 0.5):
        """In-place uniform crossover via boolean-mask swap."""
        swap = np_rng.random(len(ind1)) < indpb
        # Single tmp buffer; standard 3-way swap on the masked positions
        tmp = ind1[swap].copy()
        ind1[swap] = ind2[swap]
        ind2[swap] = tmp
        return ind1, ind2

    def mut_flip_np(ind: NRPInd, indpb: float | None = None):
        """In-place per-bit flip mutation. Reads the current rate from
        ``mut_state['prob']`` so self-adaptive updates take effect."""
        rate = mut_state["prob"] if indpb is None else indpb
        flip = np_rng.random(len(ind)) < rate
        if flip.any():
            ind[flip] = 1 - ind[flip]
        return (ind,)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", cx_uniform_np, indpb=0.5)
    toolbox.register("mutate", mut_flip_np)  # rate read from mut_state at call time
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    # ----- evolution loop with explicit eval budget -----
    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    evals = pop_size

    def _best_in_pop():
        i = max(range(len(pop)), key=lambda k: pop[k].fitness.values[0])
        bits = np.asarray(pop[i], dtype=bool)
        return Solution(bits.copy(), int(pop[i].fitness.values[0]),
                        problem.total_cost(bits))

    best = _best_in_pop()
    history = [best.profit] * evals  # backfill so len(history) == evals

    # ---- self-adaptation bookkeeping ----
    from collections import deque
    success_window = deque(maxlen=mut_window)
    mut_trajectory: list[tuple[int, float]] = [(evals, mut_state["prob"])]

    while evals < max_evals:
        prev_best = best.profit  # for success-rate tracking this generation

        # Elitism: top-K, single selection call then clone
        elite_src = tools.selBest(pop, elitism)
        elites = []
        for e in elite_src:
            clone = _attach_fitness(e.copy())
            clone.fitness.values = e.fitness.values
            elites.append(clone)

        # Parents: tournament selection, single call then clone
        parent_src = toolbox.select(pop, pop_size - elitism)
        offspring = []
        for o in parent_src:
            clone = _attach_fitness(o.copy())
            clone.fitness.values = o.fitness.values
            offspring.append(clone)

        # Variation
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if _stdrandom.random() < cx_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for m in offspring:
            toolbox.mutate(m)
            del m.fitness.values

        # Evaluate offspring respecting the eval budget
        for ind in offspring:
            if ind.fitness.valid:
                continue
            if evals >= max_evals:
                ind.fitness.values = (0,)
                continue
            ind.fitness.values = toolbox.evaluate(ind)
            evals += 1
            f = ind.fitness.values[0]
            if f > best.profit:
                bits = np.asarray(ind, dtype=bool)
                best = Solution(bits.copy(), int(f), problem.total_cost(bits))
            history.append(best.profit)

        pop[:] = elites + offspring

        #Self-adaptive mutation rate update (1/5 success rule)
        if adaptive_mut:
            success_window.append(1 if best.profit > prev_best else 0)
            if len(success_window) == mut_window:
                success_rate = sum(success_window) / mut_window
                if success_rate > mut_target:
                    mut_state["prob"] = min(mut_state["prob"] * mut_factor, mut_max)
                elif success_rate < mut_target:
                    mut_state["prob"] = max(mut_state["prob"] / mut_factor, mut_min)
                # ties (success_rate == mut_target): no change
                mut_trajectory.append((evals, mut_state["prob"]))

    return RunResult(
        algorithm="AdaptiveGA" if adaptive_mut else "GeneticAlgorithm",
        best=best,
        history=history[:max_evals],
        runtime_seconds=time.perf_counter() - timer_start,
        n_evaluations=evals,
        seed=seed,
        meta={
            "mut_prob_initial": float(mut_prob),
            "mut_prob_final": float(mut_state["prob"]),
            "mut_trajectory": mut_trajectory if adaptive_mut else [],
            "adaptive_mut": adaptive_mut,
        },
    )


def adaptive_genetic_algorithm(
    problem: NRPProblem,
    max_evals: int,
    rng: np.random.Generator,
    *,
    seed: int = 0,
    **kwargs,
) -> RunResult:
    """GA with self-adaptive mutation rate (1/5 success rule).

    Convenience wrapper: thin alias for ``genetic_algorithm(..., adaptive_mut=True)``
    so it can be plugged into the algorithm registry.
    """
    kwargs.setdefault("adaptive_mut", True)
    return genetic_algorithm(problem, max_evals=max_evals, rng=rng, seed=seed, **kwargs)


# register the algos
ALGORITHMS: dict[str, Callable] = {
    "random": random_search,
    "hc": hill_climbing,
    "sa": simulated_annealing,
    "ga": genetic_algorithm,
    "aga": adaptive_genetic_algorithm,
}