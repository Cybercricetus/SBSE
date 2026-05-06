"""Search algorithms for the single-objective NRP.

All algorithms share the signature:

    def algorithm(problem, max_evals, rng, **kwargs) -> RunResult

and use the same evaluation budget so cross-algorithm comparison is fair.
Constraints are handled via repair (high-cost-first removal) inside
``problem.repair``; this keeps the comparison clean of penalty tuning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from .problem import NRPProblem


# ---------------------------------------------------------------------------
# Common dataclasses
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
    history: List[int]            # best-so-far profit at each evaluation milestone
    runtime_seconds: float
    n_evaluations: int
    seed: int


# ---------------------------------------------------------------------------
# Helper: evaluate a bitstring with repair, return Solution
# ---------------------------------------------------------------------------

def _eval(problem: NRPProblem, bits: np.ndarray) -> Solution:
    """Repair to feasibility (if needed) then evaluate."""
    repaired = problem.repair(bits) if problem.total_cost(bits) > problem.budget else bits.astype(bool)
    prof = problem.profit(repaired)
    cost = problem.total_cost(repaired)
    return Solution(repaired, prof, cost)


# ---------------------------------------------------------------------------
# 1) Random Search baseline
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
# 2) (Stochastic) Hill Climbing with random restarts
# ---------------------------------------------------------------------------

def hill_climbing(
    problem: NRPProblem,
    max_evals: int,
    rng: np.random.Generator,
    *,
    restart_after: int = 2000,
    seed: int = 0,
) -> RunResult:
    """Stochastic 1-bit-flip hill climbing with random restart on stagnation.

    Each iteration: flip one random bit, accept if profit improves (or equals
    with a probability tiebreaker disabled here). Restart from a random
    solution after ``restart_after`` consecutive non-improving evaluations.
    """
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
# 3) Simulated Annealing
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
    """Boltzmann-acceptance SA with geometric cooling.

    If ``t0`` is None, it is auto-tuned so the initial acceptance probability
    of a typical worsening move is ~0.5 (Kirkpatrick-style).
    """
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
# 4) Genetic Algorithm (DEAP)
# ---------------------------------------------------------------------------

def genetic_algorithm(
    problem: NRPProblem,
    max_evals: int,
    rng: np.random.Generator,
    *,
    pop_size: int = 100,
    cx_prob: float = 0.8,
    mut_prob: float | None = None,    # per-bit; default 1/n
    tournsize: int = 3,
    elitism: int = 2,
    seed: int = 0,
) -> RunResult:
    """Generational GA implemented on top of DEAP.

    Operators:
      - selection   : tournament (size 3) + elitism
      - crossover   : uniform (per-bit swap with prob 0.5)
      - mutation    : per-bit flip with rate 1/n
      - constraints : repair-on-evaluation
    """
    # DEAP's RNG is the stdlib `random`; seed it for reproducibility.
    import random as _stdrandom
    from deap import base, creator, tools

    timer_start = time.perf_counter()
    _stdrandom.seed(seed)
    np_rng = rng  # for our own random ops

    if mut_prob is None:
        mut_prob = 1.0 / problem.n_reqs

    # creator classes are global in DEAP - guard against duplicate creation
    if not hasattr(creator, "_NRP_FitnessMax"):
        creator.create("_NRP_FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("_NRP_Individual", list, fitness=creator._NRP_FitnessMax)

    Individual = creator._NRP_Individual

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", _stdrandom.random)  # placeholder

    def init_individual():
        # Bias toward cost_ratio density to keep early pop near budget
        bits = (np_rng.random(problem.n_reqs) < problem.cost_ratio).astype(np.int8)
        return Individual(bits.tolist())

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_ind(ind):
        bits = np.asarray(ind, dtype=bool)
        sol = _eval(problem, bits)
        # write back the (possibly repaired) genotype so children are valid
        ind[:] = sol.bits.astype(np.int8).tolist()
        return (sol.profit,)

    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mut_prob)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    # ----- evolution loop with explicit eval budget -----
    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    evals = pop_size

    # Track best
    def _best(pop):
        i = max(range(len(pop)), key=lambda k: pop[k].fitness.values[0])
        bits = np.asarray(pop[i], dtype=bool)
        return Solution(bits, int(pop[i].fitness.values[0]), problem.total_cost(bits))

    best = _best(pop)
    history = [best.profit] * evals  # backfill so length equals evals

    while evals < max_evals:
        # Elitism: carry over top-K
        elites = tools.selBest(pop, elitism)
        elites = [Individual(e[:]) for e in elites]
        for e, src in zip(elites, tools.selBest(pop, elitism)):
            e.fitness.values = src.fitness.values

        # Parents -> offspring
        offspring = toolbox.select(pop, pop_size - elitism)
        offspring = [Individual(o[:]) for o in offspring]
        for o, src in zip(offspring, toolbox.select(pop, pop_size - elitism)):
            o.fitness.values = src.fitness.values

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
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            if evals >= max_evals:
                # Use parent fitness as fallback (assigned an arbitrary low value
                # only if it was deleted - ensure something exists)
                ind.fitness.values = (0,)
                continue
            ind.fitness.values = toolbox.evaluate(ind)
            evals += 1
            # update best & history each eval to keep parity with HC/SA traces
            f = ind.fitness.values[0]
            if f > best.profit:
                bits = np.asarray(ind, dtype=bool)
                best = Solution(bits, int(f), problem.total_cost(bits))
            history.append(best.profit)

        pop[:] = elites + offspring

    return RunResult(
        algorithm="GeneticAlgorithm",
        best=best,
        history=history[:max_evals],
        runtime_seconds=time.perf_counter() - timer_start,
        n_evaluations=evals,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Registry for CLI / experiment driver
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, Callable] = {
    "random": random_search,
    "hc": hill_climbing,
    "sa": simulated_annealing,
    "ga": genetic_algorithm,
}
