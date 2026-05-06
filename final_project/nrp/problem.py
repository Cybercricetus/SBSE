"""NRP file parsing and problem representation.

File format (Xuan et al. 2012 NRP benchmark):
    1                            # header constant
    n_reqs                       # number of requirements
    c_1 c_2 ... c_n              # cost of each requirement (whitespace separated)
    0                            # header constant
    n_customers                  # number of customers
    p_i q_i r_1 r_2 ... r_qi     # profit, num requested, requirement IDs (1-based)
    ...

Reference: J. Xuan et al., "Solving the Large Scale Next Release Problem
with a Backbone Based Multilevel Algorithm", IEEE TSE 38(5), 2012.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _stream_tokens(path: Path):
    """Yield whitespace-separated tokens from a file lazily.

    Lines starting with '^' (or with a leading '^' token) are treated as
    comments and skipped. This is needed for ``example.txt`` which annotates
    each section; the real ``nrp-*.txt`` instances do not contain comments.
    """
    with open(path, "r") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("^"):
                continue
            for tok in line.split():
                yield tok


def parse_nrp_file(path: str | Path, cost_ratio: float = 0.3) -> "NRPProblem":
    """Parse an NRP instance file.

    Parameters
    ----------
    path : str or Path
        Path to the .txt instance file.
    cost_ratio : float
        Budget ratio in (0, 1). Budget = cost_ratio * total_cost.
        Typical values: 0.3 or 0.5 (0.7 yields trivial instances).
    """
    path = Path(path)
    tokens = _stream_tokens(path)

    # Header "1"
    header_one = next(tokens)
    if header_one != "1":
        raise ValueError(f"Expected first token '1', got '{header_one}' in {path}")

    # Number of requirements (single level)
    n_reqs = int(next(tokens))

    costs = np.fromiter(
        (int(next(tokens)) for _ in range(n_reqs)),
        dtype=np.int64,
        count=n_reqs,
    )

    # Header "0"
    header_zero = next(tokens)
    if header_zero != "0":
        raise ValueError(f"Expected '0' header before customers, got '{header_zero}'")

    n_customers = int(next(tokens))

    profits = np.empty(n_customers, dtype=np.int64)
    customer_reqs: List[np.ndarray] = []

    for i in range(n_customers):
        profits[i] = int(next(tokens))
        q_i = int(next(tokens))
        # Convert 1-based -> 0-based; deduplicate (some instances repeat IDs)
        reqs = np.fromiter(
            (int(next(tokens)) - 1 for _ in range(q_i)),
            dtype=np.int64,
            count=q_i,
        )
        customer_reqs.append(np.unique(reqs))

    # Sanity: any leftover tokens?
    leftover = list(tokens)
    if leftover:
        raise ValueError(
            f"Unexpected trailing tokens in {path}: {len(leftover)} remaining"
        )

    return NRPProblem.build(
        costs=costs,
        profits=profits,
        customer_reqs=customer_reqs,
        cost_ratio=cost_ratio,
        instance_name=path.stem,
    )


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

@dataclass
class NRPProblem:
    """Single-level Next Release Problem.

    Decision variable: x in {0,1}^n where n = number of requirements.
    Maximize total profit from satisfied customers under a budget constraint.
    A customer is "satisfied" iff every requirement they request is selected.
    """

    instance_name: str
    costs: np.ndarray              # shape (n_reqs,) int64
    profits: np.ndarray            # shape (n_customers,) int64
    customer_reqs: List[np.ndarray]  # 0-based per-customer requirement lists
    cost_ratio: float
    budget: int

    # Precomputed for fast vectorized evaluation
    M: csr_matrix = field(repr=False)         # (n_customers, n_reqs) incidence
    requested_count: np.ndarray = field(repr=False)  # (n_customers,) row sums

    # For repair heuristic
    sort_by_cost_desc: np.ndarray = field(repr=False)  # indices sorted high->low cost

    @classmethod
    def build(
        cls,
        costs: np.ndarray,
        profits: np.ndarray,
        customer_reqs: List[np.ndarray],
        cost_ratio: float,
        instance_name: str = "nrp",
    ) -> "NRPProblem":
        n_reqs = len(costs)
        n_cust = len(profits)

        # Build sparse incidence matrix: M[i, j] = 1 iff customer i requests req j
        rows = np.concatenate([np.full(len(r), i) for i, r in enumerate(customer_reqs)])
        cols = np.concatenate(customer_reqs) if customer_reqs else np.array([], dtype=np.int64)
        data = np.ones(len(rows), dtype=np.int8)
        M = csr_matrix((data, (rows, cols)), shape=(n_cust, n_reqs))

        requested_count = np.asarray(M.sum(axis=1)).ravel().astype(np.int64)
        budget = int(cost_ratio * costs.sum())

        return cls(
            instance_name=instance_name,
            costs=costs,
            profits=profits,
            customer_reqs=customer_reqs,
            cost_ratio=cost_ratio,
            budget=budget,
            M=M,
            requested_count=requested_count,
            sort_by_cost_desc=np.argsort(-costs, kind="stable"),
        )

    @property
    def n_reqs(self) -> int:
        return len(self.costs)

    @property
    def n_customers(self) -> int:
        return len(self.profits)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def total_cost(self, bits: np.ndarray) -> int:
        return int(self.costs @ bits.astype(np.int64))

    def profit(self, bits: np.ndarray) -> int:
        """Binary all-or-nothing profit: customer satisfied iff all reqs selected."""
        bits_i = bits.astype(np.int64)
        selected_per_customer = self.M @ bits_i  # (n_customers,)
        satisfied = selected_per_customer >= self.requested_count
        return int(self.profits[satisfied].sum())

    def evaluate(self, bits: np.ndarray) -> Tuple[int, int, bool]:
        """Return (profit, cost, feasible). Profit is computed as-is, even if infeasible."""
        cost = self.total_cost(bits)
        prof = self.profit(bits)
        return prof, cost, cost <= self.budget

    # ------------------------------------------------------------------
    # Repair heuristic - greedy removal until feasible
    # ------------------------------------------------------------------

    def repair(self, bits: np.ndarray) -> np.ndarray:
        """Remove selected requirements (highest cost first) until cost <= budget.

        Modifies a *copy* of bits and returns the repaired array. We deliberately
        use a simple, deterministic heuristic so all algorithms see the same
        repair behavior - keeps the comparison fair.
        """
        bits = bits.copy().astype(bool)
        cost = self.total_cost(bits)
        if cost <= self.budget:
            return bits

        # Walk through requirements in descending cost order; drop selected ones
        # until we are within budget. This is O(n) and good enough for repair.
        for j in self.sort_by_cost_desc:
            if cost <= self.budget:
                break
            if bits[j]:
                bits[j] = False
                cost -= int(self.costs[j])
        return bits

    # ------------------------------------------------------------------
    # Random feasible solution
    # ------------------------------------------------------------------

    def random_solution(self, rng: np.random.Generator, p: float | None = None) -> np.ndarray:
        """Random bit-string, biased near the budget. Always returned feasible."""
        if p is None:
            p = self.cost_ratio  # natural prior - sum of costs ~ ratio of total
        bits = rng.random(self.n_reqs) < p
        return self.repair(bits)

    # ------------------------------------------------------------------
    # Pretty info
    # ------------------------------------------------------------------

    def info(self) -> str:
        return (
            f"NRP[{self.instance_name}] "
            f"reqs={self.n_reqs}, customers={self.n_customers}, "
            f"total_cost={int(self.costs.sum())}, "
            f"budget={self.budget} (ratio={self.cost_ratio}), "
            f"max_possible_profit={int(self.profits.sum())}"
        )
