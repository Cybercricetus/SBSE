"""
=============================================================================
 HOME ASSIGNMENT: Genetic Algorithm for Software Developer Assignment
=============================================================================

COURSE:   Search-Based Software Engineering
TOPIC:    Genetic Algorithms — Software Resource Management
DUE:      [Insert date]
POINTS:   100

=============================================================================
 SCENARIO
=============================================================================

You are the engineering manager at a software company. You have 8 DEVELOPERS
and 12 TASKS that must be completed in the current sprint. Each developer has
different skill levels across 4 domains:

    Frontend, Backend, Database, Testing

Each task requires a PRIMARY skill and takes a certain number of hours.
Developers have limited capacity (hours available this sprint).

YOUR GOAL: Assign each task to exactly one developer to:
    1. MAXIMIZE skill match   (assign tasks to developers skilled in that area)
    2. MINIMIZE overload      (don't exceed any developer's available hours)
    3. MAXIMIZE balance       (spread work evenly, avoid idle developers)

This is an NP-hard assignment problem. With 8 developers and 12 tasks,
the search space is 8^12 = 68,719,476,736 possible assignments.

=============================================================================
 YOUR TASKS (100 points total)
=============================================================================

PART A — Understanding (20 points)
    A1. (5 pts)  What is the chromosome length? Why?
    A2. (5 pts)  What is the search space size? Show calculation.
    A3. (5 pts)  Why can't we use binary encoding here?
    A4. (5 pts)  Explain why this problem is multi-objective.

PART B — Implementation (50 points)
    B1. (15 pts) Complete the fitness function (evaluate)
    B2. (10 pts) Complete the crossover operator
    B3. (10 pts) Complete the mutation operator
    B4. (15 pts) Run the GA and report:
                 - Best fitness achieved
                 - Generation where convergence occurred
                 - The actual assignment (who does what)

PART C — Experimentation (20 points)
    C1. (10 pts) Run experiments with these parameter variations:
                 a) Population size: 30, 50, 100, 200
                 b) Mutation rate: 0.01, 0.05, 0.1, 0.3
                 Record best fitness and convergence generation for each.
                 Present results in a table.

    C2. (10 pts) Answer: Which parameter had the biggest impact on
                 solution quality? Why do you think that is?

PART D — Critical Thinking (10 points)
    D1. (5 pts)  A developer quits mid-sprint. How would you modify
                 the GA to handle this constraint change?
    D2. (5 pts)  Management adds a new objective: minimize task
                 dependencies (tasks that share data should go to the
                 same developer). How would you modify the fitness
                 function? Would you use NSGA-II? Why or why not?

=============================================================================
 STARTER CODE — Fill in the sections marked >>> YOUR CODE HERE <<<
=============================================================================
"""

import random
import copy
from collections import defaultdict
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. PROBLEM DATA (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────

# Skills: 0=Frontend, 1=Backend, 2=Database, 3=Testing
SKILL_NAMES = ["Frontend", "Backend", "Database", "Testing"]

# 8 Developers: name, skill_levels[4] (0-10), available_hours
DEVELOPERS = [
    {"name": "Alice",   "skills": [9, 4, 3, 5], "hours": 35},
    {"name": "Bob",     "skills": [3, 8, 6, 4], "hours": 40},
    {"name": "Carol",   "skills": [7, 5, 4, 8], "hours": 30},
    {"name": "Dave",    "skills": [2, 9, 7, 3], "hours": 40},
    {"name": "Eve",     "skills": [6, 3, 2, 9], "hours": 35},
    {"name": "Frank",   "skills": [4, 7, 9, 5], "hours": 40},
    {"name": "Grace",   "skills": [8, 6, 5, 7], "hours": 25},
    {"name": "Hank",    "skills": [5, 5, 8, 6], "hours": 35},
]

# 12 Tasks: name, required_skill (index), estimated_hours, priority(1-5)
TASKS = [
    {"name": "Login UI",          "skill": 0, "hours": 8,  "priority": 5},
    {"name": "Dashboard",         "skill": 0, "hours": 12, "priority": 4},
    {"name": "REST API",          "skill": 1, "hours": 10, "priority": 5},
    {"name": "Auth Service",      "skill": 1, "hours": 8,  "priority": 5},
    {"name": "DB Migration",      "skill": 2, "hours": 6,  "priority": 4},
    {"name": "Query Optimizer",   "skill": 2, "hours": 10, "priority": 3},
    {"name": "Unit Tests",        "skill": 3, "hours": 8,  "priority": 4},
    {"name": "Integration Tests", "skill": 3, "hours": 12, "priority": 5},
    {"name": "Search Feature",    "skill": 0, "hours": 10, "priority": 3},
    {"name": "Cache Layer",       "skill": 1, "hours": 6,  "priority": 3},
    {"name": "Data Pipeline",     "skill": 2, "hours": 8,  "priority": 4},
    {"name": "E2E Tests",         "skill": 3, "hours": 10, "priority": 4},
]

NUM_DEVELOPERS = len(DEVELOPERS)
NUM_TASKS = len(TASKS)

# ─────────────────────────────────────────────────────────────
# 2. GA PARAMETERS
# ─────────────────────────────────────────────────────────────

POP_SIZE = 80
GENERATIONS = 150
CROSSOVER_RATE = 0.85
MUTATION_RATE = 1 / NUM_TASKS   # ~1/L
TOURNAMENT_K = 3
ELITE_COUNT = 2
STAGNATION_LIMIT = 30

# Fitness weights (must sum to 1.0)
W_SKILL = 0.50      # Reward: skill match quality
W_OVERLOAD = 0.30   # Penalty: developer overload
W_BALANCE = 0.20    # Reward: workload balance

# ─────────────────────────────────────────────────────────────
# 3. CHROMOSOME REPRESENTATION
# ─────────────────────────────────────────────────────────────
#
# Integer encoding: chromosome[i] = developer assigned to task i
# Length = NUM_TASKS (12)
# Each gene ∈ {0, 1, ..., NUM_DEVELOPERS-1}
#
# Example: [2, 0, 3, 1, 5, 5, 4, 2, 0, 3, 7, 6]
#          Task 0 → Developer 2 (Carol)
#          Task 1 → Developer 0 (Alice)
#          Task 2 → Developer 3 (Dave)
#          ... etc.


def random_chromosome():
    """Create a random valid assignment."""
    return [random.randint(0, NUM_DEVELOPERS - 1) for _ in range(NUM_TASKS)]


# ─────────────────────────────────────────────────────────────
# 4. FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────

def evaluate(chromosome):
    """
    >>> YOUR CODE HERE (Part B1 — 15 points) <<<

    Calculate fitness as a weighted combination of three objectives:

    SKILL MATCH (maximize):
        For each task i assigned to developer j:
            skill_score += developer_j.skills[task_i.skill] * task_i.priority
        Normalize by dividing by max_possible_skill_score
        (max = if every task got a developer with skill level 10)

    OVERLOAD PENALTY (minimize):
        For each developer, sum the hours of tasks assigned to them.
        If total_hours > developer.available_hours:
            overload += (total_hours - available_hours)
        Normalize by dividing by total_task_hours

    WORKLOAD BALANCE (maximize):
        Calculate the workload ratio for each developer:
            ratio = assigned_hours / available_hours  (capped at 1.0)
        balance_score = 1 - std_dev(ratios)
        (Perfect balance = all devs equally loaded → std_dev = 0 → score = 1)

    FINAL FITNESS:
        f = W_SKILL * skill_score - W_OVERLOAD * overload_score + W_BALANCE * balance_score

    HINTS:
        - Use defaultdict(float) to track hours per developer
        - max_possible_skill = sum(task.priority * 10 for all tasks)
        - total_task_hours = sum(task.hours for all tasks)
        - For std_dev: sqrt(sum((x - mean)^2) / n)
    """

    # --- Replace the line below with your implementation ---

    total_score = 0
    dev_h = [0.0 for _ in range(NUM_DEVELOPERS)]

    for tidx, devidx in enumerate(chromosome):
        task = TASKS[tidx]
        dev = DEVELOPERS[devidx]
        skill_lvl = dev["skills"][task["skill"]]
        total_score += skill_lvl * task['priority']
        dev_h[devidx] += task["hours"]

    capacities = np.array([d['hours'] for d in DEVELOPERS])
    assigned_hours = np.array(dev_h)
    
    # overload
    overloads = np.maximum(assigned_hours - capacities, 0)
    total_overload = np.sum(overloads)
    
    # balance
    ratios = assigned_hours / capacities
    ratios = np.clip(ratios, 0, 1.0) 
    _std = np.std(ratios)
    # input(_std)


    max_skill = sum(t['priority'] * 10 for t in TASKS)
    norm_skill = total_score / max_skill

    task_hours = sum(t['hours'] for t in TASKS)
    norm_overload = total_overload / task_hours
    
    balance_score = 1.0-_std

    fitness = (W_SKILL*norm_skill)- (W_OVERLOAD * norm_overload) + (W_BALANCE * balance_score)

    return fitness



# ─────────────────────────────────────────────────────────────
# 5. GENETIC OPERATORS
# ─────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses):
    """Tournament selection (provided — do not modify)."""
    contestants = random.sample(range(len(population)), TOURNAMENT_K)
    winner = max(contestants, key=lambda i: fitnesses[i])
    return copy.copy(population[winner])


def crossover(parent_a, parent_b):
    """
    >>> YOUR CODE HERE (Part B2 — 10 points) <<<

    Implement UNIFORM CROSSOVER (not single-point!):
        For each gene position i:
            - With 50% probability, child gets gene from parent_a
            - Otherwise, child gets gene from parent_b
        Create two children (one biased toward A, one toward B)

    Return both children.
    Only perform crossover with probability CROSSOVER_RATE.
    If no crossover, return copies of parents.

    WHY UNIFORM? For assignment problems, single-point crossover
    creates a strong positional bias. Uniform crossover treats each
    task assignment independently, which is more appropriate here.
    """
    # --- Replace the lines below with your implementation ---

    if random.random() > CROSSOVER_RATE:
        return copy.copy(parent_a), copy.copy(parent_b)
    
    # for safety, convert to numpy arr
    p_a = np.array(parent_a)
    p_b = np.array(parent_b)

    # generate mask for crossover
    mask = np.random.rand(len(p_a)) < 0.5

    # corssover
    child_a = np.where(mask, p_a, p_b)
    child_b = np.where(mask, p_b, p_a)
    return child_a.tolist(), child_b.tolist()



def mutate(chromosome):
    """
    >>> YOUR CODE HERE (Part B3 — 10 points) <<<

    Implement RANDOM RESET MUTATION:
        For each gene position i:
            With probability MUTATION_RATE:
                Replace chromosome[i] with a random developer (0 to NUM_DEVELOPERS-1)

    This is the integer-encoding equivalent of bit-flip mutation.

    Return the mutated chromosome.
    """

    # --- Replace the lines below with your implementation ---

    # for safety, similarly, convert to numpy arr
    chromosome = np.array(chromosome)
    # generate mutation mask
    mtt_mask = np.random.rand(len(chromosome)) < MUTATION_RATE
    # gen random dvelopers for mutation
    rand_devs = np.random.randint(0, NUM_DEVELOPERS, size=len(chromosome))
    # try the mutation
    mtt_arr = np.where(mtt_mask, rand_devs, chromosome)

    return mtt_arr.tolist()


# ─────────────────────────────────────────────────────────────
# 6. DECODE & DISPLAY
# ─────────────────────────────────────────────────────────────

def decode(chromosome):
    """Convert chromosome to human-readable assignment."""
    assignments = defaultdict(list)
    for task_idx, dev_idx in enumerate(chromosome):
        assignments[dev_idx].append(task_idx)

    result = {}
    for dev_idx in range(NUM_DEVELOPERS):
        dev = DEVELOPERS[dev_idx]
        task_indices = assignments.get(dev_idx, [])
        task_hours = sum(TASKS[t]["hours"] for t in task_indices)
        task_names = [TASKS[t]["name"] for t in task_indices]

        # Calculate skill match for each task
        matches = []
        for t in task_indices:
            skill_req = TASKS[t]["skill"]
            skill_level = dev["skills"][skill_req]
            matches.append((TASKS[t]["name"], SKILL_NAMES[skill_req], skill_level))

        result[dev["name"]] = {
            "tasks": task_names,
            "hours": task_hours,
            "capacity": dev["hours"],
            "utilization": task_hours / dev["hours"] * 100 if dev["hours"] > 0 else 0,
            "overloaded": task_hours > dev["hours"],
            "matches": matches
        }
    return result


def display_solution(chromosome):
    """Pretty-print the solution."""
    result = decode(chromosome)
    print(f"\n  {'Developer':<10} {'Tasks':<45} {'Hours':>6} {'Cap':>5} {'Util':>6} {'Status'}")
    print(f"  {'─'*10} {'─'*45} {'─'*6} {'─'*5} {'─'*6} {'─'*10}")
    for name, info in result.items():
        tasks_str = ", ".join(info["tasks"]) if info["tasks"] else "(idle)"
        status = "⚠ OVERLOAD" if info["overloaded"] else "✓ OK"
        print(f"  {name:<10} {tasks_str:<45} {info['hours']:>5}h {info['capacity']:>4}h {info['utilization']:>5.1f}% {status}")

    print(f"\n  Skill Matches:")
    for name, info in result.items():
        for task_name, skill_name, level in info["matches"]:
            quality = "★★★" if level >= 8 else "★★" if level >= 5 else "★"
            print(f"    {name:<8} → {task_name:<20} needs {skill_name:<10} skill={level}/10 {quality}")


# ─────────────────────────────────────────────────────────────
# 7. EVOLUTION LOOP (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────

def run_ga():
    """Main GA loop."""
    random.seed(42)

    population = [random_chromosome() for _ in range(POP_SIZE)]
    fitnesses = [evaluate(ind) for ind in population]

    best_ever = max(fitnesses)
    best_chromosome = copy.copy(population[fitnesses.index(best_ever)])
    history = {"best": [], "mean": []}
    stagnation = 0

    print("\n" + "=" * 70)
    print("  GA FOR SOFTWARE DEVELOPER ASSIGNMENT")
    print("=" * 70)
    print(f"  {NUM_TASKS} tasks, {NUM_DEVELOPERS} developers")
    print(f"  Search space: {NUM_DEVELOPERS}^{NUM_TASKS} = {NUM_DEVELOPERS**NUM_TASKS:,}")
    print(f"  Pop={POP_SIZE} | Gens={GENERATIONS} | Cx={CROSSOVER_RATE} | Mut={MUTATION_RATE:.3f}")
    print(f"  Weights: skill={W_SKILL}, overload={W_OVERLOAD}, balance={W_BALANCE}")

    print(f"\n  {'Gen':<6} {'Best':>8} {'Mean':>8} {'Status'}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*20}")

    for gen in range(GENERATIONS):
        fitnesses = [evaluate(ind) for ind in population]
        gen_best = max(fitnesses)
        gen_mean = sum(fitnesses) / len(fitnesses)

        history["best"].append(gen_best)
        history["mean"].append(gen_mean)

        if gen_best > best_ever:
            best_ever = gen_best
            best_chromosome = copy.copy(population[fitnesses.index(gen_best)])
            stagnation = 0
            status = "★ NEW BEST"
        else:
            stagnation += 1
            status = ""

        if gen % 10 == 0 or status:
            print(f"  {gen:<6} {gen_best:>8.4f} {gen_mean:>8.4f} {status}")

        if stagnation >= STAGNATION_LIMIT:
            print(f"\n  ⏹  Converged at generation {gen}")
            break

        # Elitism
        sorted_idx = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)
        new_population = [copy.copy(population[sorted_idx[i]]) for i in range(ELITE_COUNT)]

        while len(new_population) < POP_SIZE:
            pa = tournament_select(population, fitnesses)
            pb = tournament_select(population, fitnesses)
            ca, cb = crossover(pa, pb)
            ca = mutate(ca)
            cb = mutate(cb)
            new_population.append(ca)
            if len(new_population) < POP_SIZE:
                new_population.append(cb)

        population = new_population

    return best_chromosome, best_ever, history


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Print problem data
    print("=" * 70)
    print("  PROBLEM DATA")
    print("=" * 70)

    print(f"\n  DEVELOPERS:")
    print(f"  {'Name':<10} {'Frontend':>9} {'Backend':>8} {'Database':>9} {'Testing':>8} {'Hours':>6}")
    print(f"  {'─'*10} {'─'*9} {'─'*8} {'─'*9} {'─'*8} {'─'*6}")
    for d in DEVELOPERS:
        print(f"  {d['name']:<10} {d['skills'][0]:>9} {d['skills'][1]:>8} {d['skills'][2]:>9} {d['skills'][3]:>8} {d['hours']:>5}h")

    print(f"\n  TASKS:")
    print(f"  {'Task':<22} {'Skill':<10} {'Hours':>6} {'Priority':>9}")
    print(f"  {'─'*22} {'─'*10} {'─'*6} {'─'*9}")
    for t in TASKS:
        print(f"  {t['name']:<22} {SKILL_NAMES[t['skill']]:<10} {t['hours']:>5}h {t['priority']:>9}")

    total_hours = sum(t["hours"] for t in TASKS)
    total_capacity = sum(d["hours"] for d in DEVELOPERS)
    print(f"\n  Total task hours: {total_hours}h | Total capacity: {total_capacity}h | Utilization target: {total_hours/total_capacity*100:.1f}%")

    # Run GA
    best_chrom, best_fit, history = run_ga()

    # Display results
    print("\n" + "=" * 70)
    print("  BEST SOLUTION FOUND")
    print("=" * 70)
    print(f"\n  Chromosome: {best_chrom}")
    print(f"  Fitness:    {best_fit:.4f}")
    display_solution(best_chrom)

    # Convergence curve (ASCII)
    print("\n" + "─" * 70)
    print("  FITNESS CONVERGENCE")
    print("─" * 70)
    bh = history["best"]
    if bh:
        mx, mn = max(bh), min(bh)
        rows = 10
        cols = min(50, len(bh))
        step = max(1, len(bh) // cols)
        sampled = [bh[i * step] for i in range(cols)]
        for row in range(rows, -1, -1):
            thr = mn + (mx - mn) * row / rows if mx > mn else mn
            line = "  │"
            for v in sampled:
                line += "█" if v >= thr else " "
            if row == rows: line += f"  {mx:.4f}"
            elif row == 0:  line += f"  {mn:.4f}"
            print(line)
        print("  └" + "─" * cols)
        print(f"   Gen 0{' ' * (cols - 10)}Gen {len(bh)-1}")

    print("\n" + "=" * 70)
    print("  SUBMISSION CHECKLIST")
    print("=" * 70)
    print("""
    □ Part A: Written answers (A1-A4)
    □ Part B: Completed code (evaluate, crossover, mutate) + output
    □ Part C: Parameter experiment table + analysis
    □ Part D: Written answers (D1-D2)

    Submit: This .py file with your code + a PDF with written answers.
    """)
