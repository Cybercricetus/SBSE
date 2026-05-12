# SBSE: Guide of Final Project

**ONLY `./final_project` directory is the final project.**

## Step 1: Clone the repo
- Clone the repo: 
```bash
git clone git@github.com:Cybercricetus/SBSE.git
cd SBSE/final_project
```

- Make sure that you download the dataset from [here](http://oscar-lab.org/people/~jxuan/page/project/nrp/) and unzip to `SBSE/final_project/data`

## Step 2: Install the environment

- Create a virtual env and install all packages you need
```bash
python3 -m venv venv    # strongly recommand to have a virtual env
source venv/bin/activate            
pip install -r requirements.txt
```

- Validation:
```bash
python -c "from nrp import parse_nrp_file; print('OK')"
```

## Run a single instance

```bash
python main.py --instance data/nrp-e1.txt
```

- By default: `--cost-ratio 0.3`,`--runs 30`, `--evals 50000`,`--seed 0`, and all 5 algorithms will run

- You may change these flags.

- Full format:

```bash
python main.py \
    --instance data/nrp-e1.txt \
    --cost-ratio 0.3 \
    --runs 30 \
    --evals 50000 \
    --seed 0 \
    --algorithms random hc sa ga aga \
    --output results/e1_r03.json \
    --plot results/e1_r03
```

- Run a subset:

```python
# GA + AGA ONLY
python main.py --instance data/nrp-e1.txt --algorithms ga aga

# baseline + SA only
python main.py --instance data/nrp-e1.txt --algorithms random hc sa
```

## Run in Batch
- Give the bash script execution previlege:
```bash
chmod +x run.sh
```

- Run all 12 instances with the default configuraiton

```bash
./run.sh
```

- You can adjust the configuration by changing the environment vars:

```bash

RATIO=0.5 ./run.sh

RUNS=10 EVALS=20000 ./run.sh

SEED=42 ./run.sh

RATIO=0.5 RUNS=10 EVALS=20000 SEED=42 ./run.sh
```

## Hyperparameter Changing

- For GA series: Open `final_project/nrp/algorithms.py` (or the exact file name) and modify the parameters in the `genetic_algorithm` function:

```python
# GA
def genetic_algorithm(
    problem,
    max_evals,
    rng,
    *,
    pop_size: int = 100,              # ← 100/300/600
    cx_prob: float = 0.8,
    mut_prob: float | None = None,    # 5.0/problem.n_reqs
    tournsize: int = 3,
    elitism: int = 2,
    ...
)

# AGA
def adaptive_genetic_algorithm(
    problem,
    max_evals,
    rng,
    *,
    pop_size: int = 100,        
    mut_window: int = 5,        # sliding window
    mut_target: float = 0.2,    # 1/5 rule, can be changed
    mut_factor: float = 1.2,    # factor beta
    ...
)
```

## Analyze the result

```bash

# Analyze single config, all instances
python analyze.py results/<your path>/*.json --csv aga_100_1n_summary.csv


# change baseline
python analyze.py results/<your path>/*.json --baseline sa


# analyze only a single instance
python analyze.py results/<your path>/e1_r03.json
```

## Directory Tree

```text
.
├── README.md
├── final_project
│   ├── README_zh.md
│   ├── analyze.py
│   ├── data
│   ├── main.py
│   ├── nrp
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── algorithms.py
│   │   ├── experiment.py
│   │   ├── plotting.py
│   │   └── problem.py
│   ├── requirements.txt
│   ├── results
│   └── run.sh
└── hw1
    ├── ga_assignment_student.py
    ├── outputs
    │   ├── base.log
    │   ├── mttr_0.01.log
    │   ├── mttr_0.05.log
    │   ├── mttr_0.1.log
    │   ├── mttr_0.3.log
    │   ├── pop_100.log
    │   ├── pop_200.log
    │   ├── pop_30.log
    │   └── pop_50.log
    ├── written.md
    └── written.pdf
```