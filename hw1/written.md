# Homework 1

## Part A

> TO BE DONE HERE

## Part B

*Check the attached `.py` file*

## Part C

#### Table of Population Size

| Population Size | Fitness | Gen. Convergence |
| --------------- | ------- | ---------------- |
| 30              | 0.6002  | 54               |
| 50              | 0.6081  | 75               |
| 80 (baseline)   | 0.6121  | 73               |
| 100             | 0.6136  | 63               |
| 200             | 0.6136  | 50               |

#### Table of Mutation Rate

| Mutation Rate   | Fitness | Gen. Convergence |
| --------------- | ------- | ---------------- |
| 0.01            | 0.6037  | 40               |
| 0.05            | 0.6056  | 53               |
| `1 / NUM_TASKS` | 0.6121  | 73               |
| 0.1             | 0.6134  | 100              |
| 0.3             | 0.6047  | 99               |

#### Observations

Based on the data listed above, I think population size has a bigger impact on the quality of the result. We can see the range of fitness: 

​	$$ Range(pop) = 0.6136-0.6002 = 0.01340$$

​	$$Range(m\_rate)=0.6134 - 0.6037 = 0.0097$$

Reason: A small population means a limited gene pool. The algorithm quickly makes all individuals to look identical, leading to “inbreeding” and getting stuck in local optima. This explains why convergence occurred after only 54 generations at `size=30`, yet the score remained low (premature convergence).