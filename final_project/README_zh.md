# Final Project Thoughts
## 单目标基准
### 问题描述
我们的研究目标是下一版本问题. 下一版本问题(Next Release Problem) 旨在从众多备选需求（Requirements）中，选择一个最优子集加入到产品的下一个版本中。其目标通常是： 
- 最大化收益：提高客户满意度或公司的商业利润。
- 最小化成本：在预算、人力和开发时间的限制内完成开发。
我们使用单目标基准的配置下以NRP为场景探索不同基于搜索的软件工程算法的表现以及可能的改进.

### 数据集
我们采用Xuan等人的[数据集](http://oscar-lab.org/people/~jxuan/page/project/nrp/).

### 综述性实验
该部分旨在探索现有常用单目标优化算法的表现. 我们比较以下四种算法:
1. Random Search
    - sample_p = problem.cost_ratio：每位独立以此概率置 1（biased toward feasible region） 
2. Hill Climbing
    - 邻域：1-bit flip
    - 接受规则：strict improvement（仅 profit 严格变大才接受）
    - restart_after = 2000：连续 2000 次 non-improving evals 触发 random restart
    - 重启时用 sample_p = cost_ratio
3. Simulated Annealing
    - 邻域：1-bit flip
    - 接受规则：Boltzmann（improvement 必接受；worsening 以 exp(Δ/T) 接受）
    - t0 = None：auto-tune（用 50 个随机邻居的平均 |worsening|，使初始 acceptance prob ≈ 0.5）
    - alpha = 0.9995：每 eval 几何冷却 T ← T × α
    - t_min = 1e-3：温度下限
4. Classic Genetic Algorithm
    - pop_size = 100/300/600
    - cx_prob = 0.8：crossover 触发概率
    - crossover：uniform，per-bit swap with indpb = 0.5
    - mut_prob = None：default [1/5]/n_reqs（每位独立 flip 概率）
    - mutation：bit-flip，per-bit 独立
    - tournsize = 3：tournament selection
    - elitism = 2：每代保留 top-2 不变
    - 评估时 repair 后写回基因型（Lamarckian）

实验发现: 算法1, 2效果较差且过早收敛. 算法4的表现在默认配置下(100; 1/n_reqs)劣于算法3, 也有过早收敛的趋势. 在增加算法4的pop_size至300后, 虽然依旧弱于算法3但有所改善. pop_size增加至600后表现优于算法4. 在增加算法4的mut_prob至5/n_reqs后, 即使采用较小的pop_size (=100)其表现也与算法3相近或优于算法3.

结论: 算法1与算法2难以解决NRP一类的多条件限制优化问题. 算法3表现稳定, 但收敛速度较慢, 可以在budget充足或对实时性要求不高时采用. 算法4中mut_prob参数的对表现的影响能力大于pop_size的影响能力; 由于GA算法的特性, 算法4收敛快速, 可在budget有限的情况下采用.

### 改进设计 (Adaptive GA)
从综述性试验中可知, GA算法在较高mut_prob时表现优异. 但较高的mut_prob在优化后期会破坏积累下来的优秀基因组合(Building Blocks),导致 GA 在后期无法进行精细的局部微调(丧失了 Exploitation 能力), 无法锁定最终的全局最优点.

一种可行的解决方案是将mut_prob动态化. 在优化初期, 我们希望尽量采取已有的优质基因组和,故采用较小的mut_prob(exploitation); 优化中期, 我们鼓励系统去探索更多组合, 故这一阶段mut_prob应较大(exploration); 优化后期, 系统已经积攒了足够的优秀基因组合, 我们希望其利用这些组合收敛,故减小mut_prob. 

**1/5法则(Rechenberg (1973))**: 通常在连续域优化中，如果成功率高，说明当前步长（突变）合适，可以增大步长以加速；如果成功率低，说明步长太大导致跳过了最优解，需要减小步长进行微调。我们令该阈值为1/5 (0.2), 成功率超过该阈值(合适)则将mut_prob乘以mut_factor, 否则除以mut_factor.

实验发现: 在默认配置(mut_prob=1/n_reqs; pop_size = 100)时表现远优于算法4, 在50K步的优化前期表现多优于算法3. 后期则与算法3持平或略劣于算法3. mut_prob图表反映出了其expoitation -> exploration -> exploitation的趋势. 

结论: 与算法3相比, Adaptive GA在20K-30K budget时表现优异, 适合budget有限或对延迟有一定要求的优化任务. 而在当前配置下,算法3直到 50k evals 都还在做有意义的 Boltzmann 探索，没有结构性的减速机制，斜率近乎线性, 因此算法3适合budget充足的长期优化.

