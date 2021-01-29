## Searchers

### MCTSSearcher

蒙特卡洛树搜索（Monte-Carlo Tree Search）是强化学习的一个分支，有着非常高效的搜索效率，可以满足高维动态搜索空间的效率需求。 MCTS扩展了著名的Multi-armed Bandit算法到树结构的搜索空间，MCTS通过selection, expansion, playout 和 backpropagation四个阶段不断迭代完成搜索。 

**Code example**
```
from hypernets.searchers import MCTSSearcher

searcher = MCTSSearcher(search_space_fn, use_meta_learner=False, max_node_space=10, candidates_size=10, optimize_direction='max')
```

**必选参数**
- *space_fn*: callable, 可以返回`HyperSpace`的搜索空间函数。

**可选参数**
- *policy*: hypernets.searchers.mcts_core.BasePolicy, (default=None), 设置*Selection* and *Backpropagation* 阶段的实现策略, 默认使用 `UCT` 策略.
- *max_node_space*: int, (default=10), 节点*Expansion*的最大空间。
- *use_meta_learner*: bool, (default=True), Meta-learner使用已评估的样本做为数据训练一个模型用来评估未知样本的表现，它使用模拟的方式在一组候选路径中选择最有前途的一条，而不需要对特定样本完成真正的训练，可以节省大量的时间和资源，有效的提升搜索效率。
- *candidates_size*: int, (default=10), Meta-learner在评估候选路径时采样的路径数量。
- *optimize_direction*: 'min' or 'max', (default='min'), 搜索器的优化方向，是寻找reward的最小值还是最大值。
- *space_sample_validation_fn*: callable or None, (default=None), 用于校验来自搜索空间的样本的有效性，可以通过这个函数来指定特定搜索空间的约束规则以减小搜索空间的大小。


### EvolutionSearcher

进化算法（EA）是evolutionary computation的子集，这是一种基于种群的通用启发式优化算法。 EA受生物进化启发的机制，例如reproduction, mutation, recombination, and selection。 优化的候选解决方案在总体中扮演个体的角色，而fitness函数决定了解的质量（另请参见损失函数）。 然后，在重复应用上述算子之后，种群就会发生演化。


**Code example**
```
from hypernets.searchers import EvolutionSearcher

searcher = EvolutionSearcher(search_space_fn, population_size=20, sample_size=5, optimize_direction='min')
```

**必选参数**
- *space_fn*: callable, 可以返回`HyperSpace`的搜索空间函数。
- *population_size*: int, 种群的数量
- *sample_size*: int, 在每一次进化循环中选择的父代候选样本的数量。

**可选参数**
- *regularized*: bool, (default=False), 是否启用正则进化。
- *use_meta_learner*: bool, (default=True), Meta-learner使用已评估的样本做为数据训练一个模型用来评估未知样本的表现，它使用模拟的方式在一组候选路径中选择最有前途的一条，而不需要对特定样本完成真正的训练，可以节省大量的时间和资源，有效的提升搜索效率。
- *candidates_size*: int, (default=10), Meta-learner在评估候选路径时采样的路径数量。
- *optimize_direction*: 'min' or 'max', (default='min'), 搜索器的优化方向，是寻找reward的最小值还是最大值。
- *space_sample_validation_fn*: callable or None, (default=None), 用于校验来自搜索空间的样本的有效性，可以通过这个函数来指定特定搜索空间的约束规则以减小搜索空间的大小。


### RandomSearch

随机搜索器是一种每一次在搜索空间中随机选取样本进行搜索的算法

**Code example**
```
from hypernets.searchers import RandomSearcher

searcher = RandomSearcher(search_space_fn, optimize_direction='min')
```

**必选参数**
- *space_fn*: callable, 可以返回`HyperSpace`的搜索空间函数。

**可选参数**
- *optimize_direction*: 'min' or 'max', (default='min'), 搜索器的优化方向，是寻找reward的最小值还是最大值。
- *space_sample_validation_fn*: callable or None, (default=None), 用于校验来自搜索空间的样本的有效性，可以通过这个函数来指定特定搜索空间的约束规则以减小搜索空间的大小。
