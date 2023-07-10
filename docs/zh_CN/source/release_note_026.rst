Version 0.3.0
-------------

本次发布优化及新增的特性：

* 多目标优化

    * 优化算法
        - 新增MOEA/D(基于分解的多目标优化框架)
        - 新增目标分解方法: Tchebycheff, Weighted Sum, Penalty-based boundary intersection approach(PBI)
        - 新增适用于遗传算法的交叉算子: shuffle crossover, uniform crossover, single point crossover
        - 动态归一化不同量纲目标
        - 自动将最大化问题转换为最小化问题
        - 新增NSGA-II(非支配解快速排序算法)
        - 新增R-NSGA-II(为决策偏好提供支持的多目标优化算法)

    * 内置目标
        - 使用到特征估计
        - 预测性能
