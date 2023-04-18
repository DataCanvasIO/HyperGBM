Version 0.3.0
-------------

We add a few new features to this version:

* Multi-objectives optimization

    * optimization algorithm
        - add MOEA/D(Multiobjective Evolutionary Algorithm Based on Decomposition)
        - add Tchebycheff, Weighted Sum, Penalty-based boundary intersection approach(PBI) decompose approachs
        - add shuffle crossover, uniform crossover, single point crossover strategies for GA based algorithms
        - automatically normalize objectives of different dimensions
        - automatically convert maximization problem to minimization problem
        - add NSGA-II(Non-dominated Sorting Genetic Algorithm)
        - add R-NSGA-II(A new dominance relation for multicriteria decision making)

    * builtin objectives
        - number of features
        - prediction performance
