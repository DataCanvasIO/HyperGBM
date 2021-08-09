
关于HyperGBM
======================

HyperGBM是一款全Pipeline自动机器学习工具，可以端到端的完整覆盖从数据清洗、预处理、特征加工和筛选以及模型选择和超参数优化的全过程，是一个真正的结构化数据AutoML工具包。

大部分的自动机器学习工具主要解决的是算法的超参数优化问题，而HyperGBM是将从数据清洗到算法优化整个的过程放入同一个搜索空间中统一优化。这种端到端的优化过程更接近于SDP(Sequential Decision Process)场景，因此HyperGBM采用了强化学习、蒙特卡洛树搜索等算法并且结合一个meta-leaner来更加高效的解决全Pipeline优化的问题，并且取得了非常出色的效果。

正如名字中的含义，HyperGBM中的机器学习算法使用了目前最流行的几种GBM算法（更准确的说是梯度提升树模型），目前包括XGBoost、LightGBM和Catboost三种。同时，HyperGBM也引入了Hypernets的CompeteExperiment在数据清理、特征工程、特征筛选、模型融合等环节的很多高级特性。

HyperGBM中的优化算法和搜索空间表示技术以及CompeteExperiment由 [Hypernets](https://github.com/DataCanvasIO/Hypernets)项目提供支撑。
