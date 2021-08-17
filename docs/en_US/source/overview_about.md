About HyperGBM
======================

HyperGBM is a Full-Pipeline Automated Machine Learning Tool with functions ranging from data cleaning, preprocessing and feature engineering to model selection and hyperparameter tuning. It is an advanced AutoML tool for tabular data.

While a lot of AutoML tools mainly focus on the hyperparameter tuning of different algorithms, HyperGBM designs a high-level search space to include almost all components of machine learning modelling into it, such as data cleaning and algorithm optimizing. This end-to-end optimization approach is more close to a SDP(Sequential Decision Process). Therefore, combined with a meta-learner, HyperGBM adopts advanced algorithms such as reinforcement learning and Monte-Carlo tree search to solve the full-pipeline optimization problem more effectively. These strategies are proven to be effective in practice.

For the machine leanring models, HyperGBM uses popular gradient-boosting tree models ranging from XGBoost, LightGBM and HistGradientBoosting. Besides, HyperGBM also involves many advanced features of CompeteExperiment from Hypernets in data cleaning, feature engineering and model ensemble.

The optimization algorithms, representations of search space and CompeteExperiment are based on [Hypernets](https://github.com/DataCanvasIO/Hypernets).