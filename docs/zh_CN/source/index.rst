Overview
=====================

What is HyperGBM
---------------------

HyperGBM is a library that supports full-pipeline AutoML, which completely covers the end-to-end stages of data cleaning, preprocessing, feature generation and selection, model selection and hyperparameter optimization.It is a real-AutoML tool for tabular data.

Overview
---------
Unlike most AutoML approaches that focus on tackling the hyperparameter optimization problem of machine learning algorithms, HyperGBM can put the entire process from data cleaning to algorithm selection in one search space for optimization. End-to-end pipeline optimization is more like a sequential decision process, thereby HyperGBM uses reinforcement learning, Monte Carlo Tree Search, evolution algorithm combined with a meta-learner to efficiently solve such problems.
As the name implies, the ML algorithms used in HyperGBM are all GBM models, and more precisely the gradient boosting tree model, which currently includes XGBoost, LightGBM and Catboost.
The underlying search space representation and search algorithm in HyperGBM are powered by the  `Hypernets <https://github.com/DataCanvasIO/Hypernets>`_ project a general AutoML framework.


特性矩阵
---------------------

其中HyperGBM有3中运行模式，分别为：

- 单机模式
- 单机分布式
- 多机分布式

三种模式对与功能特性的执行情况如下表:

+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
|#                             |特征                                                        |单机模式                      |单机分布式模式                |多机分布式                    |
+==============================+=============================+==============================+==============================+==============================+==============================+
| | 特征工程                   | | 特征衍生                                                 | | √                          | |                            | |                            |
| |                            | | 特征降维                                                 | | √                          | | √                          | | √                          |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 数据清洗                   | | 特殊空值字符处理                                         | | √                          | | √                          | | √                          |
| |                            | | 列类型校正                                               | | √                          | | √                          | | √                          |
| |                            | | 常量列清理                                               | | √                          | | √                          | | √                          |
| |                            | | 重复列清理                                               | | √                          | | √                          | | √                          |
| |                            | | 删除标签列为空的样本                                     | | √                          | | √                          | | √                          |
| |                            | | 非法值替换                                               | | √                          | | √                          | | √                          |
| |                            | | id列清理                                                 | | √                          | | √                          | | √                          |
| |                            | | 共线性特征清理                                           | | √                          | | √                          | | √                          |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 数据集拆分                 | | 对抗验证                                                 | | √                          | |                            | |                            |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 建模算法                   | | XGBoost                                                  | | √                          | | √                          | | √                          |
| |                            | | Catboost                                                 | | √                          | | √                          | |                            |
| |                            | | LightGBM                                                 | | √                          | | √                          | |                            |
| |                            | | HistGridientBoosting                                     | | √                          | |                            | |                            |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 模型训练                   | | 自动任务类型推断                                         | | √                          | | √                          | | √                          |
| |                            | | 命令行工具                                               | | √                          | |                            | |                            |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 评估方法                   | | 交叉验证(Cross-Validation)                               | | √                          | | √                          | | √                          |
| |                            | | Train-Validation-Holdout验证                             | | √                          | | √                          | | √                          |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 搜索算法                   | | 蒙特卡洛树算法                                           | | √                          | | √                          | | √                          |
| |                            | | 进化算法                                                 | | √                          | | √                          | | √                          |
| |                            | | 随机搜索                                                 | | √                          | | √                          | | √                          |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 类平衡                     | | Class Weight                                             | | √                          | | √                          | |                            |
| |                            | | 降采样(Near miss,Tomeks links,Random)                    | | √                          | |                            | |                            |
| |                            | | 过采样(SMOTE,ADASYN,Random)                              | | √                          | |                            | |                            |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 提前停止策略               | | n次搜索都不再提升,提前停止                               | | √                          | | √                          | | √                          |
| |                            | | 最大用时间提前停止                                       | | √                          | | √                          | | √                          |
| |                            | | expected_reward                                          | | √                          | | √                          | | √                          |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+
| | 高级特性                   | | 二阶段搜索(伪标签,特征选择)                              | | √                          | | √                          | | √                          |
| |                            | | 概念漂移处理                                             | | √                          | | √                          | | √                          |
| |                            | | 模型融合                                                 | | √                          | | √                          | | √                          |
+------------------------------+------------------------------------------------------------+------------------------------+------------------------------+------------------------------+


Hypernets related projects
--------------------------

* `HyperGBM/HyperDT <https://github.com/DataCanvasIO/HyperGBM>`_ : A full pipeline AutoML tool integrated various GBM models.
* `DeepTables <https://github.com/DataCanvasIO/DeepTables>`_: An AutoDL tool for tabular data.
* `HyperKeras <https://github.com/DataCanvasIO/HyperKeras>`_: An AutoDL tool for Neural Architecture Search and Hyperparameter Optimization on Tensorflow and Keras.
* `Cooka <https://github.com/DataCanvasIO/Cooka>`_: Lightweight interactive AutoML system.
* `Hypernets <https://github.com/DataCanvasIO/Hypernets>`_: A general automated machine learning framework.

.. image:: ../../static/images/datacanvas_automl_toolkit.png


DataCanvas
-----------
HyperGBM is an open source project created by `DataCanvas <https://www.datacanvas.com>`_ .



.. toctree::
   :maxdepth: 2
   :caption: Home:

   概览<overview.md>
   安装教程<installation.md>
   快速开始<quick_start.md>
   样例<example.rst>
   How-To<how_to.md>
   Release Note<release_note.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
