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



Feature matrix
---------------------
There are 3 training modes：

- Standalone
- Distributed on single machine
- Distributed on multiple machines

Here is feature matrix of training modes:

+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|   #                        | Feature                                              | Standalone | Distributed on single machine | Distributed on multiple machines |
+============================+======================================================+============+===============================+==================================+
| Feature engineering        | | Feature generation                                 | | √        | |                             | |                                |
|                            | | Feature dimension reduction                        | | √        | | √                           | | √                              |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
| Data clean                 | | Correct data type                                  | | √        | | √                           | | √                              |
|                            | | Special empty value handing                        | | √        | | √                           | | √                              |
|                            | | Id-ness features cleanup                           | | √        | | √                           | | √                              |
|                            | | Duplicate features cleanup                         | | √        | | √                           | | √                              |
|                            | | Empty label rows cleanup                           | | √        | | √                           | | √                              |
|                            | | Illegal values replacement                         | | √        | | √                           | | √                              |
|                            | | Constant features cleanup                          | | √        | | √                           | | √                              |
|                            | | Collinearity features cleanup                      | | √        | | √                           | | √                              |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Data set split              | Adversarial validation                               | | √        | |                             | |                                |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Modeling algorithms         | | XGBoost                                            | | √        | | √                           | | √                              |
|                            | | Catboost                                           | | √        | | √                           | |                                |
|                            | | LightGBM                                           | | √        | | √                           | |                                |
|                            | | HistGridientBoosting                               | | √        | |                             | |                                |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Training                    | | Task inference                                     | | √        | | √                           | | √                              |
|                            | | Command-line tools                                 | | √        | |                             | |                                |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Evaluation strategies       | | Cross-validation                                   | | √        | | √                           | | √                              |
|                            | | Train-Validation-Holdout                           | | √        | | √                           | | √                              |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Search strategies           | | Monte Carlo Tree Search                            | | √        | | √                           | | √                              |
|                            | | Evolution                                          | | √        | | √                           | | √                              |
|                            | | Random search                                      | | √        | | √                           | | √                              |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Class balancing             | | Class Weight                                       | | √        | | √                           | |                                |
|                            | | Under-Samping(Near miss,Tomeks links,Random)       | | √        | |                             | |                                |
|                            | | Over-Samping(SMOTE,ADASYN,Random)                  | | √        | |                             | |                                |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Early stopping strategies   | | max_no_improvement_trials                          | | √        | | √                           | | √                              |
|                            | | time_limit                                         | | √        | | √                           | | √                              |
|                            | | expected_reward                                    | | √        | | √                           | | √                              |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+
|Advance features            | | Two stage search(Pseudo label,Feature selection)   | | √        | | √                           | | √                              |
|                            | | Concept drift handling                             | | √        | | √                           | | √                              |
|                            | | Ensemble                                           | | √        | | √                           | | √                              |
+----------------------------+------------------------------------------------------+------------+-------------------------------+----------------------------------+


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

   Overview<overview.md>
   Installation<installation.md>
   Quick-Start<quick_start.md>
   Examples<example.rst>
   How-To <how_to.rst>
   Release Note<release_note.rst>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
