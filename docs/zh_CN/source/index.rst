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
   安装教程<installation.md>
   快速开始<quick_start.md>
   新特性<release_note.md>
   编程手册<programing_guide.md>
   样例<example.md>
   API文档<api.md>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
