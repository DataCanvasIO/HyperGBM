通过Python使用HyperGBM
=========================================


HyperGBM基于Python开发，推荐利用Python工具 *make_experiment* 创建实验并进行训练得到模型。

通过 *make_experiment* 训练模型的基本步骤：

* 准备数据集(pandas 或 dask DataFrame)
* 通过工具 *make_experiment* 创建实验
* 执行实验 *.run()* 方法进行训练得到模型
* 利用模型进行数据预测或Python工具 *pickle* 存储模型


准备数据集
-----------------

可以根据实际业务情况通过pandas或dask加载数据，得到用于模型训练的DataFrame。

以sklearn内置数据集 *breast_cancer* 为例，可如下处理得到数据集:

.. code-block:: python

    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X,y = datasets.load_breast_cancer(as_frame=True,return_X_y=True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=335)
    train_data = pd.concat([X_train,y_train],axis=1)
       

 
其中 *train_data* 用于模型训练（包括目标列），*X_test* 和 *y_test* 用于模型评价。


创建实验并进行训练
---------------------------------

假设希望最终模型有比较好的precision，为前面准备的训练数据集创建实验并开始训练模型如下：


.. code-block:: python
  
    from hypergbm import make_experiment


    experiment = make_experiment(train_data, target='target', reward_metric='precision')
    estimator = experiment.run()

其中 *estimator* 就是训练所得到的模型。


输出模型
---------------------------------


推荐利用 *pickle* 存储HyperGBM模型，如下：

.. code-block:: python

  import pickle
  with open('model.pkl','wb') as f:
    pickle.dump(estimator, f)



评价模型
---------------------------------


可利用sklearn提供的工具进行模型评价，如下： 

.. code-block:: python

    from sklearn.metrics import classification_report

    y_pred=estimator.predict(X_test)
    print(classification_report(y_test, y_pred, digits=5))

输出：

.. code-block:: console

                  precision    recall  f1-score   support

               0    0.96429   0.93103   0.94737        58
               1    0.96522   0.98230   0.97368       113

        accuracy                        0.96491       171
       macro avg    0.96475   0.95667   0.96053       171
    weighted avg    0.96490   0.96491   0.96476       171


更多信息
------------

关于 *make_experiment* 的更多信息，您可以查看该工具的docstring，如：

.. code-block:: python

  print(make_experiment.__doc__)
  
  
如果您正在Notebook或IPython中使用HyperGBM, 可以通过如下方式获取 *make_experiment* 的更多信息：

.. code-block:: ipython

  make_experiment?


