Use HyperGBM with Python
=========================================


HyperGBM is developed with Python. We recommend using the Python tool *make_experiment* to create experiment and train the model.

The basic steps for training the model with *make_experiment* are as follows：

* Prepare the dataset(pandas or dask DataFrame)
* Create experiment with *make_experiment*
* Call the *.run()* method of experiment to performing training and get the model
* Predict with trained model or save it with the Python tool *pickle*


Prepare the dataset
--------------------------

Both pandas and dask can be loaded depending on your task types to get DataFrame for training the model.

Taking loading the sklearn dataset *breast_cancer* as an example，one can get the dataset by following several procedures:

.. code-block:: python

    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X,y = datasets.load_breast_cancer(as_frame=True,return_X_y=True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=335)
    train_data = pd.concat([X_train,y_train],axis=1)
       

 
where *train_data* is used for model trianing while *X_test* are *y_test* used for evaluating the model.


Create experiment with *make_experiment*
--------------------------------------------

Users can creating experiment for the prepared dataset and start training the model following procedures below：


.. code-block:: python
  
    from hypergbm import make_experiment


    experiment = make_experiment(train_data, target='target', reward_metric='precision')
    estimator = experiment.run()

where *estimator* is the trianed model.


Save the model
---------------------------------


It is recommended to save the model with *pickle*：

.. code-block:: python

  import pickle
  with open('model.pkl','wb') as f:
    pickle.dump(estimator, f)



Evaluate the model
---------------------------------


The model can be evaluated with tools provided by sklearn： 

.. code-block:: python

    from sklearn.metrics import classification_report

    y_pred=estimator.predict(X_test)
    print(classification_report(y_test, y_pred, digits=5))

output:

.. code-block:: console

                  precision    recall  f1-score   support

               0    0.96429   0.93103   0.94737        58
               1    0.96522   0.98230   0.97368       113

        accuracy                        0.96491       171
       macro avg    0.96475   0.95667   0.96053       171
    weighted avg    0.96490   0.96491   0.96476       171


More info:
------------

Please refer to the docstring of *make_experiment* for more information about it：

.. code-block:: python

  print(make_experiment.__doc__)
  
  
If you are using Notebook or IPython, the following code can provide more information about *make_experiment*:

.. code-block:: ipython

  make_experiment?


