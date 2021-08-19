Advanced applications
=========================

HyperGBM *make_experiment* create an instance of *CompeteExperiment* in Hypernets. There are many advanced features of *CompeteExperiment* which will be covered in this section.

.. mermaid::

    flowchart LR
        dc[Data<br/>Cleaning]
        fg[Feature generation]
        cd[collinearity detection]
        dd[Drift detection]
        fs[Feature selection]
        s1[Search optimization]
        pi[2nd-stage<br/>Feature<br/>selection]
        pl[Pseudo label]
        s2[2nd-stage<br/>search optimization]
        em[Model<br/>ensemble]
        op2[op]

        subgraph 1st-stage
            direction LR
            subgraph op
                direction TB
                cd-->dd
                dd-->fs
            end
            fg-->op
            op-->s1
        end
        subgraph 2nd-stage
            direction LR
            subgraph op2
                direction TB
                pi-->pl
            end
            op2-->s2
        end
        dc-->1st-stage-->2nd-stage-->em

        style 2nd-stage stroke:#6666,stroke-width:2px,stroke-dasharray: 5, 5;



Data cleaning
-----------------

The first step of the *CompeteExperiment* is to perform data cleaning with DataCleaner in Hypernets. Note that this step can not be disabled but can be adjusted with DataCleaner in the following ways：

* nan_chars： value or list, (default None), replace some characters with np.nan
* correct_object_dtype： bool, (default True), whether correct the data types
* drop_constant_columns： bool, (default True), whether drop constant columns
* drop_duplicated_columns： bool, (default False), whether delete repeated columns
* drop_idness_columns： bool, (default True), whether drop id columns
* drop_label_nan_rows： bool, (default True), whether drop rows with target values np.nan
* replace_inf_values： (default np.nan), which values to replace np.nan with
* drop_columns： list, (default None), drop which columns
* reserve_columns： list, (default None), reserve which columns when performing data cleaning
* reduce_mem_usage： bool, (default False), whether try to reduce the memory usage
* int_convert_to： bool, (default 'float'), transform int to other types，None for no transformation


If nan is represented by '\\N' in data，users can replace '\\N' back to np.nan when performing data cleaning as follows:

.. code-block:: python

    from hypergbm import make_experiment

    train_data = ...
    experiment = make_experiment(train_data, target='...',
                                data_cleaner_args={'nan_chars': r'\N'})
    ...


Feature generation
---------------------

*CompeteExperiment* is capable of performing feature generation, which can be turned on by setting *feature_generation=True* when creating experiment with *make_experiment*. There are several options:

* feature_generation_continuous_cols：list (default None)), continuous feature, inferring automatically if set as None.
* feature_generation_categories_cols：list (default None)), categorical feature, need to be set explicitly, *CompeteExperiment* can not perform automatic inference for this one.
* feature_generation_datetime_cols：list (default None), datetime feature, inferring automatically if set as None.
* feature_generation_latlong_cols：list (default None), latitude and longtitude feature, inferring automatically if set as None. 
* feature_generation_text_cols：list (default None), text feature, inferring automatically if set as None.
* feature_generation_trans_primitives：list (default None), transformations for feature generation, inferring automatically if set as None.


When feature_generation_trans_primitives=None, *CompeteExperiment* will automatically infer the types used for transforming based on the default features. Specifically, different transformations will be adopted for different types:

* continuous_cols： None, need to be set explicitly.
* categories_cols： cross_categorical.
* datetime_cols： month, week, day, hour, minute, second, weekday, is_weekend.
* latlong_cols： haversine, geohash
* text_cols：tfidf


An example code for enabling feature generation:

.. code-block:: python

    from hypergbm import make_experiment

    train_data = ...
    experiment = make_experiment(train_data,
                               feature_generation=True,
                               ...)
    ...



Please refer to [featuretools](https://docs.featuretools.com/) for more information.


Collinearity detection
---------------------------

There will often be some highly relevant features which are not informative but are more seen as noises. They are not very useful. On the contrary, the dataset will be affected by drifts of these features more heavily.


It is possible to handle these collinear features with *CompeteExperiment*. This can be simply enabled by setting *collinearity_detection=True* when creating experiment.

Example code for using collinearity detection

.. code-block:: python

    from hypergbm import make_experiment

    train_data = ...
    experiment = make_experiment(train_data, target='...', collinearity_detection=True)
    ...



Drift detection
-------------------

Concept drift is one of the major challenge for machine learning. The model will often perform worse in practice due to the fact that the data distributions will change along with time. To handle this problem, *CompeteExeriment* adopts Adversarial Validation to detect whether there is any drifted features and drop them to maintain a good performance.

To enable drift detection, one needs to set *drift_detection=True* when creating experiment and provide *test_data*.

Relevant parameters:

* drift_detection_remove_shift_variable : bool, (default=True), whether to detect the stability of every column first.
* drift_detection_variable_shift_threshold : float, (default=0.7), stability socres higher than this value will be dropped.
* drift_detection_threshold : float, (default=0.7), detecting scores higher than this value will be dropped.
* drift_detection_remove_size : float, (default=0.1), ratio of columns to be dropped.
* drift_detection_min_features : int, (default=10), the minimal number of columns to be reserved.
* drift_detection_num_folds : int, (default=5), the number of folds for cross validation. 

An code example:

.. code-block:: python

    from io import StringIO
    import pandas as pd
    from hypergbm import make_experiment
    from hypernets.tabular.datasets import dsutils

    test_data = """
    Recency,Frequency,Monetary,Time
    2,10,2500,64
    4,5,1250,23
    4,9,2250,46
    4,5,1250,23
    4,8,2000,40
    2,12,3000,82
    11,24,6000,64
    2,7,1750,46
    4,11,2750,61
    1,7,1750,57
    2,11,2750,79
    2,3,750,16
    4,5,1250,26
    2,6,1500,41
    """

    train_data = dsutils.load_blood()
    test_df = pd.read_csv(StringIO(test_data))
    experiment = make_experiment(train_data, test_data=test_df,
                                 drift_detection=True,
                                 ...)

    ...



Feature selection
--------------------------

*CompeteExperiment* evaluates the importances of features by training a model. Then it chooses the most important ones among them to continue the model training.

To enable feature selection, one needs to set *feature_selection=True* when creating experiment. Relevant parameters:

* feature_selection_strategy：str, selection strategies(default threshold), can be chose from *threshold*, *number* and *quantile*.
* feature_selection_threshold：float, (default 0.1), selection threshold when the strategy is *threshold*, features with scores higher than this threshold will be selected.
* feature_selection_quantile：float, (default 0.2), selection threshold when the strategy is *quantile*, features with scores higher than this threshold will be selected.
* feature_selection_number：int or float, (default 0.8), selection numbers when the strategy is *number*.

An example code:

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    experiment = make_experiment(train_data,
                                 feature_selection=True,
                                 feature_selection_strategy='quantile',
                                 feature_selection_quantile=0.3,
                                 ...)



UnderSampling pre-search
-----------------------------

Normally, hyperparameter optimization will utilize all training data. However, this will cost a huge amount of time for a large dataset. To alleviate this problem, one can perform a pre-search with only a part of data to try more model parameters in the same amout of time. Better parameters will then be used for training with the whole data to obtain the optimal parameters.

To enable feature selection, one needs to set *down_sample_search=True*  when creating experiment. Relevant parameters:


* down_sample_search_size：int, float(0.0~1.0) or dict (default 0.1), number of examples used for pre-search.
* down_sample_search_time_limit：int, (default early_stopping_time_limit*0.33), time limit for pre-search.
* down_sample_search_max_trials：int, (default max_trials*3), max trail numbers for pre-search.


An example code:

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    experiment = make_experiment(train_data,
                                 down_sample_search=True,
                                 down_sample_search_size=0.2,
                                 ...)


The second stage feature selection
----------------------------------------

*CompeteExperiment* supports continuing data processing with the trained model, which is officially called  *Two-stage search*. There are two types of Two-stage processings supported by *CompeteExperiment*: Two-stage feature selection and pseudo label which will be covered in the rest of this section.

In *CompeteExperiment*, the second stage feature selection is to choose models with good performances in the first stage, and use *permutation_importance* to evaluate them to give better features.

To enable the second stage feature selection, one needs to set *feature_reselection=True*  when creating experiment. Relevant parameters:

* feature_reselection_estimator_size：int, (default=10), the number of models to be used for evaluating the importances of feature (top n best models in the first stage).
* feature_reselection_strategy：str, selection strategy(default threshold), available selection strategies include *threshold*, *number*, *quantile*.
* feature_reselection_threshold：float, (default 1e-5), threshold when the selection strategy is *threshold*, importance scores higher than this values will be choosed.
* feature_reselection_quantile：float, (default 0.2),  threshold when the selection strategy is *quantile*, importance scores higher than this values will be choosed.
* feature_reselection_number：int or float, (default 0.8), the number of features to be selected when the strategy is *number*.

An example code:

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    experiment = make_experiment(train_data,
                                 feature_reselection=True,
                                 ...)



Please refer to [scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html) for more information about  *permutation_importance*.

Pseudo label
--------------

Pseudo label is a kind of semi-supervised machine learning method. It will assign labels predicted by the model trained in the first stage to some examples in test data. Then examples with higher confidence values than a threshold will be added into the trainig set to train the model again. 

To enable feature selection, one needs to set *pseudo_labeling=True* when creating experiment. Relevant parameters:

* pseudo_labeling_strategy：str, selection strategy(default threshold), available strategies include *threshold*, *number* and  *quantile*.
* pseudo_labeling_proba_threshold：float(default 0.8),  threshold when the selection strategy is *threshold*, confidence scores higher than this values will be choosed.
* pseudo_labeling_proba_quantile：float(default 0.8),  threshold when the selection strategy is *quantile*, importance scores higher than this values will be choosed.
* pseudo_labeling_sample_number：float(0.0~1.0) or int (default 0.2), the number of top features to be selcected when the strategy is *number*.
* pseudo_labeling_resplit：bool(default=False), whether split training and validation set after adding pseudo label examples. If set as False, all examples with pseudo labels will be added into training set to train the model. Otherwise, experiment will perform training set and validation set splitting for the new dataset with pseudo labels.

An example code:

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    test_data=...
    experiment = make_experiment(train_data,
                                 test_data=test_data,
                                 pseudo_labeling=True,
                                 ...)



Note: Pseudo label is only valid for classification task.
