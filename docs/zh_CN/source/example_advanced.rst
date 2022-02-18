高级应用
==========

HyperGBM *make_experiment* 所创建的是 Hypernets 的 *CompeteExeriment* 实例，*CompeteExeriment* 具备很多建模的高级特性，本章逐一介绍。

.. mermaid::

    flowchart LR
        da[数<br/>据<br/>适<br/>配]
        dc[数<br/>据<br/>清<br/>洗]
        fg[特征衍生]
        cd[共线性检测]
        dd[漂移检测]
        fs[特征筛选]
        s1[优化搜索]
        pi[二阶段<br/>特征筛选]
        pl[伪标签]
        s2[二阶段<br/>优化搜索]
        em[模<br/>型<br/>融<br/>合]
        op2[op]

        subgraph 一阶段
            direction LR
            subgraph op
                direction TB
                cd-->dd
                dd-->fs
            end
            fg-->op
            op-->s1
        end
        subgraph 二阶段
            direction LR
            subgraph op2
                direction TB
                pi-->pl
            end
            op2-->s2
        end
        da-->dc-->一阶段-->二阶段-->em

        style 二阶段 stroke:#6666,stroke-width:2px,stroke-dasharray: 5, 5;
        style 二阶段 stroke:#6666,stroke-width:2px,stroke-dasharray: 5, 5;


数据适配
---------

此步骤仅当输入数据是pandas或cudf的数据类型时生效，该步骤检测是否有足够的系统内存容纳输入数据进行建模，如果发现内存不足则尝试对输入数据进行缩减。可通过如下参数对数据适配的细节进行调整：

* data_adaption：(default True)，是否开启数据适配
* data_adaption_memory_limit：(default 0.05)，将输入数据缩减到系统可用内存的多大比例
* data_adaption_min_cols：(default 0.1)，如果需要缩减数据的话，至少保留多少列
* data_adaption_target：(default None)，此选项仅当输入数据是pandas DataFrame时生效，将此选项设置为'cuml'或'cuda'则会利用主机的CPU和MEM对数据进行缩减，然后转换为cudf.DataFrame，使得后续实验步骤在GPU中运行

数据清洗
---------

*CompeteExeriment* 利用Hypernets的DataCleaner进行数据清洗，此步骤不可禁用，但可通过参数对DataCleaner的行为进行调整，包括：

* nan_chars： value or list, (default None), 将哪些值字符替换为np.nan
* correct_object_dtype： bool, (default True), 是否尝试修正数据类型
* drop_constant_columns： bool, (default True), 是否删除常量列
* drop_duplicated_columns： bool, (default False), 是否删除重复列
* drop_idness_columns： bool, (default True), 是否删除id列
* drop_label_nan_rows： bool, (default True), 是否删除目标列为np.nan的行
* replace_inf_values： (default np.nan), 将np.inf替换为什么值
* drop_columns： list, (default None), 强行删除哪些列
* reserve_columns： list, (default None), 数据清洗时保留哪些列不变
* reduce_mem_usage： bool, (default False), 是否尝试降低对内存的需求
* int_convert_to： bool, (default 'float'), 将int列转换为何种类型，None表示不转换


调用 *make_expperiment* 时，可通过参数 *data_cleaner_args* 对DataCleaner的配置进行调整。

假设，训练数据中用字符'\\N'表示nan，希望在数据清洗阶段将其替换为np.nan，则可如下设置:

.. code-block:: python

    from hypergbm import make_experiment

    train_data = ...
    experiment = make_experiment(train_data, target='...',
                                data_cleaner_args={'nan_chars': r'\N'})
    ...


特征衍生
----------

*CompeteExeriment* 中提供了特征衍生的能力， 在通过 *make_experiment* 创建实验时设置 *feature_generation=True* 即可，与之匹配的选项包括：

* feature_generation_continuous_cols：list (default None)), 参与特征衍生的初始连续型特征，如果为None则依据训练数据的特征类型自行推断。
* feature_generation_categories_cols：list (default None)), 参与特征衍生的初始类别型特征，需要明确指定，*CompeteExeriment* 不会自行推断参与特征衍生的初始类别型特征。
* feature_generation_datetime_cols：list (default None), 参与特征衍生的初始日期型特征，如果为None则依据训练数据的特征类型自行推断。
* feature_generation_latlong_cols：list (default None), 参与特征衍生的经纬度特征，如果为None则依据训练数据自行推断。说明：经纬度特征列的数据格式必须是 *tuple(lat,long)*。
* feature_generation_text_cols：list (default None), 参与特征衍生的初始文本性特征，如果为None则依据训练数据自行推断。
* feature_generation_trans_primitives：list (default None), 用于特征衍生的算子，如果为None则依据特征类型自行推断所采用的算子。


当feature_generation_trans_primitives=None时，*CompeteExeriment* 依据参与特征衍生的初始特征自行推断所采用的算子，针对不同类型的特征采取不同算子，如下：

* continuous_cols： 无（需自行指定）。
* categories_cols： cross_categorical。
* datetime_cols： month、week、day、hour、minute、second、weekday、is_weekend。
* latlong_cols： haversine、geohash
* text_cols：tfidf


启用特征衍生的示例代码：

.. code-block:: python

    from hypergbm import make_experiment

    train_data = ...
    experiment = make_experiment(train_data,
                               feature_generation=True,
                               ...)
    ...



关于特征衍生的更多信息请参考 [featuretools](https://docs.featuretools.com/).


共线性检测
-----------------

有时训练数据中会出现一些相关度很高的特征，这些并没有提供太多的信息量，相反，数据集拥有更多的特征意味着更容易收到噪声的影响，更容易收到特征偏移的影响等等。

*CompeteExeriment* 中提供了删除发生共线性的特征的能力， 在通过 *make_experiment* 创建实验时设置 *collinearity_detection=True* 即可。

启用共线性检测的示例代码：

.. code-block:: python

    from hypergbm import make_experiment

    train_data = ...
    experiment = make_experiment(train_data, target='...', collinearity_detection=True)
    ...



漂移检测
------------

数据漂移是建模过程中的一个主要挑战。当数据的分布随着时间在不断的发现变化时，模型的表现会越来越差，*CompeteExeriment* 中引入了对抗验证的方法专门处理数据漂移问题。这个方法会自动的检测是否发生漂移，并且找出发生漂移的特征并删除他们，以保证模型在真实数据上保持良好的状态。

为了开启飘逸检测，使用 *make_experiment* 创建实验时需要设置 *drift_detection=True* （缺省）并提供测试集 *test_data* 。

漂移检测相关的参数包括：

* drift_detection_remove_shift_variable : bool, (default=True)，是否首先检查每一列数据的稳定性。
* drift_detection_variable_shift_threshold : float, (default=0.7)，稳定性指标高于该阈值的列将被删除
* drift_detection_threshold : float, (default=0.7)，检测指标高于该阈值的列将被删除。
* drift_detection_remove_size : float, (default=0.1)，每一轮检测所删除的列占总列数的比例。
* drift_detection_min_features : int, (default=10)，至少保留多少列。
* drift_detection_num_folds : int, (default=5)，在漂移检测阶段训练模型时的cv折数。

需要注意的是，启用漂移检测时必须指定 *test_data* (不包含目标列), 示例代码：

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



特征筛选
------------

进行特征筛选也是建模过程中的一个重要环节，*CompeteExeriment* 通过训练一个常规模型对训练数据的特征重要性进行评估，进而筛选出最重要的特征参与到后续模型训练中。

在通过 *make_experiment* 创建实验时设置 *feature_selection=True* 可开启特征筛选，与之匹配的选项包括：

* feature_selection_strategy：str, 筛选策略(default threshold), 可用的策略包括 *threshold*、*number* 、 *quantile*。
* feature_selection_threshold：float, (default 0.1), 当策略为 *threshold* 时的筛选阈值，重要性高于该阈值的特征会被选择。
* feature_selection_quantile：float, (default 0.2),  当策略为 *quantile* 时的筛选阈值，重要性分位高于该阈值的特征会被选择。
* feature_selection_number：int or float, (default 0.8), 当策略为 *number* 时，筛选的特征数量。

启用特征筛选的示例代码：

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    experiment = make_experiment(train_data,
                                 feature_selection=True,
                                 feature_selection_strategy='quantile',
                                 feature_selection_quantile=0.3,
                                 ...)



降采样预搜索
----------------

通常，在进行模型参数优化搜索时是使用全部训练数据进行模型训练的，当数据量较大时使用全部训练数据进行模型训练会消耗较长的时间，为此可通过降采样减少参与模型训练的数据量，进行预搜索，以便在相同的时间内尝试更多的模型参数；然后从预搜索结果中挑选表现较好的参数再利用全量数据进行训练和评估，进一步筛选最佳的模型参数。

通过 *make_experiment* 创建实验时，设置 *down_sample_search=True* 可开启预搜索，与之相关的选项包括：

* down_sample_search_size：int, float(0.0~1.0) or dict (default 0.1）, 参与预搜索的样本数量。对于分类任务，可通过dict指定每个类别数据的采样数量。
* down_sample_search_time_limit：int, (default early_stopping_time_limit*0.33), 预搜索的时间限制。
* down_sample_search_max_trials：int, (default max_trials*3), 预搜索的最大尝试次数。


启用预搜索的示例代码：

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    experiment = make_experiment(train_data,
                                 down_sample_search=True,
                                 down_sample_search_size=0.2,
                                 ...)


二阶段特征筛选
------------------

*CompeteExperiment* 支持在模型参数优化搜索之后，利用得到的模型对训练数据进行处理，然后再次进行模型参数优化搜索，即 *二阶段搜索*。目前 *CompeteExperiment* 支持的第二阶段数据处理方式包括二阶段特征筛选和伪标签，本章余下的两个小节中分别介绍。

在 *CompeteExperiment* 中，二阶段特征筛选是指从第一阶段选择若干表现较好的模型，进行 *permutation_importance* 评估，然后筛选出重要的特征。

通过 *make_experiment* 创建实验时，设置 *feature_reselection=True* 可开启二阶段特征筛选，与之相关的配置项包括：

* feature_reselection_estimator_size：int, (default=10), 用于评估特征重要性的模型数量（在一阶段搜索中表现最好的n个模型）。
* feature_reselection_strategy：str, 筛选策略(default threshold), 可用的策略包括 *threshold*、*number* 、 *quantile*。
* feature_reselection_threshold：float, (default 1e-5), 当策略为 *threshold* 时的筛选阈值，重要性高于该阈值的特征会被选择。
* feature_reselection_quantile：float, (default 0.2),  当策略为 *quantile* 时的筛选阈值，重要性分位高于该阈值的特征会被选择。
* feature_reselection_number：int or float, (default 0.8), 当策略为 *number* 时，筛选的特征数量。

启用二阶段特征筛选的示例代码：

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    experiment = make_experiment(train_data,
                                 feature_reselection=True,
                                 ...)



关于 *permutation_importance* 的更多信息请参考 [scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html)


伪标签
-----------

伪标签是一种半监督学习技术，将测试集中未观测标签列的特征数据通过一阶段训练的模型预测标签后，将置信度高于一定阈值的样本添加到训练数据中重新训练模型，有时候可以进一步提升模型在新数据上的拟合效果。

在通过 *make_experiment* 创建实验时设置 *pseudo_labeling=True* 可开启伪标签训练，与之相关的配置项包括：

* pseudo_labeling_strategy：str, 筛选策略(default threshold), 可用的策略包括 *threshold*、*number* 、 *quantile*。
* pseudo_labeling_proba_threshold：float(default 0.8), 当策略为 *threshold* 时的筛选阈值，置信度高于该阈值的样本会被选择。
* pseudo_labeling_proba_quantile：float(default 0.8), 当策略为 *quantile* 时的筛选阈值，置信度分位高于该阈值的样本会被选择。
* pseudo_labeling_sample_number：float(0.0~1.0) or int (default 0.2), 当策略为 *number* 时，对样本按置信度高低排序后选择的样本数（top n）。
* pseudo_labeling_resplit：bool(default=False), 添加新的伪标签数据后是否重新分割训练集和评估集. 如果为False, 直接把所有伪标签数据添加到训练集中重新训练模型，否则把训练集、评估集及伪标签数据合并后重新分割。

启用伪标签技术的示例代码：

.. code-block:: python

    from hypergbm import make_experiment

    train_data=...
    test_data=...
    experiment = make_experiment(train_data,
                                 test_data=test_data,
                                 pseudo_labeling=True,
                                 ...)



说明： 伪标签 仅对分类任务有效。

