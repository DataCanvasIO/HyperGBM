
功能特性
======================

HyperGBM有3中运行模式，分别为：

- 单机模式：在一台服务器上运行，使用Pandas和Numpy数据结构
- 单机分布式：在一台服务器上运行，使用Dask数据结构，在运行HyperGBM之前需要创建运行在单机上的Dask集群
- 多机分布式：在多台服务器上运行，使用Dask数据结构，在运行HyperGBM之前需要创建能管理多台服务器资源的Dask集群


不同运行模式的功能特性支持稍有差异，HyperGBM的功能特性清单及各种的运行模式的支持情况如下表:

.. csv-table:: 
   :stub-columns: 1
   :header: ,功能特性,单机模式,单机分布式,多机分布式
   :widths: 15,40,10,10,10
   
   数据清洗,特殊空值字符处理,√,√,√
    ,自动识别类别列,√,√,√
    ,列类型校正,√,√,√
    ,常量列清理,√,√,√
    ,重复列清理,√,√,√
    ,删除标签列为空的样本,√,√,√
    ,非法值替换,√,√,√
    ,id列清理,√,√,√
   数据集拆分,按比例拆分,√,√,√
    ,对抗验证,√,√,√
   特征工程, 特征衍生,√,√,√
     ,特征降维,√,√,√
   数据预处理,SimpleImputer,√,√,√
    ,SafeOrdinalEncoder,√,√,√
    ,SafeOneHotEncoder,√,√,√
    ,TruncatedSVD,√,√,√
    ,StandardScaler,√,√,√
    ,MinMaxScaler,√,√,√
    ,MaxAbsScaler,√,√,√
    ,RobustScaler,√,√,√
   数据不平衡处理,ClassWeight,√,√,√
    ,"降采样(Nearmiss,Tomekslinks,Random)",√,,
    ,"过采样(SMOTE,ADASYN,Random)",√,,
   搜索算法,蒙特卡洛树算法,√,√,√
    ,进化算法,√,√,√
    ,随机搜索,√,√,√
    ,历史回放,√,√,√
   提前停止策略,最大用时间提前停止,√,√,√
    ,"n次搜索都不再提升,提前停止",√,√,√
    ,expected_reward,√,√,√
    ,trail discriminator,√,√,√
   建模算法,XGBoost,√,√,√
    ,LightGBM,√,√,√
    ,CatBoost,√,√,
    ,HistGridientBoosting,√,,
   评估方法,交叉验证(Cross-Validation),√,√,√
    ,Train-Validation-Holdout验证,√,√,√
   高级特性,自动任务类型推断,√,√,√
    ,共线性特征检测,√,√,√
    ,数据漂移检测,√,√,√
    ,特征筛选,√,√,√
    ,特征筛选(二阶段),√,√,√
    ,伪标签(二阶段),√,√,√
    ,通过降采样进行预搜索,√,√,√
    ,模型融合,√,√,√

