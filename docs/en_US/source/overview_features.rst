
Features
======================

There are three types of running HyperGBM：

- Single machine：running in a single machine and using Pandas and Numpy datatypes
- Distributed trianing in a single machine：running in a single machine and using Dask datatype which requires creating Dask collection before using HyperGBM
- Distributed training in more machines：running in multiple machines and using Dask datatype which requires creating Dask collection to manage resources for mutiple machines before using HyperGBM


The supported features are different for different running types as in the following table:

.. csv-table:: 
   :stub-columns: 1
   :header: ,Features,Single Machine,Distributed training in a single machine,Distributed training in more machines
   :widths: 15,40,10,10,10
   
   Data Cleaning,Empty characters handling,√,√,√
    ,Recognizing columns types automatically,√,√,√
    ,Columns types correction,√,√,√
    ,Constant columns cleaning,√,√,√
    ,Repeated columns cleaning,√,√,√
    ,Deleating examples without targets,√,√,√
    ,Illegal characters replacing,√,√,√
    ,id columns cleaning,√,√,√
   Dataset splitting,Splitting by ratio,√,√,√
    ,Adversarial validation,√,√,√
   Feature engineering, Feature generation,√,√,√
     ,Feature dimension reduction,√,√,√
   Data preprocessing,SimpleImputer,√,√,√
    ,SafeOrdinalEncoder,√,√,√
    ,SafeOneHotEncoder,√,√,√
    ,TruncatedSVD,√,√,√
    ,StandardScaler,√,√,√
    ,MinMaxScaler,√,√,√
    ,MaxAbsScaler,√,√,√
    ,RobustScaler,√,√,√
   Imbalanced data handling,ClassWeight,√,√,√
    ,"UnderSampling(Nearmiss,Tomekslinks,Random)",√,,
    ,"OverSampling(SMOTE,ADASYN,Random)",√,,
   Search algorithms,MCTS,√,√,√
    ,Evolution,√,√,√
    ,Random search,√,√,√
    ,Play back,√,√,√
   Early stopping,Early stopping with time limit,√,√,√
    ,"Early stopping when no improvements are made after n times searching",√,√,√
    ,expected_reward,√,√,√
    ,trail discriminator,√,√,√
   Modeling algorithms,XGBoost,√,√,√
    ,LightGBM,√,√,√
    ,CatBoost,√,√,
    ,HistGridientBoosting,√,,
   Evaluation,Cross-Validation,√,√,√
    ,Train-Validation-Holdout,√,√,√
   Advanced features,Automatica task type inference,√,√,√
    ,Colinear features detection,√,√,√
    ,Data drift detection,√,√,√
    ,Feature selection,√,√,√
    ,Feature selection(Two-stage),√,√,√
    ,Pseudo lable(Two-stage),√,√,√
    ,Pre-searching with UnderSampling,√,√,√
    ,Model ensemble,√,√,√

