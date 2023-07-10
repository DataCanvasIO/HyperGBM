
Features
======================

There are four running types of HyperGBM：

- Single node：running in a single machine and using Pandas and Numpy datatype
- Single node with NVIDIA GPU device：running in a single machine with NVIDIA GPU devices and using cuDF and cupy datatype
- Distributed with single node：running in a single machine and using Dask datatype which requires creating Dask collections before using HyperGBM
- Distributed with multi nodes：running in multiple machines and using Dask datatype which requires creating Dask collections to manage resources for multiple machines before using HyperGBM


The overview of supported features for different running types are displayed in the following table:

.. csv-table:: 
   :stub-columns: 1
   :header: ,Features,Single node,Single node with GPU,Distributed with single node,Distributed with multi nodes
   :widths: 15,40,10,10,10,10
   
   Data Cleaning,Empty characters handling,√,√,√,√
    ,Recognizing columns types automatically,√,√,√,√
    ,Columns types correction,√,√,√,√
    ,Constant columns cleaning,√,√,√,√
    ,Repeated columns cleaning,√,√,√,√
    ,Deleting examples without targets,√,√,√,√
    ,Illegal characters replacing,√,√,√,√
    ,id columns cleaning,√,√,√,√
   Dataset splitting,Splitting by ratio,√,√,√,√
    ,Adversarial validation,√,√,√,√
   Feature engineering, Feature generation,√,,√,√
     ,Feature dimension reduction,√,√,√,√
   Data preprocessing,SimpleImputer,√,√,√,√
    ,SafeOrdinalEncoder,√,√,√,√
    ,TargetEncoder,√,√,,
    ,SafeOneHotEncoder,√,√,√,√
    ,TruncatedSVD,√,√,√,√
    ,StandardScaler,√,√,√,√
    ,MinMaxScaler,√,√,√,√
    ,MaxAbsScaler,√,√,√,√
    ,RobustScaler,√,√,√,√
   Imbalanced data handling,ClassWeight,√,√,√,√
    ,"UnderSampling(Nearmiss,Tomekslinks,Random)",√,,,
    ,"OverSampling(SMOTE,ADASYN,Random)",√,,,
   Search algorithms,MCTS,√,√,√,√
    ,Evolution,√,√,√,√
    ,Random search,√,√,√,√
    ,NSGA-II,√,,,
    ,R-NSGA-II,√,,,
    ,MOEA/D,√,,,
    ,Play back,√,√,√,√
   Early stopping,time limit,√,√,√,√
    ,"no improvements are made after n trials",√,√,√,√
    ,expected_reward,√,√,√,√
    ,trail discriminator,√,√,√,√
   Modeling algorithms,XGBoost,√,√,√,√
    ,LightGBM,√,√,√,√
    ,CatBoost,√,√,√,
    ,HistGridientBoosting,√,,,
   Evaluation,Cross-Validation,√,√,√,√
    ,Train-Validation-Holdout,√,√,√,√
   Advanced,Automatica task type inference,√,√,√,√
    ,Data adaption,√,√,,
    ,Collinearity detection,√,,√,√
    ,Data drift detection,√,√,√,√
    ,Feature selection,√,√,√,√
    ,Feature selection(Two-stage),√,√,√,√
    ,Pseudo label(Two-stage),√,√,√,√
    ,Pre-searching with UnderSampling,√,√,√,√
    ,Model ensemble,√,√,√,√

