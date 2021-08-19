Version 0.2.3
=====================

We add the following new features to this version:

* Data cleaning
  - Support automatically recognizing categorical columns among features with numerical datatypes
  - Support performing data cleaning with several specific columns reserved

* Feature generation
  - Support datatime, text and Latitude and Longitude features
  - Support distributed training

* Modelling algorithms
  - XGBoost：Change distributed training from `dask_xgboost` to `xgboost.dask` to be compatible with official website of XGBoost
  - LightGBM：Support distributed trianing for more machines

* Model training
  - Support reproducing the searching process
  - Support searching with low fidelity
  - Predicting learning curves based on statistical information
  - Support hyperparameter optimizing without making modification
  - Time limit of EarlyStopping is now adjusted to the whole experiment life-cycle
  - Support defining pos_label
  - eval-set supports Dask dataset for distributed training
  - Optimizing the cache strategy for model training

* Search algorithms
  - Add GridSearch algorithm
  - Add Playback algorithm

* Advanced Features
  - Add feature selection with various strategies for the first stage
  - Feature selection for the second stage now supports more strategies
  - Pseudo-label supports various data selection strategies and multi-class classification
  - Optimizing performance of concepts drift handling
  - Add cache mechanism during processing of advanced features

* Visualization
  - Experiment information visualization
  - Training process visualization
  
* Command Line tool
  - Most features of experiments for model training are now supported by command line tools
  - Support model evaluating
  - Support model predicting