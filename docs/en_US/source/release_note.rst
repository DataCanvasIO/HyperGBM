Release Note
=====================

Feature list
---------------------

This release(v2.1) with follow new features:

Feature engineering	
  - Feature generation
  - Feature selection

Data clean
  - Special empty value handing 
  - Correct data type
  - Id-ness features cleanup
  - Duplicate features cleanup
  - Empty label rows cleanup
  - Illegal values replacement
  - Constant features cleanup
  - Collinearity features cleanup

Data set split
  - Adversarial validation

Modeling algorithms
  - XGBoost
  - Catboost
  - LightGBM
  - HistGridientBoosting

Training 
  - Task inference
  - Command-line tools

Evaluation strategies:
  - Cross-validation
  - Train-Validation-Holdout

Search strategies
  - Monte Carlo Tree Search
  - Evolution
  - Random search

Imbalance data 
  - Class Weight
  - Under-Samping 
    - Near miss
    - Tomeks links 
    - Random
  - Over-Samping
    - SMOTE
    - ADASYN
    - Random

Early stopping strategies
  - max_no_improvement_trials
  - time_limit
  - expected_reward

Advance features:
  - Two stage search
    - Pseudo label
    - Feature selection
  - Concept drift handling
  - Ensemble




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
