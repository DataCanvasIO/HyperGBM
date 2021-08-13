## Handling Imbalanced Data

Imbalanced data problem is one of the most often encountered challanges in practice, which will usually leads to not ideal models. To alleviate this problem, HyperGBM supports two solutions as follows:


### Adopt ClassWeight

When building the model such as LightGBM, one first calculates the data distributions assign different weights to different classes according to their distributions when computing loss. To enable ClassWeight algorithm, one can simply set the parameter ``class_balancing='ClassWeight'` when using `make_experiment`. 

```python
from hypergbm import make_experiment

train_data = ...
experiment = make_experiment(train_data,
                             class_balancing='ClassWeight',
                             ...)


```


### Under Sampling and Over Sampling

The most common approach to handle the data imbalance problem is to modify the data distribution to get a more balanced dataset then train the model with the modified dataset. Currently, HyperGBM supports several resampling strategies including *RandomOverSampler*, *SMOTE*, *ADASYN*, *RandomUnderSampler*, *NearMiss*, *TomekLinks*, and *EditedNearestNeighbours*. To enable different sampling method, one only needs to set `class_balancing='<selected strategy>'` when using `make_experiment`. Pleas refer to the following example:

为了利用欠采样或过采样技建模，在调用`make_experiment`时，设置参数`class_balancing=‘<采用策略>’`即可，示例如下：

```python
from hypergbm import make_experiment

train_data = ...
experiment = make_experiment(train_data,
                             class_balancing='SMOTE',
                             ...)


```

For more information regarding these sampling methods, please see [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn).