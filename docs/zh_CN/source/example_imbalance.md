## 处理不平衡数据

数据不平衡也是业务建模过程中的一个重要挑战，数据的不平衡往往会导致建模效果不理想。HyperGBM 在建模时支持两种类型的不平衡数据处理方式，下面分别介绍。


### 利用ClassWeight建模

在利用底层建模算法（如lightgbm等）建模时，首先计算样本分布比例，然后利用底层算法对模型进行优化。
 


为了利用ClassWeight建模，在调用`make_experiment`时，设置参数`class_balancing=‘ClassWeight’`即可，示例如下：


```python
from hypergbm import make_experiment

train_data = ...
experiment = make_experiment(train_data,
                             class_balancing='ClassWeight',
                             ...)


```


### 欠采样或过采样

在建模之前，通过欠采样或过采样技术调整样本数据，然后再利用调整后的数据进行建模，以得到表现较好的模型。目前支持的采样策略包括：*RandomOverSampler* 、*SMOTE* 、*ADASYN* 、*RandomUnderSampler* 、*NearMiss* 、*TomekLinks* 、*EditedNearestNeighbours*。

为了利用欠采样或过采样技建模，在调用`make_experiment`时，设置参数`class_balancing=‘<采用策略>’`即可，示例如下：


```python
from hypergbm import make_experiment

train_data = ...
experiment = make_experiment(train_data,
                             class_balancing='SMOTE',
                             ...)


```
 

 关于欠采样或过采样技术的更多信息，请参考 [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)。
 
 