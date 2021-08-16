## 通过命令行使用HyperGBM


HyperGBM提供了命令行工具 `hypergbm` 进行模型训练、评估和预测数据，查看命令行帮助：
```bash
hypergm -h

usage: hypergbm [-h] [--log-level LOG_LEVEL] [-error] [-warn] [-info] [-debug]
                [--verbose VERBOSE] [-v] [--enable-dask ENABLE_DASK] [-dask]
                [--overload OVERLOAD]
                {train,evaluate,predict} ...

```



`hypergbm`提供三个子命令：`train`、`evaluate`、`predict`，可通过`hypergbm <子命令> -h`获取更多信息，如：

```console
hypergbm train -h
usage: hypergbm train [-h] --train-data TRAIN_DATA [--eval-data EVAL_DATA]
                      [--test-data TEST_DATA]
                      [--train-test-split-strategy {None,adversarial_validation}]
                      [--target TARGET]
                      [--task {binary,multiclass,regression}]
                      [--max-trials MAX_TRIALS] [--reward-metric METRIC]
                      [--cv CV] [-cv] [-cv-] [--cv-num-folds NUM_FOLDS]
                      [--pos-label POS_LABEL]
                      ...
```



### 准备数据

使用命令行工具训练模型时，训练数据必须是csv或parquet格式的文件，并以 `.csv`或`.parquet`结尾；输出模型是pickle格式，以`.pkl`结尾。

以训练数据Bank Marketing为例子，可准备数据如下：

```python
from hypernets.tabular.datasets import dsutils
from sklearn.model_selection import train_test_split

df = dsutils.load_bank().head(10000)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=9527)
df_train.to_csv('bank_train.csv', index=None)
df_test.to_csv('bank_eval.csv', index=None)

df_test.pop('y')
df_test.to_csv('bank_to_pred.csv', index=None)

```

其中：
* bank_train.csv：用于模型训练
* bank_eval.csv：用于模型评价
* bank_to_pred.csv：是没有目标列的数据，用于模拟待预测数据



### 模型训练

在准备好训练数据之后，可通过命令进行模型训练：

```bash
hypergbm train --train-data bank_train.csv --target y --model-file model.pkl
```

等待命令结束，可看到模型文件：`model.pkl`
```bash
ls -l model.pkl

rw-rw-r-- 1 xx xx 9154959    17:09 model.pkl
```



### 模型评价

在模型训练之后，可利用评价数据对所得到的模型进行评价：
```bash
hypergbm evaluate --model model.pkl --data bank_eval.csv --metric f1 recall auc

{'f1': 0.7993779160186626, 'recall': 0.7099447513812155, 'auc': 0.9705420982746849}

```



### 数据预测

在模型训练之后，可利用所得到的模型进行数据预测：

```bash
hypergbm predict --model model.pkl --data bank_to_pred.csv --output bank_output.csv
```

预测结果会保存到文件`bank_output.csv` 中。



如果您希望将预测数据的某一列数据（如"id"）与预测结果一起写到输出文件，则可通过参数 `--with-data` 指定，如：

```bash
hypergbm predict --model model.pkl --data bank_to_pred.csv --output bank_output.csv --with-data id
head bank_output.csv

id,y
1563,no
124,no
218,no
463,no
...
```



如果，您希望输出文件除了包含预测结果之外，还希望有预测数据的所有列，则可将参数 `--with-data` 设置为"*"，如：

```bash
hypergbm predict --model model.pkl --data bank_to_pred.csv --output bank_output.csv --with-data '*'
head bank_output.csv

id,age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,y
1563,55,entrepreneur,married,secondary,no,204,no,no,cellular,14,jul,455,13,-1,0,unknown,no
124,51,management,single,tertiary,yes,-55,yes,no,cellular,11,may,281,2,266,6,failure,no
218,49,blue-collar,married,primary,no,305,yes,yes,telephone,10,jul,834,10,-1,0,unknown,no
463,35,blue-collar,divorced,secondary,no,3102,yes,no,cellular,20,nov,138,1,-1,0,unknown,no
2058,50,management,divorced,tertiary,no,201,yes,no,cellular,24,jul,248,1,-1,0,unknown,no
...
```


