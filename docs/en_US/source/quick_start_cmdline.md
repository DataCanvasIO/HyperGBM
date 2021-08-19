## Use HyperGBM with Command Line


HyperGBM offers command line tool `hypergbm` to perform model training, evaluation and prediction. The following code enables the user to view command line help:
```bash
hypergm -h

usage: hypergbm [-h] [--log-level LOG_LEVEL] [-error] [-warn] [-info] [-debug]
                [--verbose VERBOSE] [-v] [--enable-dask ENABLE_DASK] [-dask]
                [--overload OVERLOAD]
                {train,evaluate,predict} ...

```



`hypergbm` offers three commands: `train`, `evaluate` and `predict`. To get more information, one can use `hypergbm <command> -h`:
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



### Prepare the Data

When training model with command line, the training data must be saved in a file of form of csv or parque. The returned model is in the form of pickle whoes file ends with `.pkl`.

For an example of training Bank Marketing data, one can prepare the data as follows:

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

where
* bank_train.csv is used for training
* bank_eval.csv is used for evaluating the model
* bank_to_pred.csv is data without targets for predicting



### Train the Model

After preparing the data, one can also perform model training with command line:

```bash
hypergbm train --train-data bank_train.csv --target y --model-file model.pkl
```

one will see `model.pkl` after this process
```bash
ls -l model.pkl

rw-rw-r-- 1 xx xx 9154959    17:09 model.pkl
```



### Evaluate the Model
The trained model can be evaluated with the evaluation data:
```bash
hypergbm evaluate --model model.pkl --data bank_eval.csv --metric f1 recall auc

{'f1': 0.7993779160186626, 'recall': 0.7099447513812155, 'auc': 0.9705420982746849}

```



### Predict the Test Data

The trained model can be used for predicting a given data as follows:

```bash
hypergbm predict --model model.pkl --data bank_to_pred.csv --output bank_output.csv
```

where the predicting result will be saved to `bank_output.csv`.



To add other columns of your predicted data to the above file, one can use the parameter `--with-data` explicitly:

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



Furthermore, including all columns of the test data besides the predicting results to the file `bank_output.csv` can be done by setting `--with-data` as "*":

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


