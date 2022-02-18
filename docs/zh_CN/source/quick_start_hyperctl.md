# 与 Hyperctl 整合

Hyperctl 是一个批量任务管理工具，可以使用它来运行HyperGBM的训练实验。
HyperGBM提供了一个脚本`hypergbm/job.py`用来读取hyperctl中的任务参数并创建实验来运行。
它把读取的参数送给方法`hypergbm.make_experiment`去构建实验并运行实验。

值得注意的点： 
1. `hypergbm.make_experiment`的参数`train_data`, `eval_data`, `test_data`需要的是DataFrame数据对象，在配置文件中
我们需要把这几个参数替换成数据文件地址，数据文件可以是`csv`或者`parquet`格式。
2. 在job中配置的参数`run_kwargs` 是用来运行实验而非构建实验，也就是会送给方法`hypernets.experiment.compete.CompeteExperiment.run`

**使用Hyperctl训练HyerGBM实验**

这一节演示如何使用Hyperctl调用HyperGBM训练一个二分类数据集。

首先创建一个目录，后续的操作在此目录中进行：
```shell
mkdir /tmp/hyperctl-example
cd /tmp/hyperctl-example
```

下载训练用到的数据集[heart-disease-uci](https://www.kaggle.com/ronitf/heart-disease-uci) 文件：

```shell
# curl -O heart-disease-uci.csv https://raw.githubusercontent.com/DataCanvasIO/Hypernets/master/hypernets/tabular/datasets/heart-disease-uci.csv
python -c "from hypernets.tabular.datasets.dsutils import load_heart_disease_uci;load_heart_disease_uci().to_csv('heart-disease-uci.csv', index=False)"
```

创建hyperclt任务配置文件 `batch.json` 并写入以下配置:
```json
{
    "jobs": [
        {
            "params": {
                "train_data": "/tmp/hyperctl-example/heart-disease-uci.csv",
                "target": "target",
                "log_level": "info",
                "run_kwargs": {
                  "max_trials": 10
                }
            },
            "execution": {
                "command": "python -m hypergbm.job"
            }
        }
    ]
}
```

运行任务：
```shell
hyperctl run --config ./batch.json
```
