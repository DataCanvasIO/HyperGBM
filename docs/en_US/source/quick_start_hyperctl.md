# HyperGBM：Job management with Hyperctl

[Hyperctl](https://hypernets.readthedocs.io/en/latest/hyperctl.html) is a general multi-job management tool, which includes but not limit to training, testing and comparison. This section will introduce how to use hyperctl to manage the HyperGBM training tasks. 

Firstly, use the python script `hypergbm/job.py` or `hypergbm/gpu_job.py`(for using gpu to train) provided by HyperGBM to read all parameters of the job of hyperctl. Then configure these parameters and transfer them to the function `hypergbm.make_experiment` to create an experiment. Lastly, the experiment is executed.


## Example: Use Hyperctl to train a HyperGBM classification model 

- Create an directory and all operation will be executed within this directory
```shell
mkdir /tmp/hyperctl-example
cd /tmp/hyperctl-example
```

- Download the training dataset [heart-disease-uci](https://www.kaggle.com/ronitf/heart-disease-uci)：

```shell
# curl -O heart-disease-uci.csv https://raw.githubusercontent.com/DataCanvasIO/Hypernets/master/hypernets/tabular/datasets/heart-disease-uci.csv
python -c "from hypernets.tabular.datasets.dsutils import load_heart_disease_uci;load_heart_disease_uci().to_csv('heart-disease-uci.csv', index=False)"
```

- Create the hyperctl job configuration file `batch.json`. We could see that it includes the parameters: `train_data`,`target`,`log_level` and `run_kwargs`:
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
Please note: 
1. In the configuration file, parameters like 'train_data','eval_data' and 'test_data' should be replaced by the corresponding file path. And the file format coule be 'csv' or 'parquet'.
2. In the configuration file, sub parameters within 'run_kwargs' are used to configure the experiment. They will be transfered to the function `hypernets.experiment.compete.CompeteExperiment.run`



- Execute the configured job：
```shell
hyperctl run --config ./batch.json
```
