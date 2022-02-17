## Installation Guide
We recommend installing HyperGBM with `conda` or `pip`. Installing and using HyperGBM in a Docker container are also possible if you have a Docker environment.

Python version 3.6 or above is necessary to install HyperGBM.

### Using Conda

Install HyperGBM with `conda` from the channel *conda-forge*:

```bash
conda install -c conda-forge hypergbm
```

On the Windows system, recommend install pyarrow(required by hypernets) 4.0 or earlier version with HyperGBM:

```bash
conda install -c conda-forge hypergbm "pyarrow<=4.0"
```


### Using Pip
Install HyperGBM with different `pip` options:

* Typical installation:
```bash
pip install hypergbm
```

* To run HyperGBM in JupyterLab/Jupyter notebook, install with command:
```bash
pip install hypergbm[notebook]
```

* To support dataset with simplified Chinese in feature generation,
  * Install `jieba` package before running HyperGBM. 
  * OR install with command:
```bash
pip install hypergbm[zhcn]
```

* Install all above with one command:
```bash
pip install hypergbm[all]
```


### Using Docker

It is possible to use HyperGBM in a Docker container. To do this, users can install HyperGBM with `pip` in the Dockerfile. We also publish a mirror image in Docker Hub which can be downloaded directly and includes the following components:

* Python3.7
* HyperGBM and its dependent packages
* JupyterLab


Download the mirror image:
```bash
docker pull datacanvas/hypergbm
```

Use the mirror image:
```bash
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm
```

Then one can visit `http://<your-ip>:8888` in the browser and type in the default token to start.


### 安装GPU加速的依赖包

* cuML and cuDF

HyperGBM利用NVIDIA RAPIDS中的 cuML 和 cuDF对数据处理进行加速，所以如果要利用GPU对HyperGBM进行加速的话，您需要在运行HyperGBM之前安装这两个软件。这两个软件的安装方法请参考 NVIDIA RAPIDS官网 [https://rapids.ai/start.html#get-rapids](https://rapids.ai/start.html#get-rapids) for more details.

* 支持 GPU 的 LightGBM 

通过默认方式安装的LightGBM并不能利用GPU进行训练，所以你需要自己编译和安装能够支持GPU的LightGBM。建议在安装HyperGBM之前就安装好支持GPU的LightGMB.  关于如何在LightGBM中开启GPU支持的方法请参考其官网  [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html) 。

* 支持GPU的 XGBoost 和 CatBoost

通过默认方式安装的 XGBoost 和 CatBoost已经内置了对GPU的支持，所以您不要再做其他的动作。但是，如果您希望自己手动从源代码中编译和安装这两个软件的话，请开启他们支持GPU的选项。

### Requirements for GPU acceleration

* cuML and cuDF

HyperGBM accelerate data processing with NVIDIA RAPIDS cuML and cuDF, please install them before running HyperGBM on GPU. See [https://rapids.ai/start.html#get-rapids](https://rapids.ai/start.html#get-rapids) for more details.

* LightGBM with GPU support

Default installation of LightGBM does not support GPU training, please build and install LightGBM with GPU support before installing HyperGBM.  See [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html) for more details.

* XGBoost and CatBoost with GPU support

The default installation of XGBoost and CatBoost can train model on GPU, so you don't need to do any action for them to run HyperGBM on GPU. If you build them form source code yourself, please enable GPU support.
