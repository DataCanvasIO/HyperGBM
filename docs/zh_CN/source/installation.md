## 安装HyperGBM
推荐使用`conda`或`pip`命令来安装HyperGBM（请提前准备好Python3.6以上版本的环境）；如果您有Docker环境，也可以在Docker容器中安装并运行HyperGBM。

### 使用Conda

可以从 *conda-forge* 安装HyperGBM:

```bash
conda install -c conda-forge hypergbm
```

对于Windows系统, 安装HyperGBM时建议将pyarrow(hypernets的依赖)的版本限制在4.0或更早:

```bash
conda install -c conda-forge hypergbm "pyarrow<=4.0"
```

### 使用Pip

基本的，使用如下`pip`命令安装HyperGBM:
```bash
pip install hypergbm
```

可选的, 如果您希望在JupyterLab中使用HyperGBM, 可通过如下命令安装HyperGBM:
```bash
pip install hypergbm[notebook]
```
可选的, 如果您希望在特征衍生时支持中文字符中, 可通过如下命令安装HyperGBM:
```bash
pip install hypergbm[zhcn]
```

可选的, 如果您希望使用基于Web的实验可视化，可通过如下命令安装HyperGBM:
```bash
pip install hypergbm[board]
```

可选的, 如果您希望安装HyperGBM以及所有依赖包，则可通过如下形式安装：

```bash
pip install hypergbm[all]
```


### 使用Docker

HyperGBM支持在Docker容器中运行，您可在Dockerfile中通过 `pip` 安装HyperGBM，然后使用。

我们在Docker Hub中发布了一个参考镜像，可直接下载使用，该镜像中包括：

* Python 3.8
* HyperGBM及其依赖包
* JupyterLab



下载镜像：
```bash
docker pull datacanvas/hypergbm
```

运行镜像:

```bash
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm
```

打开浏览器，访问`http://<your-ip>:8888`，输入您设置的token即可开始使用。


### 安装 GPU 加速的依赖包

* cuML and cuDF

HyperGBM利用NVIDIA RAPIDS中的 cuML 和 cuDF对数据处理进行加速，所以如果要利用GPU对HyperGBM进行加速的话，您需要在运行HyperGBM之前安装这两个软件。这两个软件的安装方法请参考 NVIDIA RAPIDS官网 [https://rapids.ai/start.html#get-rapids](https://rapids.ai/start.html#get-rapids) .

* 支持 GPU 的 LightGBM 

通过默认方式安装的LightGBM并不能利用GPU进行训练，所以您需要自己编译和安装能够支持GPU的LightGBM。建议您在安装HyperGBM之前就安装好支持GPU的LightGMB，HyperGBM 安装程序会复用已经存在的软件包。关于如何在LightGBM中开启GPU支持的方法请参考其官网文档 [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html) 。

* 支持GPU的 XGBoost 和 CatBoost

通过默认方式安装的 XGBoost 和 CatBoost已经内置了对GPU的支持，所以您不要再做其他的动作。但是，如果您希望自己手动从源代码中编译和安装这两个软件的话，请开启他们支持GPU的选项。
