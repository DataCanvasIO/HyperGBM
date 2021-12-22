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

可选的, 如果您希望安装HyperGBM以及所有依赖包，则可通过如下形式安装：

```bash
pip install hypergbm[all]
```


### 使用Docker

HyperGBM支持在Docker容器中运行，您可在Dockerfile中通过 `pip` 安装HyperGBM，然后使用。

我们在Docker Hub中发布了一个参考镜像，可直接下载使用，该镜像中包括：

* Python3.7
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

