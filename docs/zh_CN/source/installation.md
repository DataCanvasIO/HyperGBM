## 安装教程
可以使用Docker或者pip来安装HyperGBM。

### 使用pip
它需要`Python3.6`或以上版本, 使用pip安装HyperGBM：
```shell script
pip install --upgrade pip setuptools # (optional)
pip install hypergbm 
```

**安装shap(可选)**

HyperGBM提供了基于[shap](https://github.com/slundberg/shap) 模型解释功能，如果需要请参照[文档](how_to/install_shap.md)安装。

### 使用Docker
您也可以通过我们提供的Docker镜像内置的Jupyter来使用HyperGBM:
```shell script
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm:0.2.0
```

打开浏览器，访问`http://<your-ip>:8888`，所需要的Token即为您设置的"your-token"，如果没设置则为空。
