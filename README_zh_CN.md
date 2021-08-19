# HyperGBM
[![Python Versions](https://img.shields.io/pypi/pyversions/hypergbm.svg)](https://pypi.org/project/hypergbm)
[![Downloads](https://pepy.tech/badge/hypergbm)](https://pepy.tech/project/hypergbm)
[![PyPI Version](https://img.shields.io/pypi/v/hypergbm.svg)](https://pypi.org/project/hypergbm)

[English](README.md)

## HyperGBM是什么？
HyperGBM是一款全Pipeline自动机器学习工具，可以端到端的完整覆盖从数据清洗、预处理、特征加工和筛选以及模型选择和超参数优化的全过程，是一个真正的结构化数据AutoML工具包。

## 概览 
大部分的自动机器学习工具主要解决的是算法的超参数优化问题，而HyperGBM是将从数据清洗到算法优化整个的过程放入同一个搜索空间中统一优化。这种端到端的优化过程更接近于SDP(Sequential Decision Process)场景，因此HyperGBM采用了强化学习、蒙特卡洛树搜索等算法并且结合一个meta-leaner来更加高效的解决全Pipeline优化的问题，并且取得了非常出色的效果。

正如名字中的含义，HyperGBM中的机器学习算法使用了目前最流行的几种GBM算法（更准确的说是梯度提升树模型），目前包括XGBoost、LightGBM和Catboost三种。

HyperGBM中的优化算法和搜索空间表示技术由 [Hypernets](https://github.com/DataCanvasIO/Hypernets) 项目提供支撑。

## 安装

推荐使用`pip`命令安装HyperGBM:
```bash
pip install hypergbm
```

可选的, 如果您希望在JupyterLab中使用HyperGBM, 可通过如下命令安装HyperGBM:
```bash
pip install hypergbm[notebook]
```
可选的,  如果您希望在特征衍生时支持中文字符中, 可通过如下命令安装HyperGBM:
```bash
pip install hypergbm[zhcn]
```

可选的, 可通过如下命令安装HyperGBM所有组件及依赖包:
```bash
pip install hypergbm[all]
``` 

## 相关文章
* [HyperGBM用4记组合拳提升AutoML模型泛化能力](https://zhuanlan.zhihu.com/p/349824150)
* [HyperGBM用Adversarial Validation解决数据漂移问题](https://zhuanlan.zhihu.com/p/349432455)
* [HyperGBM的三种Early Stopping方式](https://zhuanlan.zhihu.com/p/350051541)
* [如何HyperGBM解决分类样本不均衡问题](https://zhuanlan.zhihu.com/p/350052055)
* [HyperGBM轻松实现Pseudo-labeling半监督学习](https://zhuanlan.zhihu.com/p/355419632)

## Hypernets 相关项目

* [HyperGBM](https://github.com/DataCanvasIO/HyperGBM): 一个集成了多个GBM模型的全Pipeline AutoML工具.
* [HyperDT/DeepTables](https://github.com/DataCanvasIO/DeepTables): 一个面向结构化数据的AutoDL工具.
* [HyperKeras](https://github.com/DataCanvasIO/HyperKeras): 一款为Tensorflow和Keras提供神经架构搜索和超参数优化的AutoDL工具.
* [Cooka](https://github.com/DataCanvasIO/Cooka): 一个交互式的轻量级自动机器学习系统.
* [Hypernets](https://github.com/DataCanvasIO/Hypernets): 一个通用的自动机器学习框架.

![DataCanvas AutoML Toolkit](docs/static/images/datacanvas_automl_toolkit.png)


## 参考文档

* [概览](https://hypergbm.readthedocs.io/zh_CN/latest/overview.html)
* [安装](https://hypergbm.readthedocs.io/zh_CN/latest/overview.html)
* [快速开始](https://hypergbm.readthedocs.io/zh_CN/latest/quick_start.html)
* [使用示例](https://hypergbm.readthedocs.io/zh_CN/latest/example.html)
* [How-To](https://hypergbm.readthedocs.io/zh_CN/latest/how_to.html)
* [Release Notes](https://hypergbm.readthedocs.io/zh_CN/latest/release_note.html)

## DataCanvas
HyperGBM是由数据科学平台领导厂商 [DataCanvas](https://www.datacanvas.com/) 创建的开源项目.