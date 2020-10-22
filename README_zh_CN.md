# HyperGBM
[![Python Versions](https://img.shields.io/pypi/pyversions/hypergbm.svg)](https://pypi.org/project/hypergbm)
[![Downloads](https://pepy.tech/badge/hypergbm)](https://pepy.tech/project/hypergbm)
[![PyPI Version](https://img.shields.io/pypi/v/hypergbm.svg)](https://pypi.org/project/hypergbm)

[English](README.md)

## HyperGBM是什么？
HyperGBM是一款全Pipeline自动机器学习工具，可以端到端的完整覆盖从数据清洗、预处理、特征加工和筛选以及模型选择和超参数优化的全过程，是一个真正的结构化数据AutoML工具包。

## 概览 
大部分的自动机器学习工具主要解决的是算法的超参数优化问题，而HyperGBM是将从数据清洗到算法优化整个的过程放入同一个搜索空间中统一优化。这种端到端的优化过程更接近于SDP(Sequential Decision Process)场景，因此HyperGBM采用了强化学习、蒙特卡洛树搜索等算法并且结合一个meta-leaner来更加高效的接近全Pipeline优化的问题，并且取得了非常出色的效果。

正如名字中的含义，HyperGBM中的机器学习算法部分采用了目前最流行的几种GBM算法（更准确的说是梯度提升树模型），目前包括XGBoost、LightGBM和Catboost三种。

HyperGBM中的搜索空间表示和基础的搜索算法由 [Hypernets](https://github.com/DataCanvasIO/Hypernets) 项目提供。

## 安装
```shell script
pip install hypergbm
```

## Hypernets 相关项目

* [HyperGBM](https://github.com/DataCanvasIO/HyperGBM): 一个集成了多个GBM模型的全Pipeline AutoML工具.
* [HyperDT/DeepTables](https://github.com/DataCanvasIO/DeepTables): 一个面向结构化数据的AutoDL工具.
* [HyperKeras](https://github.com/DataCanvasIO/HyperKeras): 一款为Tensorflow和Keras提供神经架构搜索和超参数优化的AutoDL工具.
* [Cooka](https://github.com/DataCanvasIO/Cooka): 一个交互式的轻量级自动机器学习系统.
* [Hypernets](https://github.com/DataCanvasIO/Hypernets): 一个通用的自动机器学习框架.

![DataCanvas AutoML Toolkit](docs/source/images/datacanvas_automl_toolkit.png)


## DataCanvas
HyperGBM是由数据科学平台领导厂商 [DataCanvas](https://www.datacanvas.com/) 创建的开源项目.