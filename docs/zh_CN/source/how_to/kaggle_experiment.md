## 如何在Kaggle中使用HyperGBM?



Kaggle中提供了在线notebook方便用户进行数据处理和模型训练，您可以在Kaggle的notebook中安装并使用HyperGBM。建议使用`pip` 在kaggle的notebook安装HyperGBM。

需要注意的是，由于kaggle的notebook中默认安装了许多软件，其中个别软件的依赖与HyperGBM'的依赖有冲突，建议 在安装HyperGBM之前先卸载这些软件。您可首先尝试安装HyperGBM，然后根据`pip`命令 的提示信息检查哪些软件与HyperGBM有冲突，重启notebook并删除存在冲突的软件后再试尝试安装HyperGBM。



实际使用示例可参考 [notebook_hypergbm_bank_marketing_kaggle](https://www.kaggle.com/tele6224/notebook-hypergbm-bank-marking)：

