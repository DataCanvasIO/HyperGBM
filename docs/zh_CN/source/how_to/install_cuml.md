## 如何安装cuML及相关包?

cuML是英伟达 [RAPIDS](https://rapids.ai/) 框架中的一部分 ，使用GPU进行模型训练时，HyperGBM利用cuML（及cuDF）进行数据预处理。依据RAPIDS官方资料，可以通过conda、docker或源代码安装/使用RAPIDS（或特定组件）。推荐使用conda安装cuML和HyperGBM。

1. 需要考虑的因素：
	* 操作系统：目前RAPIDS只支持Linux，您可以从Ubuntu、CentOS、RHEL中选择
	* CUDA版本：目前RAPIDS支持的CUDA版本包括11.0、11.2、11.4、11.5（请从RAPIDS官网获取最新信息），请确保您的系统中已经安装了能够被支持的CUDA驱动
	* RAPIDS版本：推荐使用21.10或以上的稳定版本
	* Python版本：推荐使用Python3.8
	
2. 安装cuML及HyperGBM

    通过conda命令安装 cuML和 HyperGBM，示例:

    ```bash
    conda create -n hypergbm_with_cuml -c rapidsai -c nvidia -c conda-forge python=3.8 cudatoolkit=11.2 cudf=21.12 cuml=21.12  hypergbm 
    ```
    
    注意，请将示例命令中的软件版本替换为适您的选项。



更多关于RAPIDS的信息请参考RAPIDS官方网站 [https://rapids.ai/](https://rapids.ai/)。

