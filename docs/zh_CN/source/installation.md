## 安装教程
可以使用Docker或者pip来安装HyperGBM。

### 使用pip
它需要`Python3.6`或以上版本, 使用pip安装HyperGBM：
```shell script
pip install --upgrade pip setuptools # (optional)
pip install hypergbm 
```

**安装shap(可选)**

hypergbm提供了基于[shap](https://github.com/slundberg/shap) 模型解释功能，如果需要请安装, 以centos7为例：

1. 安装系统依赖
    ```shell script
    yum install epel-release centos-release-scl -y  && yum clean all && yum make cache # llvm9.0 is in epel, gcc9 in scl
    yum install -y llvm9.0 llvm9.0-devel python36-devel devtoolset-9-gcc devtoolset-9-gcc-c++ make cmake 
    ```

2. 配置安装环境
    ```shell script
    whereis llvm-config-9.0-64  # find your `llvm-config` path
    # llvm-config-9: /usr/bin/llvm-config-9.0-64
    
    export LLVM_CONFIG=/usr/bin/llvm-config-9.0-64  # set to your path
    scl enable devtoolset-9 bash
    ```

3. 安装shap
    ```shell script
    pip3 -v install numpy==1.19.1  # prepare shap dependency
    pip3 -v install scikit-learn==0.23.1  # prepare shap dependency
    pip3 -v install shap==0.28.5
    ```

如果下载shap的依赖包很慢，考虑使用更快的pip和setuptools的镜像源, 以使用[aliyun](http://mirrors.aliyun.com)提供的镜像源为例，
创建文件`~/.pip/pip.conf` 内容为：
```shell script
[global]
index-url = https://mirrors.aliyun.com/pypi/simple
```

继续创建文件`~/.pydistutils.cfg`内容为：
```shell script
[easy_install]
index_url = https://mirrors.aliyun.com/pypi/simple
```
然后再安装HyperGBM。

### 使用Docker
您也可以通过我们提供的Docker镜像内置的Jupyter来使用HyperGBM:
```shell script
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm-examples:1.2
```

打开浏览器，访问`http://<your-ip>:8888`，所需要的Token即为您设置的"your-token"，如果没设置则为空。
