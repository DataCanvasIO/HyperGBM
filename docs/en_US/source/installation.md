## Installation

You can use docker or pip to install HyperGBM.

### Using pip

It requires `Python3.6` or above, and uses pip to install HyperGBM:

```shell script
pip install --upgrade pip setuptools # (optional)
pip install hypergbm
```

**Install shap(Optional)**

HyperGBM provides model interpretation based on [shap](https://github.com/slundberg/shap). Please install it if necessary. Take centos7 as an example:

1. Install system dependencies
    ```shell script
    yum install epel-release centos-release-scl -y  && yum clean all && yum make cache # llvm9.0 is in epel, gcc9 in scl
    yum install -y llvm9.0 llvm9.0-devel python36-devel devtoolset-9-gcc devtoolset-9-gcc-c++ make cmake 
    ```

2. Configure install environment
    ```shell script
    whereis llvm-config-9.0-64  # find your `llvm-config` path
    # llvm-config-9: /usr/bin/llvm-config-9.0-64
    
    export LLVM_CONFIG=/usr/bin/llvm-config-9.0-64  # set to your path
    scl enable devtoolset-9 bash
    ```

3. Install shap
    ```shell script
    pip3 -v install numpy==1.19.1  # prepare shap dependency
    pip3 -v install scikit-learn==0.23.1  # prepare shap dependency
    pip3 -v install shap==0.28.5
    ```

If it is very slow to download dependencies package of shap, consider using faster PIP and setuptools mirros. Take using the mirror provided by [aliyun](http://mirrors.aliyun.com)  as an example, Create file `~/.pip/pip.conf` with content:
```shell script
[global]
index-url = https://mirrors.aliyun.com/pypi/simple
```

Continue create file `~/.pydistutils.cfg` with content：
```shell script
[easy_install]
index_url = https://mirrors.aliyun.com/pypi/simple
```

After that reinstall HyperGBM.

### Using Docker
You can also use HypergGBM through our built-in jupyter docker image with command:
```shell script
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm-examples:1.2
```

Open browser visit site `http://<your-ip>:8888`，the token is what you have set "you-token"，it can also be empty if do not specified.
