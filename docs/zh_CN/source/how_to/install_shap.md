## 如何在centos7上安装shap?

1. 安装系统依赖
    ```bash
    yum install epel-release centos-release-scl -y  && yum clean all && yum make cache # llvm9.0 is in epel, gcc9 in scl
    yum install -y llvm9.0 llvm9.0-devel python36-devel devtoolset-9-gcc devtoolset-9-gcc-c++ make cmake 
    ```

2. 配置安装环境
    ```bash
    whereis llvm-config-9.0-64  # find your `llvm-config` path
    # llvm-config-9: /usr/bin/llvm-config-9.0-64
    
    export LLVM_CONFIG=/usr/bin/llvm-config-9.0-64  # set to your path
    scl enable devtoolset-9 bash
    ```

3. 安装shap
    ```bash
    pip -v install numpy==1.19.1  # prepare shap dependency
    pip -v install scikit-learn==0.23.1  # prepare shap dependency
    pip -v install shap==0.28.5
    ```
