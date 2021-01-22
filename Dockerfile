FROM centos:7

USER root
ENV LANG C.UTF-8
ENV NotebookToken ''

# llvm9 in epel, need by compile shap;
# gcc9 in centos-release-scl; gcc9 is need by compile xgboost
# `scl enable devtoolset-9 bash` to enable gcc9
# `echo "source /opt/rh/devtoolset-9/enable" >> /etc/profile` is equals to `scl enable devtoolset-9 bash`
# xgboost need cmake3

RUN  yum install epel-release  centos-release-scl -y \
     && yum clean all \
     && yum makecache \
     && yum install -y llvm9.0 llvm9.0-devel python36-devel devtoolset-9-gcc devtoolset-9-gcc-c++ make cmake3 \
	 && ln -s /opt/rh/devtoolset-9/root/usr/bin/gcc /usr/local/bin \
	 && ln -s /opt/rh/devtoolset-9/root/usr/bin/g++ /usr/local/bin \
	 && ln -s /usr/bin/cmake3 /usr/bin/cmake \
     && yum install -y graphviz \
     && yum install -y git

ENV LLVM_CONFIG /usr/bin/llvm-config-9.0-64

RUN mkdir -p /root/.pip \
    && echo -e "[global]\n\
index-url = https://mirrors.aliyun.com/pypi/simple" > /root/.pip/pip.conf

# For install shap
RUN echo -e "[easy_install]\n\
index_url = https://mirrors.aliyun.com/pypi/simple" > /root/.pydistutils.cfg

RUN mkdir -p /opt/datacanvas

RUN git clone https://github.com/DataCanvasIO/hypergbm.git  /opt/datacanvas/hypergbm

RUN pip3 install jupyter # Docker Image deps

RUN pip3 -v install numpy==1.19.1 scikit-learn==0.23.1  # Prepare for shap
RUN pip3 -v install shap==0.28.5 matplotlib  # Prepare for hypergbm

RUN cd /opt/datacanvas/hypergbm && pip3 install .

EXPOSE 8888

CMD [ "bash", "-c", "/usr/local/bin/jupyter notebook --notebook-dir=/opt/datacanvas/hypergbm/hypergbm/examples  --allow-root --NotebookApp.token=$NotebookToken"]

# docker run --rm  -p 8888:8888 -e NotebookToken=your-token datacanvas/hypergbm-example
