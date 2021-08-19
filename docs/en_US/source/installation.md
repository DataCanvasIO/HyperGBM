## Installing HyperGBM
We recommend installing HyperGBM with `pip`. Installing and using HyperGBM in a Docker container are also possible if you have a Docker environment.



### pip
Python version 3.6 or above is necessary before installing HyperGBM. Here is how to use pip to install HyperGBM:
```bash
pip install hypergbm
```

Note that HyperGBM and its necessary packages will be installed at this time. If you want to install HyperGBM along with dependent packages such as `shap`, the following way is recommended:

```bash
pip install hypergbm[all]
```



### Docker

It is possible to use HyperGBM in a Docker container. To do this, users can install HyperGBM with `pip` in the Dockerfile. We also publish a mirror image in Docker Hub which can be downloaded directly and includes the following components:

* Python3.7
* HyperGBM and its dependent packages
* JupyterLab


Download the mirror image:
```bash
docker pull datacanvas/hypergbm
```

Use the mirror image:
```bash
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm
```

Then one can visit `http://<your-ip>:8888` in the browser and type in the default token to start.