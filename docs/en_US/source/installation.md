## Installation Guide
We recommend installing HyperGBM with `conda` or `pip`. It's also possible to install and use HyperGBM in a Docker container if you have a Docker environment.

As for software, Python version 3.6 or above is necessary to install HyperGBM.

### Using Conda

Install HyperGBM with `conda` from the channel *conda-forge*:

```bash
conda install -c conda-forge hypergbm
```

On the Windows system, recommend install pyarrow(required by hypernets) 4.0 or earlier version with HyperGBM:

```bash
conda install -c conda-forge hypergbm "pyarrow<=4.0"
```


### Using Pip
Install HyperGBM with different `pip` options:

* Typical installation:
```bash
pip install hypergbm
```

* To run HyperGBM in JupyterLab/Jupyter notebook, install with command:
```bash
pip install hypergbm[notebook]
```

* To support experiment visualization base on web, install with command:
```bash
pip install hypergbm[board]
```

* To run HyperGBM in distributed Dask cluster, install with command:
```bash
pip install hypergbm[dask]
```

* To support dataset with simplified Chinese in feature generation,
  * Install `jieba` package before running HyperGBM. 
  * OR install with command:
```bash
pip install hypergbm[zhcn]
```

* Install all above with one command:
```bash
pip install hypergbm[all]
```


### Using Docker

It is possible to use HyperGBM in a Docker container. To do this, users can install HyperGBM with `pip` in the Dockerfile. We also publish a mirror image in Docker Hub which can be downloaded directly and includes the following components:

* Python 3.8
* HyperGBM and its dependent packages
* JupyterLab

Docker image tag naming scheme:

* <hypergbm_version>: Python + JupyterLab + HyperGBM + HyperGBM notebook plugins
* <hypergbm_version>-cuda<cuda_version>-cuml<cuml_version>: above + CUDA toolkit + cuML
* <hypergbm_version>-cuda<cuda_version>-cuda<cuml_version>-lgbmgpu: above + GPU enabled LightGBM

Download the docker image:
```bash
docker pull datacanvas/hypergbm
```

Run a docker container:
```bash
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm
```

Then one can visit `http://<your-ip>:8888` in the browser and type in the default token to start.


### Requirements for GPU acceleration

* cuML and cuDF

HyperGBM accelerates data processing with NVIDIA RAPIDS cuML and cuDF. Please install them before running HyperGBM on GPU. For detailed instructions, check the link [Get RAPIDS](https://rapids.ai/start.html#get-rapids).

* LightGBM with GPU support

Default installation of LightGBM does not support GPU training. Please ensure LightGBM with GPU support before installing HyperGBM. For detailed instructions, check the link [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html).

* XGBoost and CatBoost with GPU support

Default installations of XGBoost and CatBoost have supported GPU training. However, if you build them from source code, please enable GPU support.
