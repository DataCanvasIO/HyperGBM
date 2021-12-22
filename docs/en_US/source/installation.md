## Installing HyperGBM
We recommend installing HyperGBM with `conda` or `pip`. Installing and using HyperGBM in a Docker container are also possible if you have a Docker environment.

Python version 3.6 or above is necessary before installing HyperGBM with `conda` or `pip`.

### Conda

Install HyperGBM with `conda` from the channel *conda-forge*:

```bash
conda install -c conda-forge hypergbm
```

On the Windows system, recommend install pyarrow(required by hypernets) 4.0 or earlier version with HyperGBM:

```bash
conda install -c conda-forge hypergbm "pyarrow<=4.0"
```

### Pip
Install HyperGBM with different `pip` options:

* Typical installation:
```bash
pip install hypergbm
```

* To run HyperGBM in JupyterLab/Jupyter notebook, install with command:
```bash
pip install hypergbm[notebook]
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