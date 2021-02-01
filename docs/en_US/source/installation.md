## Installation

You can use pip or docker to install HyperGBM.

### Using pip

It requires `Python3.6` or above, and uses pip to install HyperGBM:

```shell script
pip install --upgrade pip setuptools # (optional)
pip install hypergbm
```

**Install shap(Optional)**

HyperGBM provides model interpretation based on [shap](https://github.com/slundberg/shap), you can install it refer to this [guide](how_to/install_shap.md) if necessary.


### Using Docker
You can also use HypergGBM through our built-in jupyter docker image with command:
```shell script
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/hypergbm:0.2.0
```

Open browser visit site `http://<your-ip>:8888`，the token is what you have set "you-token"，it can also be empty if do not specified.
