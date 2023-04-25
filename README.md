# CellRank 2's reproducibility repository

## Installation

### Developer installation

```bash
conda create -n cr2-py38 python=3.8 --yes && conda activate cr2-py38
pip install -e ".[dev]"
pre-commit install
```

Jupyter lab and the corresponding kernel can be installed with

```bash
pip install jupyterlab ipywidgets
python -m ipykernel install --user --name cr2-py38 --display-name "cr2-py38"
```
