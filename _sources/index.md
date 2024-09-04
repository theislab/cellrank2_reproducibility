# CellRank 2's reproducibility repository

This repsitory contains the code to reproduce results shown in [_CellRank 2: unified fate mapping in multiview single-cell data_](https://doi.org/10.1038/s41592-024-02303-9)
and has been rendered as a Jupyter book [here](https://theislab.github.io/cellrank2_reproducibility/index.html). All datasets are freely available via CellRank's
API or [figshare](https://doi.org/10.6084/m9.figshare.c.6843633.v1). If you use our tool in your own work,
please cite it as

```
@article{weiler:24,
    title = {CellRank 2: unified fate mapping in multiview single-cell data},
    volume = {21},
    ISSN = {1548-7105},
    url = {http://dx.doi.org/10.1038/s41592-024-02303-9},
    DOI = {10.1038/s41592-024-02303-9},
    number = {7},
    journal = {Nature Methods},
    publisher = {Springer Science and Business Media LLC},
    author = {Weiler,  Philipp and Lange,  Marius and Klein,  Michal and Pe’er,  Dana and Theis,  Fabian},
    year = {2024},
    month = jun,
    pages = {1196–1205}
}
```

## Installation

To run the analyses notebooks locally, clone and install the repository as follows:

```bash
conda create -n cr2-py310 python=3.10 --yes && conda activate cr2-py310
conda install -c conda-forge cellrank

git clone https://github.com/theislab/cellrank2_reproducibility.git
cd cellrank2_reproducibility
pip install -e .
python -m ipykernel install --user --name cr2-py310 --display-name "cr2-py310"
```
