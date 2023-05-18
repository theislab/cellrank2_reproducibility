# %% [markdown]
# # Intestinal organoid differentiation - Dynamo velocity for pulse experiment
#
# Calculate RNA velocity using Dynamo and only pulse experiment

# %% [markdown]
# ## Library imports

# %%
import sys

import numpy as np
from scipy.sparse import csr_matrix

import dynamo as dyn
import scanpy as sc
import scvelo as scv

from cr2 import prepare_data_for_dynamo, running_in_notebook

sys.path.extend(["../../../", "."])
from paths import DATA_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo")

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "sceu_organoid" / "processed" / "raw.h5ad")

adata = adata[adata.obs["labeling_time"] != "dmso", :].copy()
adata = adata[adata.obs["experiment"] == "Pulse", :].copy()
adata = adata[~adata.obs["cell_type"].isin(["Tuft cells"]), :]
adata.obs["labeling_time"] = adata.obs["labeling_time"].astype(float) / 60

# Rename labeling time column; Dynamo does not work otherwise
adata.obs.rename({"labeling_time": "time"}, axis=1, inplace=True)

adata.layers["new"] = adata.layers.pop("labeled")
del (
    adata.layers["labeled_spliced"],
    adata.layers["labeled_unspliced"],
    adata.layers["unlabeled"],
    adata.layers["unlabeled_spliced"],
    adata.layers["unlabeled_unspliced"],
)

adata

# %% [markdown]
# ## Data preprocessing

# %%
adata.obs["cell_type_merged"] = adata.obs["cell_type"].copy()
adata.obs["cell_type_merged"].replace({"Enteroendocrine cells": "Enteroendocrine progenitors"}, inplace=True)

# %%
scv.pp.filter_and_normalize(adata, min_counts=50, layers_normalize=["X", "new", "total"], n_top_genes=1000)

# %%
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

# %%
# Setting entries needed by Dynamo
prepare_data_for_dynamo(adata, experiment_type="one-shot")

adata.layers["X_total"] = csr_matrix(adata.layers["total"].copy())
adata.layers["X_new"] = csr_matrix(adata.layers["new"].copy())

# %%
ntr, var_ntr = dyn.preprocessing.utils.calc_new_to_total_ratio(adata)

adata.obs["ntr"] = ntr
adata.var["ntr"] = var_ntr

# %%
dyn.tl.moments(adata, conn=adata.obsp["connectivities"].copy(), group="time")
adata

# %%
sc.tl.umap(adata)
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", color="cell_type", legend_loc="right")

# %% [markdown]
# ## Parameter inference

# %%
dyn.tl.dynamics(adata, model="deterministic", tkey="time", assumption_mRNA="ss")

# %%
adata = adata[:, ~adata.var["gamma"].isnull()]

adata.var["beta"] = adata.var["beta"].replace({None: np.nan})
adata.var["gamma"] = adata.var["gamma"].replace({None: np.nan})
adata.var["half_life"] = adata.var["half_life"].replace({None: np.nan})
adata.var["alpha_b"] = adata.var["alpha_b"].replace({None: np.nan})
adata.var["alpha_r2"] = adata.var["alpha_r2"].replace({None: np.nan})
adata.var["gamma_b"] = adata.var["gamma_b"].replace({None: np.nan})
adata.var["gamma_r2"] = adata.var["gamma_r2"].replace({None: np.nan})
adata.var["gamma_logLL"] = adata.var["gamma_logLL"].astype(float)
adata.var["delta_b"] = adata.var["delta_b"].astype(float)
adata.var["delta_r2"] = adata.var["delta_r2"].astype(float)
adata.var["bs"] = adata.var["bs"].astype(float)
adata.var["bf"] = adata.var["bf"].astype(float)
adata.var["uu0"] = adata.var["uu0"].astype(float)
adata.var["ul0"] = adata.var["uu0"].astype(float)
adata.var["su0"] = adata.var["su0"].astype(float)
adata.var["sl0"] = adata.var["sl0"].astype(float)
adata.var["U0"] = adata.var["U0"].astype(float)
adata.var["S0"] = adata.var["S0"].astype(float)
adata.var["total0"] = adata.var["total0"].astype(float)
adata.var["beta_k"] = adata.var["beta_k"].astype(float)
adata.var["gamma_k"] = adata.var["gamma_k"].astype(float)

# %%
adata.write(DATA_DIR / "sceu_organoid" / "processed" / "adata_dynamo-pulse-1000features.h5ad")
