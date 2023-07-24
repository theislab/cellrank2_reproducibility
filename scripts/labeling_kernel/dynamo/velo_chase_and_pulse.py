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
# ## Constants

# %%
N_JOBS = 8

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "sceu_organoid" / "processed" / "preprocessed.h5ad")

adata.obs.rename({"labeling_time": "time"}, axis=1, inplace=True)
adata.layers["new"] = adata.layers.pop("labeled")

del adata.layers["unlabeled"]

adata

# %% [markdown]
# ## Data preprocessing

# %%
# Setting entries needed by Dynamo
prepare_data_for_dynamo(adata, experiment_type="mix_pulse_chase")

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
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", color="cell_type", legend_loc="right")

# %% [markdown]
# ## Parameter inference

# %%
dyn.tl.dynamics(adata, model="deterministic", tkey="time", assumption_mRNA="ss", cores=N_JOBS)

# %%
adata = adata[:, ~adata.var["gamma"].isnull()]

adata.var["alpha"] = adata.var["alpha"].astype(float)
adata.var["beta"] = adata.var["beta"].replace({None: np.nan})
adata.var["gamma"] = adata.var["gamma"].replace({None: np.nan})
adata.var["half_life"] = adata.var["half_life"].replace({None: np.nan})
adata.var["a"] = adata.var["a"].astype(float)
adata.var["b"] = adata.var["b"].astype(float)
adata.var["alpha_a"] = adata.var["alpha_a"].astype(float)
adata.var["alpha_i"] = adata.var["alpha_i"].astype(float)
adata.var["p_half_life"] = adata.var["p_half_life"].astype(float)
adata.var["cost"] = adata.var["cost"].astype(float)
adata.var["logLL"] = adata.var["logLL"].astype(float)

del adata.uns["dynamics"]["X_fit_data"]

# %%
adata.write(DATA_DIR / "sceu_organoid" / "processed" / "adata_dynamo-chase_and_pulse-2000features.h5ad")
