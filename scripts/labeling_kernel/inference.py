# %% [markdown]
# # Intestinal organoid differentiation - Parameter inference
#
# Estimation of transcription and degradation rate excluding the Tuft cell cluster in the scEU-seq organoid data.

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Library imports

# %%
import os
import sys

from scipy.sparse import csr_matrix

import scanpy as sc
import scvelo as scv
from scvelo.inference._sceu import (
    get_labeling_time_mask,
    get_labeling_times,
    get_n_neighbors,
    get_obs_dist_argsort,
    get_parameters,
)

from cr2 import running_in_notebook

sys.path.extend(["../../", "."])
from paths import DATA_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo")

# %%
os.makedirs(DATA_DIR / "sceu_organoid" / "results", exist_ok=True)

# %%
N_JOBS = 8

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "sceu_organoid" / "processed" / "preprocessed.h5ad")
adata

# %% [markdown]
# ## Data preprocessing

# %%
adata.layers["labeled_smoothed"] = csr_matrix.dot(adata.obsp["connectivities"], adata.layers["labeled"]).A
adata.layers["unlabeled_smoothed"] = csr_matrix.dot(adata.obsp["connectivities"], adata.layers["unlabeled"]).A
adata.layers["total_smoothed"] = csr_matrix.dot(adata.obsp["connectivities"], adata.layers["total"]).A

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", color="cell_type", legend_loc="right")

# %% [markdown]
# ## Parameter inference

# %%
time_key = "labeling_time"
labeling_times = get_labeling_times(adata=adata, time_key="labeling_time")

labeling_time_mask = get_labeling_time_mask(adata=adata, time_key=time_key, labeling_times=labeling_times)

obs_dist_argsort = get_obs_dist_argsort(adata=adata, labeling_time_mask=labeling_time_mask)

# %%
n_neighbors = get_n_neighbors(
    adata,
    labeling_time_mask=labeling_time_mask,
    obs_dist_argsort=obs_dist_argsort,
    n_nontrivial_counts=20,
    use_rep="labeled_smoothed",
    n_jobs=N_JOBS,
)

# %%
alpha, gamma, r0, success, opt_res = get_parameters(
    adata=adata,
    use_rep="labeled_smoothed",
    time_key="labeling_time",
    experiment_key="experiment",
    n_neighbors=n_neighbors,
    n_jobs=N_JOBS,
)

alpha.to_csv(DATA_DIR / "sceu_organoid" / "results" / "alpha.csv")
gamma.to_csv(DATA_DIR / "sceu_organoid" / "results" / "gamma.csv")
r0.to_csv(DATA_DIR / "sceu_organoid" / "results" / "r0.csv")
success.to_csv(DATA_DIR / "sceu_organoid" / "results" / "success.csv")
