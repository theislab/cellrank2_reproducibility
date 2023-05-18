# %% [markdown]
# # Fixed point analysis - Chase and pulse experiment
#
# Dynamo's fixed-point-based analysis of scEU-seq organoid data using only the chase and pulse experiment.

# %% [markdown]
# ## Library imports

# %%
import sys

import pandas as pd

import dynamo as dyn
import scanpy as sc
import scvelo as scv

sys.path.extend(["../../../", "."])
from paths import DATA_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3
scv.settings.verbosity = 3

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "sceu_organoid" / "processed" / "adata_dynamo-chase_and_pulse-2000features.h5ad")
adata

# %% [markdown]
# ## Preprocessing

# %%
dyn.tl.reduceDimension(adata, layer="X_new", enforce=True)

# %% [markdown]
# ## Fixed point analysis

# %% [markdown]
# ### Velocity with total RNA

# %%
dyn.tl.cell_velocities(adata, ekey="M_t", vkey="velocity_T", enforce=True)
dyn.vf.VectorField(adata, basis="umap")
dyn.vf.topography(adata)

# %%
df = pd.DataFrame(adata.uns["VecFld_umap"]["Xss"], columns=["umap_1", "umap_2"])
df["fixed_point_type"] = adata.uns["VecFld_umap"]["ftype"]
df["fixed_point_type"].replace({-1: "stable", 0: "saddle", 1: "unstable"}, inplace=True)

neighbor_idx = dyn.tools.utils.nearest_neighbors(df[["umap_1", "umap_2"]], adata.obsm["X_umap"])
df["cell_type"] = [
    adata.obs.loc[adata.obs_names[neighbors], "cell_type"].mode().values[0] for neighbors in neighbor_idx
]

print(f"Stable FPs: {df.loc[df['fixed_point_type'] == 'stable', 'cell_type'].unique()}")

# %% [markdown]
# ### Velocity with new RNA

# %%
dyn.tl.cell_velocities(adata, ekey="M_n", vkey="velocity_N", enforce=True)
dyn.vf.VectorField(adata, basis="umap")
dyn.vf.topography(adata)

# %%
df = pd.DataFrame(adata.uns["VecFld_umap"]["Xss"], columns=["umap_1", "umap_2"])
df["fixed_point_type"] = adata.uns["VecFld_umap"]["ftype"]
df["fixed_point_type"].replace({-1: "stable", 0: "saddle", 1: "unstable"}, inplace=True)

neighbor_idx = dyn.tools.utils.nearest_neighbors(df[["umap_1", "umap_2"]], adata.obsm["X_new_umap"])
df["cell_type"] = [
    adata.obs.loc[adata.obs_names[neighbors], "cell_type"].mode().values[0] for neighbors in neighbor_idx
]

print(f"Stable FPs: {df.loc[df['fixed_point_type'] == 'stable', 'cell_type'].unique()}")
