# %% [markdown]
# # Hematopoiesis - Initial state identification
#
# Construct diffusion pseudotime on NeurIPS 2021 hematopoiesis data.

# %%
import sys

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv

from cr2 import running_in_notebook

sys.path.extend(["../../../", "."])
from paths import DATA_DIR, FIG_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
cr.settings.verbosity = 4
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / "pseudotime_kernel" / "hematopoiesis").mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "pdf"

# %%
(DATA_DIR / "hematopoiesis" / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Constants

# %%
CELLTYPES_TO_KEEP = [
    "HSC",
    "MK/E prog",
    "Proerythroblast",
    "Erythroblast",
    "Normoblast",
    "cDC2",
    "pDC",
    "G/M prog",
    "CD14+ Mono",
]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "hematopoiesis" / "processed" / "gex_preprocessed.h5ad")
adata

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="X_umap", c="l2_cell_type", dpi=200, title="", legend_fontsize=5, legend_fontweight=1)

# %% [markdown]
# ## Data preprocessing

# %%
adata = adata[adata.obs["l2_cell_type"].isin(CELLTYPES_TO_KEEP), :].copy()
adata

# %%
sc.pp.neighbors(adata, use_rep="MultiVI_latent")
sc.tl.umap(adata)

# %%
if running_in_notebook():
    scv.pl.scatter(
        adata,
        basis="X_umap",
        c="l2_cell_type",
        dpi=200,
        title="",
        legend_fontsize=10,
        legend_fontweight=5,
    )

# %% [markdown]
# ## Diffusion pseudotime

# %%
sc.tl.diffmap(adata, n_comps=15)

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata, basis="diffmap", color=["l2_cell_type"], components=["4, 5"], size=25, dpi=100, title="", ax=ax
    )

# %%
adata.obs["hsc_cluster"] = (
    adata.obs["l2_cell_type"]
    .astype("str")
    .replace(
        {
            "MK/E prog": "nan",
            "Proerythroblast": "nan",
            "Erythroblast": "nan",
            "Normoblast": "nan",
            "cDC2": "nan",
            "pDC": "nan",
            "G/M prog": "nan",
            "CD14+ Mono": "nan",
        }
    )
    .astype("category")
    .cat.reorder_categories(["nan", "HSC"])
    .copy()
)

celltype_colors = dict(zip(adata.obs["l2_cell_type"].cat.categories, adata.uns["l2_cell_type_colors"]))
adata.uns["hsc_cluster_colors"] = ["#dedede", celltype_colors["HSC"]]

if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata,
        basis="diffmap",
        c=["hsc_cluster"],
        legend_loc="right",
        components=["4, 5"],
        add_outline="HSC",
        title="",
        ax=ax,
    )

# %%
df = (
    pd.DataFrame(
        {
            "diff_comp": adata.obsm["X_diffmap"][:, 5],
            "cell_type": adata.obs["l2_cell_type"].values,
        }
    )
    .reset_index()
    .rename({"index": "obs_id"}, axis=1)
)
df = df.loc[df["cell_type"] == "HSC", "diff_comp"]
root_idx = df.index[df.argmax()]

set2_cmap = sns.color_palette("Set2").as_hex()
palette = [set2_cmap[-1], set2_cmap[1]]

if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="diffmap", c=root_idx, legend_loc=False, palette=palette, components=["4, 5"], ax=ax)

# %%
adata.uns["iroot"] = root_idx
sc.tl.dpt(adata, n_dcs=6)

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color=root_idx, palette=palette, color_map="viridis", size=50, ax=ax)

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="dpt_pseudotime", title="", color_map="viridis", ax=ax)

# %%
if running_in_notebook():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    sc.pl.violin(
        adata,
        keys=["dpt_pseudotime"],
        groupby="l2_cell_type",
        rotation=45,
        title="",
        legend_loc="none",
        order=[
            "HSC",
            "MK/E prog",
            "Proerythroblast",
            "Erythroblast",
            "Normoblast",
            "G/M prog",
            "CD14+ Mono",
            "cDC2",
            "pDC",
        ],
        ax=ax,
    )

    sns.reset_orig()

# %% [markdown]
# ## CellRank

# %% [markdown]
# ### Kernel

# %%
ptk = cr.kernels.PseudotimeKernel(adata, time_key="dpt_pseudotime").compute_transition_matrix(threshold_scheme="soft")
ptk.transition_matrix = ptk.transition_matrix.T

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(ptk)
estimator.compute_schur(n_components=20)
estimator.plot_spectrum(real_only=True)
plt.show()

# %%
estimator.compute_macrostates(1, cluster_key="l2_cell_type")
if running_in_notebook():
    estimator.plot_macrostates(which="all", basis="umap", title="", legend_loc="right", size=100)
    if SAVE_FIGURES:
        fpath = (
            FIG_DIR
            / "pseudotime_kernel"
            / "hematopoiesis"
            / f"umap_colored_by_cr_dpt_macrostates-initial_state.{FIGURE_FORMAT}"
        )
        estimator.plot_macrostates(which="all", basis="umap", title="", legend_loc=False, size=100, save=fpath)
