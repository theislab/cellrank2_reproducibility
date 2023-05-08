# %% [markdown]
# # Intestinal organoid differentiation - DPT
#
# Construct diffusion pseudotime on scEU-seq organoid data.

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import pandas as pd
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv

from cr2 import plot_states, running_in_notebook

sys.path.extend(["../../../", "."])
from paths import DATA_DIR, FIG_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3
scv.settings.verbosity = 3
cr.settings.verbosity = 2

# %%
scv.settings.set_figure_params("scvelo")

# %%
SAVE_FIGURES = False

if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "labeling_kernel", exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "sceu_organoid" / "processed" / "preprocessed.h5ad")
adata

# %% [markdown]
# ## Data preprocessing

# %%
adata.obs["cell_type_merged"] = adata.obs["cell_type"].copy()
adata.obs["cell_type_merged"].replace({"Enteroendocrine cells": "Enteroendocrine progenitors"}, inplace=True)

celltype_to_color = dict(zip(adata.obs["cell_type"].cat.categories, adata.uns["cell_type_colors"]))
adata.uns["cell_type_merged_colors"] = list(
    {cell_type: celltype_to_color[cell_type] for cell_type in adata.obs["cell_type_merged"].cat.categories}.values()
)

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
alpha = pd.read_csv(DATA_DIR / "sceu_organoid" / "results" / "alpha.csv", index_col=0)
alpha.index = alpha.index.astype(str)
adata.layers["transcription_rate"] = alpha.loc[adata.obs_names, adata.var_names]

gamma = pd.read_csv(DATA_DIR / "sceu_organoid" / "results" / "gamma.csv", index_col=0)
gamma.index = gamma.index.astype(str)
adata.layers["degradation_rate"] = alpha.loc[adata.obs_names, adata.var_names]

# %% [markdown]
# ## Velocity

# %%
adata.layers["velocity_labeled"] = (alpha - gamma * adata.layers["labeled_smoothed"]).values

# %% [markdown]
# ## CellRank

# %%
vk = cr.kernels.VelocityKernel(adata, xkey="labeled_smoothed", vkey="velocity_labeled").compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

combined_kernel = 0.8 * vk + 0.2 * ck

# %% [markdown]
# ### Estimator analysis

# %%
estimator = cr.estimators.GPCCA(combined_kernel)

# %%
estimator.compute_schur(n_components=20)
if running_in_notebook():
    estimator.plot_spectrum(real_only=True)

# %% [markdown]
# #### Macrostates

# %%
estimator.compute_macrostates(n_states=12, cluster_key="cell_type")

# %%
if running_in_notebook():
    plot_states(
        adata,
        estimator=estimator,
        which="macrostates",
        basis="umap",
        legend_loc="right",
        title="",
        size=100,
    )

# %% [markdown]
# ## Pseudotime construction

# %% [markdown]
# ### Identification of root cell

# %%
sc.tl.diffmap(adata)

# %%
_macrostates = estimator.macrostates.cat.categories.tolist()
adata.obs["stem_cell_1_cluster"] = (
    estimator.macrostates.astype(str).astype("category").cat.reorder_categories(["nan"] + _macrostates)
)
adata.obs.loc[~adata.obs["stem_cell_1_cluster"].isin(["Stem cells_1"]), "stem_cell_1_cluster"] = "nan"
adata.obs["stem_cell_1_cluster"] = (
    adata.obs["stem_cell_1_cluster"].astype(str).astype("category").cat.reorder_categories(["nan", "Stem cells_1"])
)

macrostates = estimator.macrostates.cat.categories.tolist()
macrostates_colors = adata.uns["macrostates_fwd_colors"]
adata.uns["stem_cell_1_cluster_colors"] = ["#dedede"] + [dict(zip(macrostates, macrostates_colors))["Stem cells_1"]]

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata,
        basis="diffmap",
        c=["stem_cell_1_cluster"],
        legend_loc="right",
        components=["2, 3"],
        add_outline="Stem cells_1",
        title="",
        ax=ax,
    )

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata,
        basis="diffmap",
        c=["stem_cell_1_cluster"],
        legend_loc=False,
        components=["2, 3"],
        add_outline="Stem cells_1",
        title="",
        ax=ax,
    )

    fig.savefig(
        FIG_DIR / "labeling_kernel" / "diffmap_dc2_vs_dc3_stem_cells_1_outlined.pdf",
        format="pdf",
        transparent=True,
        bbox_inches="tight",
    )

# %%
root_idx = adata.obsm["X_diffmap"][:, 2].argmin()
adata.uns["iroot"] = root_idx

if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="diffmap", c="cell_type", legend_loc="right", components=["2, 3"], title="", ax=ax)

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="diffmap", c="cell_type", legend_loc=False, components=["2, 3"], title="", ax=ax)
    fig.savefig(
        FIG_DIR / "labeling_kernel" / "diffmap_dc2_vs_dc3_cell_types.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
set2_cmap = sns.color_palette("Set2").as_hex()
palette = [set2_cmap[-1], set2_cmap[1]]

root_idx = adata.obsm["X_diffmap"][:, 2].argmin()
adata.uns["iroot"] = root_idx

if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="diffmap", c=root_idx, legend_loc=False, palette=palette, components=["2, 3"], ax=ax)

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "labeling_kernel" / "diffmap_dc2_vs_dc3_pseudotime_root_id.eps",
        format="pdf",
        transparent=True,
        bbox_inches="tight",
    )

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color=root_idx, palette=palette, color_map="viridis", ax=ax)
if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "labeling_kernel" / "umap_pseudotime_root_id.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ### DPT pseudotime

# %%
sc.tl.dpt(adata)

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="dpt_pseudotime", title="", color_map="viridis", ax=ax)

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="dpt_pseudotime", title="", color_map="viridis", colorbar=False, ax=ax)
    fig.savefig(
        FIG_DIR / "labeling_kernel" / "umap_colored_by_dpt_pseudotime.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
if running_in_notebook():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    sc.pl.violin(
        adata,
        keys=["dpt_pseudotime"],
        groupby="cell_type",
        rotation=45,
        title="",
        legend_loc="none",
        order=[
            "Stem cells",
            "7",
            "TA cells",
            "2",
            "Enterocytes",
            "Enteroendocrine progenitors",
            "Enteroendocrine cells",
            "Goblet cells",
            "Paneth cells",
        ],
        ax=ax,
    )

if SAVE_FIGURES:
    ax.set(xticklabels=[])
    fig.savefig(
        FIG_DIR / "labeling_kernel" / "dpt_vs_cell_type.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
if running_in_notebook():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    sc.pl.violin(
        adata,
        keys=["dpt_pseudotime"],
        groupby="cell_type_merged",
        rotation=45,
        title="",
        legend_loc="none",
        order=[
            "Stem cells",
            "7",
            "TA cells",
            "2",
            "Enterocytes",
            "Enteroendocrine progenitors",
            "Goblet cells",
            "Paneth cells",
        ],
        ax=ax,
    )

if SAVE_FIGURES:
    ax.set(xticklabels=[])
    fig.savefig(
        FIG_DIR / "labeling_kernel" / "dpt_vs_cell_type_merged.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
adata.obs["dpt_pseudotime"].to_csv(DATA_DIR / "sceu_organoid" / "results" / "dpt.csv")
