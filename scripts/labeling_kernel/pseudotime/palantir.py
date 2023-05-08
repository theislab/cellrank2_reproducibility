# %% [markdown]
# # Intestinal organoid differentiation - Palantir
#
# Construct palantir pseudotime on scEU-seq organoid data.

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import palantir
import scanpy as sc
import scvelo as scv

from cr2 import running_in_notebook

sys.path.extend(["../../../", "."])
from paths import DATA_DIR, FIG_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3
scv.settings.verbosity = 3

# %%
# Inline plotting
# %matplotlib inline

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

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
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", color="cell_type", legend_loc="right")

# %% [markdown]
# ### Palantir pseudotime

# %%
pc_projection = pd.DataFrame(adata.obsm["X_pca"].copy(), index=adata.obs_names)

# diffusion maps
diff_maps = palantir.utils.run_diffusion_maps(pc_projection, n_components=5)

# multiscale space
multiscale_space = palantir.utils.determine_multiscale_space(diff_maps)

# %%
root_idx = 2899  # See DPT notebook
palantir_res = palantir.core.run_palantir(
    multiscale_space, adata.obs_names[root_idx], use_early_cell_as_start=True, num_waypoints=500
)

# %%
adata.obs["palantir_pseudotime"] = palantir_res.pseudotime

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", color=["palantir_pseudotime", root_idx], color_map="viridis")

# %%
if running_in_notebook():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    sc.pl.violin(
        adata,
        keys=["palantir_pseudotime"],
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
        FIG_DIR / "labeling_kernel" / "palantir_vs_cell_type.eps",
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
        keys=["palantir_pseudotime"],
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
        FIG_DIR / "labeling_kernel" / "palantir_vs_cell_type_merged.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
adata.obs["palantir_pseudotime"].to_csv(DATA_DIR / "sceu_organoid" / "results" / "palantir.csv")
