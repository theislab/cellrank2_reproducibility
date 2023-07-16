# %% [markdown]
# # Diffusion pseudotime

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv

sys.path.extend(["../../../", "."])
from paths import DATA_DIR, FIG_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "cytotrace_kernel" / "embryoid_body", exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "embryoid_body" / "embryoid_body.h5ad")
adata

# %%
scv.pl.scatter(adata, basis="umap", c="stage", palette="viridis")

# %%
scv.pl.scatter(adata, basis="umap", c="cell_type", dpi=200)

# %% [markdown]
# ## Pseudotime construction

# %%
sc.tl.diffmap(adata)

# %%
root_idx = 1458  # adata.obsm['X_diffmap'][:, 1].argmin()
scv.pl.scatter(adata, basis="diffmap", color=["cell_type", root_idx], legend_loc="right", components="1, 2", size=25)

# %%
adata.uns["iroot"] = root_idx

# %%
dpt_pseudotime = sc.tl.dpt(adata)

# %%
scv.pl.scatter(
    adata,
    c=["dpt_pseudotime", "stage"],
    basis="umap",
    legend_loc="right",
    color_map="viridis",
)

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata, basis="umap", c="dpt_pseudotime", title="", legend_loc=False, colorbar=False, cmap="gnuplot2", ax=ax
    )

    fig.savefig(
        FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "umap_colored_by_dpt_pseudotime.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
df = adata.obs[["dpt_pseudotime", "stage"]].copy()

sns.set_style(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(
    data=df,
    x="stage",
    y="dpt_pseudotime",
    scale="width",
    palette=["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    ax=ax,
)

ax.tick_params(axis="x", rotation=45)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
plt.show()

if SAVE_FIGURES:
    ax.set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])

    fig.savefig(
        FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "dpt_vs_stage.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )
sns.reset_orig()
