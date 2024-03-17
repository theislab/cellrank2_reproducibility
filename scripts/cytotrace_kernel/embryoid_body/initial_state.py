# %% [markdown]
# # Embryoid body development - Initial state identification
#
# Construct CytoTRACE score for embryoid body development and analyse data with the _CytoTRACEKernel_.

# %% [markdown]
# ## Library imports

# %%
import sys

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
    (FIG_DIR / "cytotrace_kernel" / "embryoid_body").mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "pdf"

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "embryoid_body" / "embryoid_body.h5ad")
adata

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", c="stage", palette="viridis")

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", c="cell_type", dpi=200)

# %% [markdown]
# ## Data preprocessing

# %%
adata.layers["spliced"] = adata.X
adata.layers["unspliced"] = adata.X

scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

# %% [markdown]
# ## CellRank

# %% [markdown]
# ### Kernel

# %%
ctk = cr.kernels.CytoTRACEKernel(adata)
ctk.compute_cytotrace()

# %%
if running_in_notebook():
    scv.pl.scatter(
        adata,
        c=["ct_pseudotime", "stage"],
        basis="umap",
        legend_loc="right",
        color_map="viridis",
    )

# %%
df = adata.obs[["ct_pseudotime", "stage"]].copy()

if running_in_notebook():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=df,
        x="stage",
        y="ct_pseudotime",
        scale="width",
        palette=["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
        ax=ax,
    )

    ax.tick_params(axis="x", rotation=45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.show()
    sns.reset_orig()

# %%
ctk.compute_transition_matrix(threshold_scheme="soft", nu=0.5)
ctk.transition_matrix = ctk.transition_matrix.T

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(ctk)
estimator.compute_schur(n_components=5)
if running_in_notebook():
    estimator.plot_spectrum(real_only=True)
    plt.show()

# %%
estimator.compute_macrostates(1, cluster_key="cell_type")

if running_in_notebook():
    estimator.plot_macrostates(which="all", basis="umap", legend_loc="right", title="", size=100)
    if SAVE_FIGURES:
        fpath = (
            FIG_DIR
            / "cytotrace_kernel"
            / "embryoid_body"
            / f"umap_colored_by_cytotrace_macrostates-initial_states.{FIGURE_FORMAT}"
        )
        estimator.plot_macrostates(which="all", basis="umap", title="", legend_loc=False, size=100, save=fpath)
