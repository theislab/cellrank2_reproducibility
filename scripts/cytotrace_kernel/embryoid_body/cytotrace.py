# %% [markdown]
# # Embryoid body development - CytoTRACE-based analysis
#
# Construct CytoTRACE score for embryoid body development and analyse data with the _CytoTRACEKernel_.

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv

from cr2 import get_state_purity, plot_state_purity, plot_states, running_in_notebook

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
    os.makedirs(FIG_DIR / "cytotrace_kernel" / "embryoid_body", exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "embryoid_body" / "embryoid_body.h5ad")
adata

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", c="stage", palette="viridis")

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", c="stage", title="", legend_loc=False, palette="viridis", ax=ax)
    fig.savefig(
        FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "umap_colored_by_stage.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", c="cell_type", dpi=200)

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", c="cell_type", title="", legend_loc=False, ax=ax)
    fig.savefig(
        FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "umap_colored_by_cell_type.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

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

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata, basis="umap", c="ct_pseudotime", title="", colorbar=False, color_map="gnuplot2", show=False, ax=ax
    )

    fig.savefig(
        FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "umap_colored_by_ct_pseudotime.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
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

if SAVE_FIGURES:
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
    ax.set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])

    fig.savefig(
        FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "cytotrace_vs_stage.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )
    sns.reset_orig()

# %%
ctk.compute_transition_matrix(threshold_scheme="soft", nu=0.5)

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(ctk)
estimator.compute_schur(n_components=20)
if running_in_notebook():
    estimator.plot_spectrum(real_only=True)
    plt.show()

# %%
estimator.compute_macrostates(20, cluster_key="cell_type")

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

if SAVE_FIGURES:
    fpath = FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "umap_colored_by_cytotrace_macrostates.pdf"
    plot_states(
        adata,
        estimator=estimator,
        which="macrostates",
        basis="umap",
        legend_loc=False,
        title="",
        size=100,
        fpath=fpath,
        format="pdf",
    )

# %%
macrostate_purity = get_state_purity(adata, estimator, states="macrostates", obs_col="cell_type")
print(f"Mean purity: {np.mean(list(macrostate_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "cytotrace_macrostate_purity.pdf"
    else:
        fpath = None

    palette = dict(zip(estimator.macrostates.cat.categories, estimator._macrostates.colors))

    plot_state_purity(macrostate_purity, palette=palette, fpath=fpath, format="eps")
    plt.show()

# %%
estimator.set_terminal_states(
    [
        "EN-1_1",
        "Posterior EN_1",
        "NC",
        "NS-1",
        "NS-2",
        "NS-3",
        "NE-1/NS-5",
        "CPs_4",
        "SMPs_1",
        "EPs",
        "Hemangioblast",
    ]
)

if running_in_notebook():
    plot_states(
        adata,
        estimator=estimator,
        which="terminal_states",
        basis="umap",
        legend_loc="right",
        title="",
        size=100,
        fpath=fpath,
        format="pdf",
    )

if SAVE_FIGURES:
    fpath = FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "umap_colored_by_cytotrace_terminal_states.pdf"
    plot_states(
        adata,
        estimator=estimator,
        which="terminal_states",
        basis="umap",
        legend_loc=False,
        title="",
        size=100,
        fpath=fpath,
        format="pdf",
    )

# %%
macrostate_purity = get_state_purity(adata, estimator, states="terminal_states", obs_col="cell_type")
print(f"Mean purity: {np.mean(list(macrostate_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "cytotrace_terminal_states_purity.pdf"
    else:
        fpath = None

    palette = dict(zip(estimator.terminal_states.cat.categories, estimator._term_states.colors))

    plot_state_purity(
        macrostate_purity,
        palette=palette,
        order=["EN-1_1", "Posterior EN_1", "NC", "NS-1", "NS-2", "NS-3", "NE-1/NS-5", "SMPs_1", "EPs", "Hemangioblast"],
        fpath=fpath,
        format="eps",
    )
    plt.show()

# %%
estimator.compute_fate_probabilities(tol=1e-7)
if running_in_notebook():
    estimator.plot_fate_probabilities(same_plot=False, basis="umap", ncols=3)

if SAVE_FIGURES:
    for terminal_state in estimator.terminal_states.cat.categories:
        adata.obs[f"fate_prob_{terminal_state}"] = adata.obsm["lineages_fwd"][:, terminal_state].X.squeeze()
        fig, ax = plt.subplots(figsize=(6, 4))

        if running_in_notebook():
            scv.pl.scatter(
                adata,
                basis="umap",
                color=f"fate_prob_{terminal_state}",
                cmap="viridis",
                title="",
                colorbar=False,
                ax=ax,
            )

            fig.savefig(
                FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "cytotrace_fate_prob_{terminal_state}.eps",
                format="eps",
                transparent=True,
                bbox_inches="tight",
            )

# %% [markdown]
# ### Driver analysis

# %%
drivers_en_1 = estimator.compute_lineage_drivers(
    return_drivers=True, cluster_key="cell_type", lineages=["EN-1_1"], clusters=["ESC"]
)

if running_in_notebook():
    estimator.plot_lineage_drivers(lineage="EN-1_1", n_genes=20, ncols=5, title_fmt="{gene} corr={corr:.2}")
    plt.show()

# %%
if SAVE_FIGURES:
    for var_name in ["FOXA2", "SOX17"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        scv.pl.scatter(adata, basis="umap", color=var_name, ax=ax)

        fig.savefig(
            FIG_DIR / "cytotrace_kernel" / "embryoid_body" / f"umap_colored_by_{var_name.lower()}.eps",
            format="eps",
            transparent=True,
            bbox_inches="tight",
        )

# %%
human_tfs = pd.read_csv(DATA_DIR / "generic" / "human_tfs.csv", index_col=0)["HGNC symbol"].str.lower().values

n_top_genes_tfs = (
    adata.varm["terminal_lineage_drivers"]["EN-1_1_corr"]
    .sort_values(ascending=False)
    .index[:50]
    .str.lower()
    .isin(human_tfs)
    .sum()
)
print(f"Number of TFs in top 50 genes: {n_top_genes_tfs}")

# %%
model = cr.models.GAM(adata)

if SAVE_FIGURES:
    save = FIG_DIR / "cytotrace_kernel" / "embryoid_body" / "heatmap_en_1_lineage.pdf"
else:
    save = None

cr.pl.heatmap(
    adata,
    model,
    genes=adata.varm["terminal_lineage_drivers"]["EN-1_1_corr"].sort_values(ascending=False).index[:50],
    show_fate_probabilities=False,
    show_all_genes=True,
    lineages="EN-1_1",
    time_key="ct_pseudotime",
    figsize=(10, 15),
    save=save,
)
plt.show()
