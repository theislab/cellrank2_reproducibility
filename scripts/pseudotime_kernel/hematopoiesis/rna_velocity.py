# %% [markdown]
# # Hematopoiesis - RNA velocity
#
# Infer RNA velocity on NeurIPS 2021 hematopoiesis data.

# %%
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

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
    os.makedirs(FIG_DIR / "pseudotime_kernel" / "hematopoiesis", exist_ok=True)

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
    os.makedirs(FIG_DIR / "pseudotime_kernel" / "hematopoiesis", exist_ok=True)

# %% [markdown]
# ## Constants

# %%
N_JOBS = 8

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
adata = sc.read(DATA_DIR / "hematopoiesis" / "processed" / "gex_velocity.h5ad")
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
scv.pp.filter_genes(adata, min_shared_counts=20)
scv.pp.normalize_per_cell(adata)

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

# %%
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

# %%
for gene in ["HBA2", "HBA1", "GYPC", "TFRC", "AKAP13", "ABCB10", "ANK1", "GATA1", "GATA2"]:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis=gene, color="l2_cell_type", frameon=False, ax=ax)

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / f"phase_portrait_{gene}.eps",
            format="eps",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## RNA velocity inference

# %%
scv.tl.recover_dynamics(adata, n_jobs=N_JOBS)

# %%
scv.tl.velocity(adata, mode="dynamical")

# %% [markdown]
# ## CellRank analysis

# %% [markdown]
# ### Kernel

# %%
vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

combined_kernel = 0.8 * vk + 0.2 * ck

# %%
combined_kernel.plot_projection(color="l2_cell_type", recompute=True, basis="X_umap", dpi=200, legend_fontsize=5)

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    combined_kernel.plot_projection(
        color="l2_cell_type",
        recompute=True,
        basis="X_umap",
        title="",
        legend_loc="none",
        alpha=0.25,
        linewidth=2,
        ax=ax,
    )

    fig.savefig(
        FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "rna_velocity_stream.png",
        format="png",
        transparent=True,
        bbox_inches="tight",
        dpi=400,
    )

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(combined_kernel)
estimator.compute_schur(n_components=20)
estimator.plot_spectrum(real_only=True)

# %%
estimator.compute_macrostates(n_states=3, cluster_key="l2_cell_type")

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
        fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_rna_velo_three_macrostates.pdf"
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
macrostate_purity = get_state_purity(adata, estimator, states="macrostates", obs_col="l2_cell_type")
print(f"Mean purity: {np.mean(list(macrostate_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "rna_velo_three_macrostate_purity.pdf"
    else:
        fpath = None

    palette = dict(zip(estimator.macrostates.cat.categories, estimator._macrostates.colors))

    plot_state_purity(macrostate_purity, palette=palette, fpath=fpath, format="eps")

# %%
estimator.set_terminal_states(["pDC", "CD14+ Mono", "Normoblast"])

if running_in_notebook():
    plot_states(
        adata,
        estimator=estimator,
        which="terminal_states",
        basis="umap",
        legend_loc="right",
        title="",
        size=100,
    )

    if SAVE_FIGURES:
        fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_cr_rna_velo_terminal_states.pdf"
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
estimator.compute_fate_probabilities(tol=1e-7)
if running_in_notebook():
    estimator.plot_fate_probabilities(same_plot=False, basis="X_umap", ncols=2)

if SAVE_FIGURES:
    adata.obs["fate_prob_pDC"] = adata.obsm["lineages_fwd"][:, "pDC"].X.squeeze()
    adata.obs["fate_prob_CD14+Mono"] = adata.obsm["lineages_fwd"][:, "CD14+ Mono"].X.squeeze()
    adata.obs["fate_prob_Normoblast"] = adata.obsm["lineages_fwd"][:, "Normoblast"].X.squeeze()

    for terminal_state in ["pDC", "CD14+Mono", "Normoblast"]:
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
                FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / f"rna_velo_fate_prob_{terminal_state}.eps",
                format="eps",
                transparent=True,
                bbox_inches="tight",
            )

# %%
if running_in_notebook():
    if SAVE_FIGURES:
        fname = f"{FIG_DIR}/pseudotime_kernel/hematopoiesis/umap_colored_by_rna_velo_fate.pdf"
    else:
        fname = None
    fig, ax = plt.subplots(figsize=(6, 4))
    estimator.plot_fate_probabilities(
        same_plot=True,
        basis="umap",
        title="",
        legend_loc=False,
        save=fname,
        ax=ax,
    )

# %%
estimator.compute_macrostates(n_states=20, cluster_key="l2_cell_type")

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
        fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_rna_velo_20_macrostates.pdf"
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
macrostate_purity = get_state_purity(adata, estimator, states="macrostates", obs_col="l2_cell_type")
print(f"Mean purity: {np.mean(list(macrostate_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "rna_velo_20_macrostate_purity.pdf"
    else:
        fpath = None

    macrostates_ordered = estimator.macrostates.cat.categories.sort_values()
    palette = dict(zip(estimator.macrostates.cat.categories, estimator._macrostates.colors))

    plot_state_purity(macrostate_purity, order=macrostates_ordered, palette=palette, fpath=fpath, format="eps")
    plt.show()
