# %% [markdown]
# # Hematopoiesis - DPT
#
# Construct diffusion pseudotime on NeurIPS 2021 hematopoiesis data.

# %%
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv
from anndata import AnnData

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
FIGURE_FORMAT = "pdf"

# %%
os.makedirs(DATA_DIR / "hematopoiesis" / "results", exist_ok=True)

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

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))

    scv.pl.scatter(adata, basis="X_umap", c="l2_cell_type", legend_loc="none", title="", ax=ax)

    fig.savefig(
        FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_l2_cell_type_full_data.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

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

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="X_umap", c="l2_cell_type", legend_loc="none", title="", ax=ax)

    fig.savefig(
        FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_l2_cell_type_subsetted_data.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_graph(
        adata, basis="umap", which_graph="connectivities", color="l2_cell_type", legend_loc=False, title="", ax=ax
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_with_knn_graph.pdf",
        format="pdf",
        transparent=True,
        bbox_inches="tight",
        dpi=400,
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

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        scv.pl.scatter(
            adata, basis="diffmap", color=["l2_cell_type"], components=["4, 5"], title="", legend_loc=False, ax=ax
        )

        fig.savefig(
            FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "diffmap_dc4_vs_dc5_colored_by_celltype.eps",
            format="eps",
            transparent=True,
            bbox_inches="tight",
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

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata,
        basis="diffmap",
        c=["hsc_cluster"],
        legend_loc=False,
        components=["4, 5"],
        add_outline="HSC",
        title="",
        ax=ax,
    )

    fig.savefig(
        FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "diffmap_dc4_vs_dc5_hsc_outlined.pdf",
        format="pdf",
        transparent=True,
        bbox_inches="tight",
        dpi=400,
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

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "diffmap_dc4_vs_dc5_pseudotime_root_id.eps",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
        )

# %%
adata.uns["iroot"] = root_idx
sc.tl.dpt(adata, n_dcs=6)

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color=root_idx, palette=palette, color_map="viridis", size=50, ax=ax)

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_pseudotime_root_id.eps",
            format="eps",
            transparent=True,
            bbox_inches="tight",
        )

# %%
if running_in_notebook():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="dpt_pseudotime", title="", color_map="viridis", ax=ax)

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="dpt_pseudotime", title="", color_map="viridis", colorbar=False, ax=ax)

    fig.savefig(
        FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_dpt_pseudotime.eps",
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

if SAVE_FIGURES:
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
    ax.set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])

    sns.reset_orig()

    fig.savefig(
        FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "dpt_per_cell_type.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## CellRank

# %% [markdown]
# ### Kernel

# %%
ptk = cr.kernels.PseudotimeKernel(adata, time_key="dpt_pseudotime").compute_transition_matrix(threshold_scheme="soft")

# %%
if running_in_notebook():
    ptk.plot_projection(color="l2_cell_type", recompute=True, basis="X_umap", dpi=200, legend_fontsize=5)

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        ptk.plot_projection(
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
            FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "dpt_stream.png",
            format="png",
            transparent=True,
            bbox_inches="tight",
            dpi=400,
        )

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(ptk)
estimator.compute_schur(n_components=20)
estimator.plot_spectrum(real_only=True)
plt.show()

# %%
estimator.compute_macrostates(3, cluster_key="l2_cell_type")

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

# %%
terminal_states = ["CD14+ Mono", "Normoblast", "cDC2", "pDC"]
cluster_key = "l2_cell_type"

if (DATA_DIR / "hematopoiesis" / "results" / "tsi-ptk.csv").is_file():
    tsi_df = pd.read_csv(DATA_DIR / "hematopoiesis" / "results" / "tsi-ptk.csv")
    estimator._tsi = AnnData(tsi_df, uns={"terminal_states": terminal_states, "cluster_key": cluster_key})
    tsi_score = estimator.tsi(n_macrostates=7, terminal_states=terminal_states, cluster_key=cluster_key)
else:
    tsi_score = estimator.tsi(n_macrostates=7, terminal_states=terminal_states, cluster_key=cluster_key)
    estimator._tsi.to_df().to_csv(DATA_DIR / "hematopoiesis" / "results" / "tsi-ptk.csv", index=False)

print(f"TSI score: {tsi_score:.2f}")

# %%
palette = {"PseudotimeKernel": "#DE8F05", "Optimal identification": "#000000"}

if SAVE_FIGURES:
    fname = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / f"tsi-dpt.{FIGURE_FORMAT}"
else:
    fname = None

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    estimator.plot_tsi(palette=palette, save=fname)
    plt.show()

# %%
estimator.compute_macrostates(6, cluster_key="l2_cell_type")

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
    fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_cr_dpt_macrostates.pdf"
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
        fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "dpt_macrostate_purity.pdf"
    else:
        fpath = None

    palette = dict(zip(estimator.macrostates.cat.categories, estimator._macrostates.colors))

    plot_state_purity(macrostate_purity, palette=palette, fpath=fpath, format="eps")
    plt.show()

# %%
estimator.set_terminal_states(["pDC", "cDC2", "CD14+ Mono", "Normoblast"])

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
    fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "umap_colored_by_cr_dpt_terminal_states.pdf"
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
terminal_state_purity = get_state_purity(adata, estimator, states="terminal_states", obs_col="l2_cell_type")
print(f"Mean purity: {np.mean(list(macrostate_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "dpt_terminal_states_purity.pdf"
    else:
        fpath = None

    palette = dict(zip(estimator.terminal_states.cat.categories, estimator._term_states.colors))

    plot_state_purity(
        terminal_state_purity,
        palette=palette,
        order=["pDC", "cDC2", "CD14+ Mono", "Normoblast"],
        fpath=fpath,
        format="eps",
    )
    plt.show()

# %%
estimator.compute_fate_probabilities(tol=1e-7)
if running_in_notebook():
    estimator.plot_fate_probabilities(same_plot=False, basis="umap", ncols=2)

if SAVE_FIGURES:
    adata.obs["fate_prob_pDC"] = adata.obsm["lineages_fwd"][:, "pDC"].X.squeeze()
    adata.obs["fate_prob_cDC2"] = adata.obsm["lineages_fwd"][:, "cDC2"].X.squeeze()
    adata.obs["fate_prob_CD14+Mono"] = adata.obsm["lineages_fwd"][:, "CD14+ Mono"].X.squeeze()
    adata.obs["fate_prob_Normoblast"] = adata.obsm["lineages_fwd"][:, "Normoblast"].X.squeeze()

    for terminal_state in ["pDC", "cDC2", "CD14+Mono", "Normoblast"]:
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
                FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / f"dpt_fate_prob_{terminal_state}.eps",
                format="eps",
                transparent=True,
                bbox_inches="tight",
            )

# %%
if running_in_notebook():
    if SAVE_FIGURES:
        fname = f"{FIG_DIR}/pseudotime_kernel/hematopoiesis/umap_colored_by_dpt_fate.pdf"
    else:
        fname = False
    fig, ax = plt.subplots(figsize=(6, 4))
    estimator.plot_fate_probabilities(
        same_plot=True,
        basis="umap",
        title="",
        legend_loc=False,
        save=fname,
        ax=ax,
    )

# %% [markdown]
# ## Driver analysis

# %%
drivers = estimator.compute_lineage_drivers(
    return_drivers=True, cluster_key="l2_cell_type", lineages=["pDC"], clusters=["HSC", "pDC"]
)

if running_in_notebook():
    estimator.plot_lineage_drivers(lineage="pDC", n_genes=20, ncols=5, title_fmt="{gene} corr={corr:.2}")
    plt.show()

# %%
model = cr.models.GAM(adata)

if SAVE_FIGURES:
    save = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "gene_trend_runx2.pdf"
else:
    save = None
cr.pl.gene_trends(
    adata,
    model=model,
    genes="RUNX2",
    time_key="dpt_pseudotime",
    hide_cells=True,
    same_plot=True,
    lineage_cmap=["#8e063b", "#f0b98d", "#d5eae7", "#f3e1eb"],
    figsize=(4, 4),
    save=save,
)
plt.show()


if SAVE_FIGURES:
    save = FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / "gene_trend_tcf4.pdf"
else:
    save = None
cr.pl.gene_trends(
    adata,
    model=model,
    genes="TCF4",
    time_key="dpt_pseudotime",
    hide_cells=True,
    same_plot=True,
    lineage_cmap=["#8e063b", "#f0b98d", "#d5eae7", "#f3e1eb"],
    figsize=(4, 4),
    save=save,
)
plt.show()
