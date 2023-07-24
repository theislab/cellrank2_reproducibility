# %% [markdown]
# # Intestinal organoid differentiation - RNA velocity with EM model
#
# Estimates RNA velocity with scVelo's _EM model_ and analyses corresponding fate mapping.

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import numpy as np
import pandas as pd

import cellrank as cr
import scanpy as sc
import scvelo as scv

from cr2 import (
    get_state_purity,
    get_var_ranks,
    plot_state_purity,
    plot_states,
    running_in_notebook,
)

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
# ## Constants

# %%
N_JOBS = 8

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "sceu_organoid" / "processed" / "raw.h5ad")

adata = adata[adata.obs["labeling_time"] != "dmso", :].copy()
adata = adata[~adata.obs["cell_type"].isin(["Tuft cells"]), :]
adata.obs["labeling_time"] = adata.obs["labeling_time"].astype(float) / 60

adata.layers["unspliced"] = adata.layers["unlabeled_unspliced"] + adata.layers["labeled_unspliced"]
adata.layers["spliced"] = adata.layers["unlabeled_spliced"] + adata.layers["labeled_spliced"]

umap_coord_df = pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "umap_coords.csv", index_col=0)
umap_coord_df.index = umap_coord_df.index.astype(str)
adata.obsm["X_umap"] = umap_coord_df.loc[adata.obs_names, :].values
del umap_coord_df

adata

# %% [markdown]
# ## Data preprocessing

# %%
adata.obs["cell_type_merged"] = adata.obs["cell_type"].copy()
adata.obs["cell_type_merged"].replace({"Enteroendocrine cells": "Enteroendocrine progenitors"}, inplace=True)

# %%
# filter s.t. at least `min_counts` per labeling time point?
scv.pp.filter_and_normalize(
    adata,
    min_counts=50,
    layers_normalize=["X", "labeled", "unlabeled", "total", "unspliced", "spliced"],
    n_top_genes=2000,
)
adata

# %%
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

scv.pp.moments(adata, n_neighbors=None, n_pcs=None)

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="umap", color="cell_type", legend_loc="right")

# %% [markdown]
# ## Parameter inference

# %%
scv.tl.recover_dynamics(adata, n_jobs=N_JOBS)

# %% [markdown]
# ## Velocity

# %%
scv.tl.velocity(adata, mode="dynamical")

# %% [markdown]
# ## CellRank

# %%
vk = cr.kernels.VelocityKernel(adata, xkey="Ms", vkey="velocity").compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

combined_kernel = 0.8 * vk + 0.2 * ck

# %% [markdown]
# ## Estimator analysis

# %%
estimator = cr.estimators.GPCCA(combined_kernel)

# %%
estimator.compute_schur(n_components=20)
if running_in_notebook():
    estimator.plot_spectrum(real_only=True)

# %% [markdown]
# #### Macrostates

# %%
estimator.compute_macrostates(n_states=18, cluster_key="cell_type")

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

    if SAVE_FIGURES:
        fpath = FIG_DIR / "labeling_kernel" / "umap_colored_by_cr_macrostates_scvelo.pdf"
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
        fpath = FIG_DIR / "labeling_kernel" / "macrostate_purity_rna_velo.pdf"
    else:
        fpath = None

    palette = dict(zip(estimator.macrostates.cat.categories, estimator._macrostates.colors))
    order = estimator.macrostates.cat.categories.sort_values().to_list()
    plot_state_purity(macrostate_purity, palette=palette, fpath=fpath, order=order, format="eps")

# %%
estimator.set_terminal_states(states=["Enterocytes", "Paneth cells", "Enteroendocrine progenitors", "Goblet cells"])
terminal_states = estimator.terminal_states.cat.categories

# %%
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
        fpath = FIG_DIR / "labeling_kernel" / "umap_colored_by_cr_terminal_states_scvelo.pdf"
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
terminal_state_purity = get_state_purity(adata, estimator, states="terminal_states", obs_col="cell_type")
print(f"Mean purity: {np.mean(list(terminal_state_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "labeling_kernel" / "terminal_state_purity_rna_velo.pdf"
    else:
        fpath = None

    palette = dict(zip(estimator.terminal_states.cat.categories, estimator._term_states.colors))
    order = estimator.terminal_states.cat.categories.sort_values().to_list()
    plot_state_purity(terminal_state_purity, palette=palette, order=order, fpath=fpath, format="eps")

# %% [markdown]
# #### Fate probabilities

# %%
estimator.compute_fate_probabilities()

# %%
if running_in_notebook():
    estimator.plot_fate_probabilities(same_plot=False, size=50, basis="umap")

# %% [markdown]
# ### Driver analysis

# %%
goblet_markers = (
    pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "goblet_markers.csv")["Gene"].str.lower().tolist()
)

goblet_markers = adata.var_names[adata.var_names.str.lower().isin(goblet_markers)]

# %%
goblet_regulators = (
    pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "goblet_regulators.csv")["Gene"].str.lower().tolist()
)

goblet_regulators = adata.var_names[adata.var_names.str.lower().isin(goblet_regulators)]

# %%
goblet_and_paneth_regulators = (
    pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "goblet_and_paneth_regulators.csv")["Gene"]
    .str.lower()
    .tolist()
)

goblet_and_paneth_regulators = adata.var_names[adata.var_names.str.lower().isin(goblet_and_paneth_regulators)]

# %%
paneth_markers = (
    pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "paneth_markers.csv")["Gene"].str.lower().tolist()
)

paneth_markers = adata.var_names[adata.var_names.str.lower().isin(paneth_markers)]

# %%
eec_markers = pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "eec_markers.csv")["Gene"].str.lower().tolist()

eec_markers = adata.var_names[adata.var_names.str.lower().isin(eec_markers)]

# %%
eec_progenitor_markers = (
    pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "eec_progenitor_markers.csv")["Gene"].str.lower().tolist()
)

eec_progenitor_markers = adata.var_names[adata.var_names.str.lower().isin(eec_progenitor_markers)]

# %%
enterocyte_markers = (
    pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "enterocyte_markers.csv")["Gene"].str.lower().tolist()
)

enterocyte_markers = adata.var_names[adata.var_names.str.lower().isin(enterocyte_markers)]

# %%
enterocyte_progenitor_markers = (
    pd.read_csv(DATA_DIR / "sceu_organoid" / "processed" / "enterocyte_progenitor_markers.csv")["Gene"]
    .str.lower()
    .tolist()
)

enterocyte_progenitor_markers = adata.var_names[adata.var_names.str.lower().isin(enterocyte_progenitor_markers)]

# %%
gene_ranks = {terminal_state: pd.DataFrame() for terminal_state in terminal_states}

# %%
drivers = estimator.compute_lineage_drivers(
    cluster_key="cell_type",
    lineages=["Enteroendocrine progenitors", "Goblet cells", "Paneth cells", "Enterocytes"],
    clusters=["Stem cells"],
    return_drivers=True,
)

for terminal_state in terminal_states:
    drivers = drivers.merge(
        pd.DataFrame(drivers.sort_values(by=f"{terminal_state}_corr", ascending=False).index)
        .reset_index()
        .rename(columns={"index": f"Corr. rank - {terminal_state}"})
        .set_index(0),
        left_index=True,
        right_index=True,
    )

# %% [markdown]
# ### Goblet cells

# %%
_df = get_var_ranks(
    var_names=goblet_markers, drivers=drivers, macrostate="Goblet cells", var_type="Marker", model="EM Model"
)
gene_ranks["Goblet cells"] = pd.concat([gene_ranks["Goblet cells"], _df])

# %%
_df = get_var_ranks(
    var_names=goblet_and_paneth_regulators,
    drivers=drivers,
    macrostate="Goblet cells",
    var_type="Goblet/Paneth regulator",
    model="EM Model",
)
gene_ranks["Goblet cells"] = pd.concat([gene_ranks["Goblet cells"], _df])

# %% [markdown]
# ### Paneth cells

# %%
_df = get_var_ranks(
    var_names=paneth_markers, drivers=drivers, macrostate="Paneth cells", var_type="Marker", model="EM Model"
)
gene_ranks["Paneth cells"] = pd.concat([gene_ranks["Paneth cells"], _df])

# %%
_df = get_var_ranks(
    var_names=goblet_and_paneth_regulators,
    drivers=drivers,
    macrostate="Paneth cells",
    var_type="Goblet/Paneth regulator",
    model="EM Model",
)
gene_ranks["Paneth cells"] = pd.concat([gene_ranks["Paneth cells"], _df])

# %% [markdown]
# ### Enteroendocrine

# %%
_df = get_var_ranks(
    var_names=eec_markers,
    drivers=drivers,
    macrostate="Enteroendocrine progenitors",
    var_type="Marker",
    model="EM Model",
)
gene_ranks["Enteroendocrine progenitors"] = pd.concat([gene_ranks["Enteroendocrine progenitors"], _df])

# %%
_df = get_var_ranks(
    var_names=eec_progenitor_markers,
    drivers=drivers,
    macrostate="Enteroendocrine progenitors",
    var_type="Progenitor marker",
    model="EM Model",
)
gene_ranks["Enteroendocrine progenitors"] = pd.concat([gene_ranks["Enteroendocrine progenitors"], _df])

# %% [markdown]
# ### Enterocytes

# %%
_df = get_var_ranks(
    var_names=enterocyte_markers, drivers=drivers, macrostate="Enterocytes", var_type="Marker", model="EM Model"
)
gene_ranks["Enterocytes"] = pd.concat([gene_ranks["Enterocytes"], _df])

# %%
_df = get_var_ranks(
    var_names=enterocyte_progenitor_markers,
    drivers=drivers,
    macrostate="Enterocytes",
    var_type="Progenitor marker",
    model="EM Model",
)
gene_ranks["Enterocytes"] = pd.concat([gene_ranks["Enterocytes"], _df])

# %%
for terminal_state in gene_ranks.keys():
    gene_ranks[terminal_state] = gene_ranks[terminal_state].sort_values(f"Corr. rank - {terminal_state}")
    gene_ranks[terminal_state].to_csv(
        DATA_DIR / "sceu_organoid" / "results" / f"gene_ranks_{terminal_state}-em_model.csv"
    )
