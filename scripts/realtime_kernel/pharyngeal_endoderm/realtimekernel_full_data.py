# %% [markdown]
# # Pharyngeal endoderm development analysis with the RealTimeKernel

# %% [markdown]
# ## Import packages

# %%
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

import cellrank as cr
import scanpy as sc
import scvelo as scv
import wot

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
    os.makedirs(FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm", exist_ok=True)

# %%
(DATA_DIR / "pharyngeal_endoderm" / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Constants

# %%
# fmt: off
S_GENES = [
    "Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1", "Ung", "Gins2",
    "Mcm6", "Cdca7", "Dtl", "Prim1", "Uhrf1", "Mlf1ip", "Hells", "Rfc2",
    "Rpa2", "Nasp", "Rad51ap1", "Gmnn", "Wdr76", "Slbp", "Ccne2", "Ubr7",
    "Pold3", "Msh2", "Atad2", "Rad51", "Rrm2", "Cdc45", "Cdc6", "Exo1",
    "Tipin", "Dscc1", "Blm", "Casp8ap2", "Usp1", "Clspn", "Pola1", "Chaf1b",
    "Brip1", "E2f8",
]

G2M_GENES = [
    "Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80",
    "Cks2", "Nuf2", "Cks1b", "Mki67", "Tmpo", "Cenpf", "Tacc3", "Fam64a",
    "Smc4", "Ccnb2", "Ckap2l", "Ckap2", "Aurkb", "Bub1", "Kif11", "Anp32e",
    "Tubb4b", "Gtse1", "Kif20b", "Hjurp", "Cdca3", "Hn1", "Cdc20", "Ttk",
    "Cdc25c", "Kif2c", "Rangap1", "Ncapd2", "Dlgap5", "Cdca2", "Cdca8",
    "Ect2", "Kif23", "Hmmr", "Aurka", "Psrc1", "Anln", "Lbr", "Ckap5",
    "Cenpe", "Ctcf", "Nek2", "G2e3", "Gas2l3", "Cbx5", "Cenpa",
]
# fmt: on

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "pharyngeal_endoderm" / "raw" / "adata_pharynx.h5ad")
adata.obsm["X_umap"] = adata.obs[["UMAP1", "UMAP2"]].values
adata.obs["day"] = adata.obs["day_str"].astype(float)
adata.obs = adata.obs[["cluster_name", "day", "is_doublet"]]
adata

# %% [markdown]
# ## Data preprocessing

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

# %%
if running_in_notebook():
    scv.pl.scatter(
        adata, basis="umap", c="cluster_name", title="", dpi=250, legend_fontsize=12, legend_fontweight="normal"
    )

if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", c="cluster_name", legend_loc=False, title="", ax=ax)
    fig.savefig(
        FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "umap_colored_by_cell_type_full_data.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", c="day", legend_loc=False, title="", ax=ax)
    fig.savefig(
        FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "umap_colored_by_day_full_data.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## CellRank

# %% [markdown]
# ### Kernel

# %%
if not (DATA_DIR / "pharyngeal_endoderm" / "tmaps_full_data").exists():
    ot_model = wot.ot.OTModel(adata)
    ot_model.compute_all_transport_maps(tmap_out=DATA_DIR / "pharyngeal_endoderm" / "tmaps_full_data" / "tmaps")

# %%
adata.obs["day"] = adata.obs["day"].astype("category")

rtk = cr.kernels.RealTimeKernel.from_wot(
    adata, path=DATA_DIR / "pharyngeal_endoderm" / "tmaps_full_data", time_key="day"
)
rtk.compute_transition_matrix(
    growth_iters=3, growth_rate_key="growth_rate_init", self_transitions="all", conn_weight=0.1
)

# %% [markdown]
# ### Estimator

# %%
estimator = cr.estimators.GPCCA(rtk)
estimator.compute_schur(n_components=15)
estimator.plot_spectrum(real_only=True)
plt.show()

# %%
estimator.compute_macrostates(n_states=13, cluster_key="cluster_name")

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
    fpath = FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "umap_colored_by_macrostates_full_data.pdf"
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
macrostate_purity = get_state_purity(adata, estimator, states="macrostates", obs_col="cluster_name")
print(f"Mean purity: {np.mean(list(macrostate_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "macrostate_purity_full_data.eps"
    else:
        fpath = None

    palette = dict(zip(estimator.macrostates.cat.categories, estimator._macrostates.colors))

    plot_state_purity(macrostate_purity, palette=palette, fpath=fpath, format="eps")
    plt.show()

# %%
estimator.set_terminal_states(
    [
        "late_Dlx2_1, late_Dlx2_2",
        "late_Runx1",
        "parathyroid",
        "cTEC_1, cTEC_2",
        "mTEC",
        "late_Grhl3",
        "late_Pitx2",
        "ubb",
        "thyroid",
        "late_Dmrt1",
    ]
)

estimator.rename_terminal_states({"cTEC_1, cTEC_2": "cTEC", "late_Dlx2_1, late_Dlx2_2": "late_Dlx2"})

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
    )

if SAVE_FIGURES:
    fpath = FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "umap_colored_by_terminal_states_full_data.pdf"
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
terminal_state_purity = get_state_purity(adata, estimator, states="terminal_states", obs_col="cluster_name")
print(f"Mean purity: {np.mean(list(terminal_state_purity.values()))}")

if running_in_notebook():
    if SAVE_FIGURES:
        fpath = FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "terminal_states_purity_full_data.eps"
    else:
        fpath = None

    palette = dict(zip(estimator.terminal_states.cat.categories, estimator._term_states.colors))

    plot_state_purity(terminal_state_purity, palette=palette, fpath=fpath, format="eps")
    plt.show()

# %%
estimator.compute_fate_probabilities(solver="gmres", use_petsc=True)

# %%
estimator.compute_fate_probabilities()
if running_in_notebook():
    estimator.plot_fate_probabilities(same_plot=False, basis="umap", perc=[0, 99], ncols=3)

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
                FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / f"fate_prob_{terminal_state}_full_data.eps",
                format="eps",
                transparent=True,
                bbox_inches="tight",
            )

# %% [markdown]
# ### Driver analysis

# %%
adata.obs["cluster_name_"] = adata.obs["cluster_name"].copy().astype(str).astype("category")

# %%
drivers_ctec = estimator.compute_lineage_drivers(
    return_drivers=True, cluster_key="cluster_name_", lineages=["cTEC"], clusters=["nan", "early_thymus", "cTEC"]
)

gene_names = drivers_ctec.loc[
    ~(
        drivers_ctec.index.str.startswith(("mt.", "Rpl", "Rps", "^Hb[^(p)]"))
        | drivers_ctec.index.isin(S_GENES + G2M_GENES)
    ),
    :,
].index

if running_in_notebook():
    estimator.plot_lineage_drivers(lineage="cTEC", n_genes=20, ncols=5, title_fmt="{gene} corr={corr:.2}")

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        scv.pl.scatter(
            adata, basis="umap", c="Foxn1", cmap="viridis", title="", legend_loc=None, colorbar=False, ax=ax, s=25
        )
        fig.savefig(
            FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "umap_colored_by_foxn1_full_data.eps",
            format="eps",
            transparent=True,
            bbox_inches="tight",
        )

np.where(gene_names == "Foxn1")

# %%
drivers_parathyroid = estimator.compute_lineage_drivers(
    return_drivers=True, cluster_key="cluster_name_", lineages=["parathyroid"], clusters=["nan"]
)

gene_names = drivers_parathyroid.loc[
    ~(
        drivers_parathyroid.index.str.startswith(("mt.", "Rpl", "Rps", "^Hb[^(p)]"))
        | drivers_parathyroid.index.isin(S_GENES + G2M_GENES)
    ),
    :,
].index

if running_in_notebook():
    estimator.plot_lineage_drivers(lineage="parathyroid", n_genes=20, ncols=5, title_fmt="{gene} corr={corr:.2}")

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        scv.pl.scatter(
            adata, basis="umap", c="Gcm2", cmap="viridis", title="", legend_loc=None, colorbar=False, ax=ax, s=25
        )
        fig.savefig(
            FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "umap_colored_by_gcm2_full_data.eps",
            format="eps",
            transparent=True,
            bbox_inches="tight",
        )

np.where(gene_names == "Gcm2")

# %%
drivers_thyroid = estimator.compute_lineage_drivers(
    return_drivers=True, cluster_key="cluster_name_", lineages=["thyroid"], clusters=["nan"]
)

gene_names = drivers_thyroid.loc[
    ~(
        drivers_thyroid.index.str.startswith(("mt.", "Rpl", "Rps", "^Hb[^(p)]"))
        | drivers_thyroid.index.isin(S_GENES + G2M_GENES)
    ),
    :,
].index

if running_in_notebook():
    estimator.plot_lineage_drivers(lineage="thyroid", n_genes=20, ncols=5, title_fmt="{gene} corr={corr:.2}")

    if SAVE_FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        scv.pl.scatter(
            adata, basis="umap", c="Hhex", cmap="viridis", title="", legend_loc=None, colorbar=False, ax=ax, s=25
        )
        fig.savefig(
            FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / "umap_colored_by_hhex_full_data.eps",
            format="eps",
            transparent=True,
            bbox_inches="tight",
        )

np.where(gene_names == "Hhex")

# %%
