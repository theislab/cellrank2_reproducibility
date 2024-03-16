# %% [markdown]
# # Real-time informed pseudotime on mouse embryonic fibroblasts

# %% [markdown]
# ## Library imports

# %%
import sys

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from mpl_toolkits.axisartist.axislines import AxesZero

import cellrank as cr
import scanpy as sc
import scvelo as scv
from anndata import AnnData
from scanpy.tools._dpt import DPT

sys.path.extend(["../../../", "."])
from paths import DATA_DIR, FIG_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
# set verbosity levels
sc.settings.verbosity = 2
cr.settings.verbosity = 4
scv.settings.verbosity = 3

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")
scv.settings.plot_prefix = ""

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / "realtime_kernel" / "mef").mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "pdf"


# %%
(DATA_DIR / "mef" / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Function definitions


# %%
def get_symmetric_transition_matrix(transition_matrix):
    """Symmetrize transition matrix."""
    sym_mat = (transition_matrix + transition_matrix.T) / 2

    # normalise transition matrix
    row_sums = sym_mat.sum(axis=1).A1
    sym_mat.data = sym_mat.data / row_sums[sym_mat.nonzero()[0]]

    return sym_mat


# %% [markdown]
# ## Data loading

# %%
adata = cr.datasets.reprogramming_schiebinger(DATA_DIR / "mef" / "reprogramming_schiebinger.h5ad")
adata = adata[adata.obs["serum"] == "True"].copy()

adata.obs["day"] = adata.obs["day"].astype(float)
adata.uns["cell_sets_colors"] = sns.color_palette("colorblind").as_hex()[: len(adata.obs["cell_sets"].cat.categories)]

adata

# %%
scv.pl.scatter(adata, basis="force_directed", c=["day", "cell_sets"], legend_loc="right", cmap="gnuplot")

# %% [markdown]
# ## Data pre-processing

# %%
sc.pp.pca(adata)

# %%
sc.pp.neighbors(adata, random_state=0)

# %% [markdown]
# ## Pseudotime construction

# %%
adata.obs["day"] = adata.obs["day"].astype(float).astype("category")

# %%
rtk = cr.kernels.RealTimeKernel.from_wot(adata, path=DATA_DIR / "mef" / "wot_tmaps", time_key="day")
rtk.transition_matrix = load_npz(DATA_DIR / "mef" / "transition_matrices" / "all_connectivities.npz")

# %%
dpt = DPT(adata=adata, neighbors_key="neighbors")
dpt._transitions_sym = get_symmetric_transition_matrix(rtk.transition_matrix)
dpt.compute_eigen(n_comps=15, random_state=0)

adata.obsm["X_diffmap"] = dpt.eigen_basis
adata.uns["diffmap_evals"] = dpt.eigen_values

# %%
"""
df = pd.DataFrame(
    {
        'diff_comp': adata.obsm['X_diffmap'][:, 1],
        'cell_type': adata.obs['cell_sets'].values,
        'day': adata.obs['day'].values,
    }
).reset_index().rename({'index': 'obs_id'}, axis=1)
df = df.loc[df['day'] == "0.0", "diff_comp"]
root_idx = df.index[df.argmax()]
"""

root_idx = 1210
scv.pl.scatter(adata, basis="diffmap", color=["cell_sets", "day", root_idx], components=["1, 2"], size=25)

# %%
adata.uns["iroot"] = root_idx
sc.tl.dpt(adata)

# %%
scv.pl.scatter(adata, basis="force_directed", c=["day", "dpt_pseudotime"], legend_loc="none", cmap="viridis")

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    sc.pl.violin(adata, keys=["dpt_pseudotime"], groupby="day", rotation=90, ax=ax, title="", legend_loc="none")
    plt.show()

    if SAVE_FIGURES:
        ax.hlines([0, 0.25, 0.5, 0.75, 1], xmin=-0.75, xmax=39, linestyles="solid", colors="black", zorder=0)
        ax.axis("off")
        fig.savefig(
            FIG_DIR
            / "realtime_kernel"
            / "mef"
            / f"real_time_informed_pseudotime_vs_time_point_labeled.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            transparent=True,
            bbox_inches="tight",
        )

# %%
spearmanr(adata.obs["dpt_pseudotime"].values, adata.obs["day"].astype(float).values)

# %% [markdown]
# ## Terminal state estimation

# %%
estimator = cr.estimators.GPCCA(rtk)

# %%
estimator.compute_schur(n_components=10)
estimator.plot_spectrum(real_only=True)

# %%
terminal_states = ["Neural", "IPS", "Trophoblast", "Stromal"]
cluster_key = "cell_sets"

if (DATA_DIR / "mef" / "results" / "tsi-rtk.csv").is_file():
    tsi_df = pd.read_csv(DATA_DIR / "mef" / "results" / "tsi-rtk.csv")
    estimator._tsi = AnnData(tsi_df, uns={"terminal_states": terminal_states, "cluster_key": cluster_key})
    tsi_score = estimator.tsi(n_macrostates=10, terminal_states=terminal_states, cluster_key=cluster_key)
else:
    tsi_score = estimator.tsi(n_macrostates=10, terminal_states=terminal_states, cluster_key=cluster_key)
    estimator._tsi.to_df().to_csv(DATA_DIR / "mef" / "results" / "tsi-rtk.csv", index=False)

print(f"TSI score: {tsi_score:.2f}")

# %%
palette = {"RealTimeKernel": "#DE8F05", "Optimal identification": "#000000"}

if SAVE_FIGURES:
    fpath = FIG_DIR / "realtime_kernel" / "mef" / f"tsi-rtk.{FIGURE_FORMAT}"
else:
    fpath = None

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    estimator.plot_tsi(palette=palette, save=fpath)
    plt.show()

# %%
estimator.compute_macrostates(n_states=4, cluster_key="cell_sets")
estimator.plot_macrostates(which="all", basis="force_directed", legend_loc="right", s=100)

# %%
estimator.set_terminal_states(states=["IPS", "Neural", "Trophoblast", "Stromal"])

# %%
estimator.compute_fate_probabilities()
estimator.plot_fate_probabilities(basis="force_directed", same_plot=False)

# %%
palette = dict(zip(adata.obs["cell_sets"].cat.categories, adata.uns["cell_sets_colors"]))
median_pt = adata.obs[["day", "dpt_pseudotime"]].groupby("day").median()["dpt_pseudotime"].values

for terminal_state in estimator.terminal_states.cat.categories:
    fate_prob = adata.obsm["lineages_fwd"][terminal_state].X.squeeze()
    ref_fate_prob = 1 - fate_prob
    log_odds = np.log(np.divide(fate_prob, 1 - fate_prob, where=fate_prob != 1, out=np.zeros_like(fate_prob)) + 1e-12)
    df = pd.DataFrame(
        {
            "Log odds": log_odds,
            "Real-time-informed pseudotime": adata.obs["dpt_pseudotime"].values,
            "Cell set": adata.obs["cell_sets"],
        }
    )
    df = df.loc[(fate_prob != 0) & (fate_prob != 1), :]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.vlines(
        median_pt,
        ymin=df["Log odds"].min(),
        ymax=df["Log odds"].max(),
        linestyles="dashed",
        colors=adata.uns["day_colors"],
        zorder=0,
    )
    sns.scatterplot(
        data=df,
        x="Real-time-informed pseudotime",
        y="Log odds",
        hue="Cell set",
        palette=palette,
        alpha=0.5,
        ax=ax,
    )
    ax.set_title(terminal_state)
    plt.show()

    if SAVE_FIGURES:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(axes_class=AxesZero)

        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
            ax.axis[direction].set_zorder(0)
        ax.axis["xzero"].set_ticklabel_direction("-")
        ax.axis["yzero"].set_ticklabel_direction("+")
        for direction in ["left", "right", "bottom", "top"]:
            ax.axis[direction].set_visible(False)

        ax.vlines(
            median_pt,
            ymin=df["Log odds"].min(),
            ymax=df["Log odds"].max(),
            linestyles="dashed",
            colors=adata.uns["day_colors"],
            zorder=0,
        )
        sns.scatterplot(
            data=df,
            x="Real-time-informed pseudotime",
            y="Log odds",
            hue="Cell set",
            palette=palette,
            alpha=0.5,
            ax=ax,
        )

        ax.get_legend().remove()
        ax.set_xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[])
        ax.set_yticks(ticks=[-4, -2, 0, 2], labels=[])

        ax.set(xlabel=None, ylabel=None)

        fig.savefig(
            FIG_DIR
            / "realtime_kernel"
            / "mef"
            / f"rtk_log_odds_vs_pt-{terminal_state.lower()}_lineage.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.2,
        )

        plt.show()
