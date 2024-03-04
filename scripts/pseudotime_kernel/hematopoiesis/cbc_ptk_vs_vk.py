# %% [markdown]
# # Cross boundary correctness score
#
# Analysis of cross boundary correctness (CBC) score of pseudotime-based analysis with the PseudotimeKernel and an RNA velocity-based analysis with VelocityKernel on the NeurIPS 2021 hematopoiesis data.

# %% [markdown]
# ## Library imports

# %%
import os
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from matplotlib.patches import Patch

import cellrank as cr
import scanpy as sc
import scvelo as scv
from anndata import AnnData

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
    os.makedirs(FIG_DIR / "pseudotime_kernel" / "hematopoiesis", exist_ok=True)

FIGURE_FORMAT = "pdf"

# %%
os.makedirs(DATA_DIR / "hematopoiesis" / "results", exist_ok=True)

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

# %%
STATE_TRANSITIONS = [
    ("HSC", "pDC"),
    ("HSC", "cDC2"),
    ("HSC", "G/M prog"),
    ("G/M prog", "CD14+ Mono"),
    ("HSC", "MK/E prog"),
    ("MK/E prog", "Proerythroblast"),
    ("Proerythroblast", "Erythroblast"),
    ("Erythroblast", "Normoblast"),
]


# %% [markdown]
# ## Function definitions


# %%
def get_significance(pvalue) -> str:
    """Assign significance symbol based on p-value."""
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."


# %%
def get_dpt_adata() -> AnnData:
    """Load and preprocess data for pseudotime-based analysis."""
    adata = sc.read(DATA_DIR / "hematopoiesis" / "processed" / "gex_preprocessed.h5ad")
    adata = adata[adata.obs["l2_cell_type"].isin(CELLTYPES_TO_KEEP), :].copy()

    sc.pp.neighbors(adata, use_rep="MultiVI_latent")
    sc.tl.umap(adata)

    sc.tl.diffmap(adata, n_comps=15)

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

    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata, n_dcs=6)

    return adata


# %%
def get_velo_adata() -> AnnData:
    """Load and preprocess data for RNA velocity-based analysis."""
    adata = sc.read(DATA_DIR / "hematopoiesis" / "processed" / "gex_velocity.h5ad")
    adata = adata[adata.obs["l2_cell_type"].isin(CELLTYPES_TO_KEEP), :].copy()

    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)

    sc.pp.neighbors(adata, use_rep="MultiVI_latent")
    sc.tl.umap(adata)

    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    scv.tl.recover_dynamics(adata, n_jobs=N_JOBS)
    scv.tl.velocity(adata, mode="dynamical")

    return adata


# %% [markdown]
# ## Data loading

# %%
adatas = {}

adatas["dpt"] = get_dpt_adata()
adatas["dpt"].obs["obs_id"] = np.arange(0, adatas["dpt"].n_obs)
adatas["dpt"]

# %%
adatas["rna_velocity"] = get_velo_adata()
adatas["rna_velocity"]

# %% [markdown]
# ## CellRank analysis

# %% [markdown]
# ### Kernel

# %%
ptk = cr.kernels.PseudotimeKernel(adatas["dpt"], time_key="dpt_pseudotime").compute_transition_matrix(
    threshold_scheme="soft"
)

vk = cr.kernels.VelocityKernel(adatas["rna_velocity"]).compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adatas["rna_velocity"]).compute_transition_matrix()
vk_ck = 0.2 * ck + 0.8 * vk

kernels = {"PseudotimeKernel": ptk, "VelocityKernel": vk_ck}

# %% [markdown]
# ### Cross-boundary correctness score

# %%
cluster_key = "l2_cell_type"
rep = "MultiVI_latent"

score_df = []
for source, target in tqdm(STATE_TRANSITIONS):
    cbc_ptk = ptk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)
    cbc_velo = vk_ck.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

    score_df.append(
        pd.DataFrame(
            {
                "State transition": [f"{source} - {target}"] * len(cbc_ptk),
                "Log ratio": np.log((cbc_ptk + 1) / (cbc_velo + 1)),
            }
        )
    )
score_df = pd.concat(score_df)

# %%
dfs = []

ttest_res = {}
significances = {}

for source, target in STATE_TRANSITIONS:
    obs_mask = score_df["State transition"].isin([f"{source} - {target}"])
    a = score_df.loc[obs_mask, "Log ratio"].values
    b = np.zeros(len(a))

    ttest_res[f"{source} - {target}"] = ttest_ind(a, b, equal_var=False, alternative="greater")
    significances[f"{source} - {target}"] = get_significance(ttest_res[f"{source} - {target}"].pvalue)

significance_palette = {"n.s.": "#dedede", "*": "#90BAAD", "**": "#A1E5AB", "***": "#ADF6B1"}

palette = {
    state_transition: significance_palette[significance] for state_transition, significance in significances.items()
}

if running_in_notebook():
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(data=score_df, x="State transition", y="Log ratio", palette=palette, ax=ax)

        ax.tick_params(axis="x", rotation=45)

        handles = [Patch(label=label, facecolor=color) for label, color in significance_palette.items()]
        fig.legend(
            handles=handles,
            labels=["n.s.", "p<1e-1", "p<1e-2", "p<1e-3"],
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.025),
        )
        fig.tight_layout()
        plt.show()

        if SAVE_FIGURES:
            ax.set(xlabel="", xticklabels="", ylabel="", yticklabels="")
            fig.legends = []

            fig.savefig(
                FIG_DIR / "pseudotime_kernel" / "hematopoiesis" / f"log_ratio_cross_boundary.{FIGURE_FORMAT}",
                format=FIGURE_FORMAT,
                transparent=True,
                bbox_inches="tight",
            )
