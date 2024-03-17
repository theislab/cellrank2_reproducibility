# %% [markdown]
# # Pharyngeal endoderm development analysis with the RealTimeKernel

# %% [markdown]
# ## Import packages

# %%
import sys

import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

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
    (FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm").mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "pdf"

# %%
(DATA_DIR / "pharyngeal_endoderm" / "processed").mkdir(parents=True, exist_ok=True)

# %%
N_JOBS = 8

# %% [markdown]
# ## Constants

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "pharyngeal_endoderm" / "processed" / "adata_velo.h5ad")

adata.obs["cluster_name"] = (
    adata.obs["cluster_name"]
    .astype(str)
    .astype("category")
    .cat.rename_categories({"nan": "progenitors"})
    .cat.reorder_categories(["progenitors"] + adata.obs["cluster_name"].cat.categories.tolist())
)
adata.uns["cluster_name_colors"] = [
    "#dedede",
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
]

adata

# %% [markdown]
# ## Data preprocessing

# %%
scv.pp.filter_and_normalize(adata, min_counts=20, n_top_genes=2000)

# %%
sc.pp.pca(adata)
sc.pp.neighbors(adata, random_state=0)

scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

# %%
if running_in_notebook():
    scv.pl.scatter(
        adata, basis="umap", c="cluster_name", title="", dpi=250, legend_fontsize=12, legend_fontweight="normal"
    )

# %% [markdown]
# ## RNA velocity inference

# %%
if (DATA_DIR / "pharyngeal_endoderm" / "results" / "adata_velo_fit-full_data.h5ad").is_file():
    adata = sc.read(DATA_DIR / "pharyngeal_endoderm" / "results" / "adata_velo_fit-full_data.h5ad")
else:
    scv.tl.recover_dynamics(adata, n_jobs=N_JOBS)
    scv.tl.velocity(adata, mode="dynamical")
    adata.write(DATA_DIR / "pharyngeal_endoderm" / "results" / "adata_velo_fit-full_data.h5ad", compression="gzip")

# %% [markdown]
# ## CellRank analysis

# %% [markdown]
# ### Kernel

# %%
vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

combined_kernel = 0.8 * vk + 0.2 * ck

# %% [markdown]
# ## Estimator

# %%
estimator = cr.estimators.GPCCA(combined_kernel)

# %%
estimator.compute_schur(n_components=20)
estimator.plot_spectrum(real_only=True)

# %%
terminal_states = [
    "late_Dlx2",
    "late_Runx1",
    "parathyroid",
    "cTEC",
    "mTEC",
    "late_Grhl3",
    "late_Pitx2",
    "ubb",
    "thyroid",
    "late_Dmrt1",
    "late_respiratory",
]
cluster_key = "cluster_name"

if (DATA_DIR / "pharyngeal_endoderm" / "results" / "tsi-full_data-vk.csv").is_file():
    tsi_df = pd.read_csv(DATA_DIR / "pharyngeal_endoderm" / "results" / "tsi-full_data-vk.csv")
    estimator._tsi = AnnData(tsi_df, uns={"terminal_states": terminal_states, "cluster_key": cluster_key})
    tsi_score = estimator.tsi(n_macrostates=20, terminal_states=terminal_states, cluster_key=cluster_key)
else:
    tsi_score = estimator.tsi(n_macrostates=20, terminal_states=terminal_states, cluster_key=cluster_key)
    estimator._tsi.to_df().to_csv(DATA_DIR / "pharyngeal_endoderm" / "results" / "tsi-full_data-vk.csv", index=False)

print(f"TSI score: {tsi_score:.2f}")

# %%
# For nice name in figure legend
estimator.kernel.__class__.__name__ = "VelocityKernel"
palette = {"VelocityKernel": "#0173b2", "Optimal identification": "#000000"}

if SAVE_FIGURES:
    fpath = FIG_DIR / "realtime_kernel" / "pharyngeal_endoderm" / f"tsi-full_data-vk.{FIGURE_FORMAT}"
else:
    fpath = None

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    estimator.plot_tsi(palette=palette, save=fpath)
    plt.show()
