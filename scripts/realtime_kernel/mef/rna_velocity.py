# %% [markdown]
# # RNA velocity analysis on mouse embryonic fibroblasts

# %% [markdown]
# ## Library imports

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

# %%
N_JOBS = 8

# %% [markdown]
# ## Function definitions


# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "mef" / "processed" / "mef_velo.h5ad")
adata

# %%
adata = adata[adata.obs["serum"] == "True"].copy()

adata.obs["day"] = adata.obs["day"].astype(float)
adata.uns["cell_sets_colors"] = sns.color_palette("colorblind").as_hex()[: len(adata.obs["cell_sets"].cat.categories)]

# %%
if running_in_notebook():
    scv.pl.scatter(adata, basis="force_directed", c=["day", "cell_sets"], legend_loc="right", cmap="gnuplot")

# %% [markdown]
# ## Data pre-processing and RNA velocity inference

# %%
if (DATA_DIR / "mef" / "results" / "adata_velo_fit.h5ad").is_file():
    adata = sc.read(DATA_DIR / "mef" / "results" / "adata_velo_fit.h5ad")
else:
    scv.pp.filter_and_normalize(adata, min_counts=20, n_top_genes=2000)

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, random_state=0)

    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    scv.tl.recover_dynamics(adata, n_jobs=N_JOBS)
    scv.tl.velocity(adata, mode="dynamical")
    adata.write(DATA_DIR / "mef" / "results" / "adata_velo_fit.h5ad", compression="gzip")

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
terminal_states = ["Neural", "IPS", "Trophoblast", "Stromal"]
cluster_key = "cell_sets"

if (DATA_DIR / "mef" / "results" / "tsi-vk.csv").is_file():
    tsi_df = pd.read_csv(DATA_DIR / "mef" / "results" / "tsi-vk.csv")
    estimator._tsi = AnnData(tsi_df, uns={"terminal_states": terminal_states, "cluster_key": cluster_key})
    tsi_score = estimator.tsi(n_macrostates=10, terminal_states=terminal_states, cluster_key=cluster_key)
else:
    tsi_score = estimator.tsi(n_macrostates=10, terminal_states=terminal_states, cluster_key=cluster_key)
    estimator._tsi.to_df().to_csv(DATA_DIR / "mef" / "results" / "tsi-vk.csv", index=False)

print(f"TSI score: {tsi_score:.2f}")

# %%
# For nice name in figure legend
estimator.kernel.__class__.__name__ = "VelocityKernel"
palette = {"VelocityKernel": "#0173b2", "Optimal identification": "#000000"}

if SAVE_FIGURES:
    fpath = FIG_DIR / "realtime_kernel" / "mef" / f"tsi-vk.{FIGURE_FORMAT}"
else:
    fpath = None

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    estimator.plot_tsi(palette=palette, save=fpath)
    plt.show()
