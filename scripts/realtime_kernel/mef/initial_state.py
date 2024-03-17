# %% [markdown]
# # Initial state identification for mouse embryonic fibroblasts

# %% [markdown]
# ## Library imports

# %%
import sys

from scipy.sparse import load_npz

import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv

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
rtk.transition_matrix = rtk.transition_matrix.T

# %% [markdown]
# ## Terminal state estimation

# %%
estimator = cr.estimators.GPCCA(rtk)

# %%
estimator.compute_schur(n_components=5)
estimator.plot_spectrum(real_only=True)

# %%
estimator.compute_macrostates(n_states=1, cluster_key="cell_sets")
estimator.plot_macrostates(which="all", basis="force_directed", legend_loc="right", s=100)
