# %% [markdown]
# # Palantir pseudotime on mouse embryonic fibroblasts

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import cellrank as cr
import palantir
import scanpy as sc
import scvelo as scv

sys.path.insert(0, "../../../")
from paths import DATA_DIR, FIG_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
# set verbosity levels
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %%
mpl.use("module://matplotlib_inline.backend_inline")
mpl.rcParams["backend"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "realtime_kernel" / "mef", exist_ok=True)

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
sc.pp.neighbors(adata, random_state=0)

# %% [markdown]
# ## Palantir pseudotime construction

# %%
pc_projection = pd.DataFrame(adata.obsm["X_pca"].copy(), index=adata.obs_names)

# %%
# diffusion maps
diff_maps = palantir.utils.run_diffusion_maps(pc_projection, n_components=5)

# %%
# multiscale space
multiscale_space = palantir.utils.determine_multiscale_space(diff_maps)

# %%
magic_imputed = palantir.utils.run_magic_imputation(adata, diff_maps)

# %%
# See DPT notebook for root cell identification
root_idx = 433
palantir_res = palantir.core.run_palantir(
    multiscale_space, adata.obs_names[root_idx], use_early_cell_as_start=True, num_waypoints=500
)

# %%
adata.obs["palantir_pseudotime"] = palantir_res.pseudotime

# %%
scv.pl.scatter(
    adata,
    c=["palantir_pseudotime", "day"],
    basis="force_directed",
    legend_loc=False,
    color_map="gnuplot",
)

# %%
adata.obs["day"] = adata.obs["day"].astype("category")

sns.set_style(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 4))
sc.pl.violin(adata, keys=["palantir_pseudotime"], groupby="day", rotation=90, ax=ax)

if SAVE_FIGURES:
    ax.hlines([0, 0.25, 0.5, 0.75, 1], xmin=-0.75, xmax=39, linestyles="solid", colors="black", zorder=0)
    ax.axis("off")
    fig.savefig(
        FIG_DIR / "realtime_kernel" / "mef" / "palantir_vs_time_point.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )

sns.reset_orig()

# %%
spearmanr(adata.obs["palantir_pseudotime"].values, adata.obs["day"].astype(float).values)
