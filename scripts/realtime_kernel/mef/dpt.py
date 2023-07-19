# %% [markdown]
# # Diffusion pseudotime on mouse embryonic fibroblasts

# %% [markdown]
# ## Library imports

# %%
import os
import sys

from scipy.stats import spearmanr

import matplotlib.pyplot as plt
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
SAVE_FIGURES = True
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

# %%
sc.pp.neighbors(adata, random_state=0)

# %% [markdown]
# ## Compute transition matrix

# %% [markdown]
# ## Pseudotime construction

# %%
sc.tl.diffmap(adata)

# %%
root_idx = 433  # adata.obsm['X_diffmap'][:, 1].argmax()
scv.pl.scatter(adata, basis="diffmap", color=["cell_sets", root_idx], components=["1, 2"], size=25)

# %%
adata.uns["iroot"] = root_idx  # np.flatnonzero(adata.obs['day'] == '0.0')[0]
sc.tl.dpt(adata)

# %%
scv.pl.scatter(adata, basis="force_directed", c=["day", "dpt_pseudotime"], legend_loc="none", cmap="gnuplot")

# %%
adata.obs["day"] = adata.obs["day"].astype("category")

sns.set_style(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 4))
sc.pl.violin(adata, keys=["dpt_pseudotime"], groupby="day", rotation=90, ax=ax)

if SAVE_FIGURES:
    ax.hlines([0, 0.25, 0.5, 0.75, 1], xmin=-0.75, xmax=39, linestyles="solid", colors="black", zorder=0)
    ax.axis("off")
    fig.savefig(
        FIG_DIR / "realtime_kernel" / "mef" / "dpt_vs_time_point.eps",
        format="eps",
        transparent=True,
        bbox_inches="tight",
    )
sns.reset_orig()

# %%
spearmanr(adata.obs["dpt_pseudotime"].values, adata.obs["day"].astype(float).values)
