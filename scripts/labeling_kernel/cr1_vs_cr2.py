# %% [markdown]
# # Classical RNA velocity vs. labeling approach - TSI

# %% [markdown]
# ## Library imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys

import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

from cr2.analysis import plot_tsi

sys.path.extend(["../../", "."])
from paths import DATA_DIR, FIG_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / "labeling_kernel").mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "pdf"

# %% [markdown]
# ## Constants

# %% [markdown]
# ## Data loading

# %%
tsi_cr1 = pd.read_csv(DATA_DIR / "sceu_organoid" / "results" / "tsi-em_model.csv")
tsi_cr1.head()

# %%
tsi_cr2 = pd.read_csv(DATA_DIR / "sceu_organoid" / "results" / "tsi-labeling_approach.csv")
tsi_cr2.head()

# %% [markdown]
# ## Data preprocessing

# %%
tsi_cr1["method"] = "CellRank 1"
tsi_cr2["method"] = "CellRank 2"
df = pd.concat([tsi_cr1, tsi_cr2])

# %% [markdown]
# ## Plotting

# %%
palette = {"CellRank 1": "#0173b2", "CellRank 2": "#DE8F05", "Optimal identification": "#000000"}

if SAVE_FIGURES:
    fname = FIG_DIR / "labeling_kernel" / f"tsi_ranking.{FIGURE_FORMAT}"
else:
    fname = None

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    plot_tsi(df=df, palette=palette, fname=fname)
    plt.show()
