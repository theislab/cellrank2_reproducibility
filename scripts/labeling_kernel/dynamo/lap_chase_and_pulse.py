# %% [markdown]
# # Least action path analysis with Dynamo

# %% [markdown]
# Driver analysis on scEU-seq organoid data using Dynamo's LAP analysis

# %%
import itertools
import sys

import numpy as np
import pandas as pd

import dynamo as dyn
import scanpy as sc

sys.path.extend(["../../../", "."])
from paths import DATA_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3

# %% [markdown]
# ## Constants

# %%
TERMINAL_STATES = ["Enterocytes", "Enteroendocrine progenitors", "Goblet cells", "Paneth cells"]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "sceu_organoid" / "processed" / "adata_dynamo-chase_and_pulse-2000features.h5ad")
adata

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

# %% [markdown]
# ## Preprocessing

# %%
markers = {}
markers["Goblet cells"] = goblet_markers.union(goblet_regulators).union(goblet_and_paneth_regulators)
markers["Paneth cells"] = paneth_markers.union(goblet_and_paneth_regulators)
markers["Enteroendocrine progenitors"] = eec_markers.union(eec_progenitor_markers)
markers["Enterocytes"] = enterocyte_markers.union(enterocyte_progenitor_markers)

# %%
dyn.tl.reduceDimension(adata, layer="X_new", enforce=True)

# %% [markdown]
# ## Least action path analysis

# %%
dyn.tl.cell_velocities(adata, ekey="M_n", vkey="velocity_N", enforce=True)
dyn.vf.VectorField(adata, basis="umap")
dyn.vf.topography(adata)

# %%
adata.obsm["X_pca_orig"] = adata.obsm["X_pca"].copy()
adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :30]

# %%
dyn.tl.cell_velocities(adata, ekey="M_n", vkey="velocity_N", basis="pca")
dyn.vf.VectorField(adata, basis="pca")

# %%
df = pd.DataFrame(adata.uns["VecFld_umap"]["Xss"], columns=["umap_1", "umap_2"])
df["fixed_point_type"] = adata.uns["VecFld_umap"]["ftype"]
df["fixed_point_type"].replace({-1: "stable", 0: "saddle", 1: "unstable"}, inplace=True)

neighbor_idx = dyn.tools.utils.nearest_neighbors(df[["umap_1", "umap_2"]], adata.obsm["X_new_umap"])
df["cell_type"] = [
    adata.obs.loc[adata.obs_names[neighbors], "cell_type"].mode().values[0] for neighbors in neighbor_idx
]

initial_obs = dyn.tools.utils.nearest_neighbors(
    df.loc[(df["fixed_point_type"] == "unstable") & (df["cell_type"] == "Stem cells"), ["umap_1", "umap_2"]].values,
    adata.obsm["X_new_umap"],
    k=10,
).flatten()

# %%
identified_terminal_states = list(
    set(TERMINAL_STATES).intersection(df.loc[df["fixed_point_type"] == "stable", "cell_type"].unique())
)

terminal_obs = {}
for terminal_state in identified_terminal_states:
    terminal_obs[terminal_state] = dyn.tools.utils.nearest_neighbors(
        df.loc[(df["fixed_point_type"] == "stable") & (df["cell_type"] == terminal_state), ["umap_1", "umap_2"]].values,
        adata.obsm["X_new_umap"],
        k=10,
    ).flatten()

# %%
dyn.tl.neighbors(adata, basis="umap", result_prefix="umap")

# %% [markdown]
# ### Dynamo stable fixed points

# %%
gene_ranks = {}

for terminal_state in terminal_obs.keys():
    rankings = []

    lst = list(itertools.product(initial_obs, terminal_obs[terminal_state]))

    np.random.seed(0)
    state_tuples = np.random.choice(len(lst), size=10, replace=False).tolist()

    initial_obs_, terminal_obs_ = zip(*[lst[idx] for idx in state_tuples])

    laps = dyn.pd.least_action(
        adata,
        init_cells=list(initial_obs_),
        target_cells=list(terminal_obs_),
    )

    gtraj = dyn.pd.GeneTrajectory(adata)

    for lap_id, lap in enumerate(laps):
        gtraj.from_pca(lap.X, t=lap.t)
        gtraj.calc_msd()

        ranking = dyn.vf.rank_genes(adata, "traj_msd")
        ranking = ranking.reset_index().rename(columns={"index": f"Corr. rank - {terminal_state}", "all": "Gene"})
        ranking["Algorithm"] = "Dynamo"
        ranking["Run"] = lap_id
        rankings.append(ranking)

    gene_ranks[terminal_state] = pd.concat(rankings)
    gene_ranks[terminal_state] = gene_ranks[terminal_state].loc[
        gene_ranks[terminal_state]["Gene"].isin(markers[terminal_state])
    ]

# %%
for terminal_state in gene_ranks.keys():
    gene_ranks[terminal_state].set_index("Gene", inplace=True)
    gene_ranks[terminal_state].index.name = None
    gene_ranks[terminal_state].to_csv(
        DATA_DIR
        / "sceu_organoid"
        / "results"
        / f"gene_ranks_{terminal_state}-chase_and_pulse-dynamo_terminal_states-dynamo.csv"
    )

# %% [markdown]
# ### CellRank terminal states

# %%
cr_terminal_states = pd.read_csv(
    DATA_DIR / "sceu_organoid" / "results" / "cr_terminal_states.csv", index_col=0
).reset_index(drop=True)["terminal_state"]
terminal_obs = {
    terminal_state: cr_terminal_states.index[cr_terminal_states == terminal_state].to_numpy()
    for terminal_state in cr_terminal_states.astype("category").cat.categories
}

# %%
gene_ranks = {}

for terminal_state in terminal_obs.keys():
    rankings = []

    lst = list(itertools.product(initial_obs, terminal_obs[terminal_state]))

    np.random.seed(0)
    state_tuples = np.random.choice(len(lst), size=10, replace=False).tolist()

    initial_obs_, terminal_obs_ = zip(*[lst[idx] for idx in state_tuples])

    laps = dyn.pd.least_action(
        adata,
        init_cells=list(initial_obs_),
        target_cells=list(terminal_obs_),
    )

    gtraj = dyn.pd.GeneTrajectory(adata)

    for lap_id, lap in enumerate(laps):
        gtraj.from_pca(lap.X, t=lap.t)
        gtraj.calc_msd()

        ranking = dyn.vf.rank_genes(adata, "traj_msd")
        ranking = ranking.reset_index().rename(columns={"index": f"Corr. rank - {terminal_state}", "all": "Gene"})
        ranking["Algorithm"] = "Dynamo"
        ranking["Run"] = lap_id
        rankings.append(ranking)

    gene_ranks[terminal_state] = pd.concat(rankings)
    gene_ranks[terminal_state] = gene_ranks[terminal_state].loc[
        gene_ranks[terminal_state]["Gene"].isin(markers[terminal_state])
    ]

# %%
for terminal_state in gene_ranks.keys():
    gene_ranks[terminal_state].set_index("Gene", inplace=True)
    gene_ranks[terminal_state].index.name = None
    gene_ranks[terminal_state].to_csv(
        DATA_DIR
        / "sceu_organoid"
        / "results"
        / f"gene_ranks_{terminal_state}-chase_and_pulse-cr_terminal_states-dynamo.csv"
    )
