from typing import Literal

import pandas as pd


def get_state_purity(adata, estimator, states: Literal["macrostates", "terminal_states"], obs_col: str):
    """Calculate purity of each state of a state type (e.g. each macrostate)."""
    states = getattr(estimator, states)

    max_obs_count_per_state = (
        pd.DataFrame({"states": states, "obs_col": adata.obs[obs_col]})[~states.isnull()]
        .groupby(["states", "obs_col"])
        .size()
        .reset_index()
        .rename(columns={0: "group_counts"})[["states", "group_counts"]]
        .groupby("states")
        .max()["group_counts"]
    )

    return (max_obs_count_per_state / states.value_counts()).to_dict()
