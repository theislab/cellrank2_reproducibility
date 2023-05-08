from typing import Literal, Optional, Union

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scvelo as scv


def plot_state_purity(state_purity, fpath: Optional[str] = None, format: str = "eps", **kwargs):
    """Plot purity of a given state (e.g. macrostate or terminal state)."""
    df = pd.DataFrame({"Purity": state_purity.values(), "State": state_purity.keys()}).sort_values(
        "Purity", ascending=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Purity", y="State", ax=ax, **kwargs)

    if fpath is not None:
        fig.savefig(fpath, format=format, transparent=True, bbox_inches="tight")


def plot_states(
    adata,
    estimator,
    which: Literal["macrostates", "terminal_states"],
    basis: str,
    inplace: bool = False,
    fpath: Optional[str] = None,
    format: str = "eps",
    dpi: Union[int, str] = "figure",
    **kwargs,
):
    if not inplace:
        adata = adata.copy()

    states = getattr(estimator, which).cat.categories.tolist()
    if which == "macrostates":
        states = estimator._macrostates
    elif which == "terminal_states":
        states = estimator._term_states
    state_names = states.assignment.cat.categories.tolist()

    adata.obs[which] = states.assignment.astype(str).astype("category").cat.reorder_categories(["nan"] + state_names)
    if which == "macrostates":
        adata.uns[f"{which}_colors"] = ["#dedede"] + states.colors
    else:
        adata.uns[f"{which}_colors"] = ["#dedede"] + states.colors.tolist()
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(
        adata,
        basis=basis,
        c=which,
        add_outline=state_names,
        ax=ax,
        **kwargs,
    )

    if fpath is not None:
        fig.savefig(fpath, format=format, transparent=True, bbox_inches="tight", dpi=dpi)
