from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes


def plot_state_purity(state_purity, fpath: Optional[str] = None, format: str = "eps", **kwargs):
    """Plot purity of a given state (e.g. macrostate or terminal state)."""
    df = pd.DataFrame({"Purity": state_purity.values(), "State": state_purity.keys()}).sort_values(
        "Purity", ascending=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Purity", y="State", ax=ax, **kwargs)

    if fpath is not None:
        fig.savefig(fpath, format=format, transparent=True, bbox_inches="tight")


def plot_tsi(
    df,
    n_macrostates: Optional[int] = None,
    x_offset: Tuple[float, float] = (0.2, 0.2),
    y_offset: Tuple[float, float] = (0.1, 0.1),
    figsize: Tuple[float, float] = (6, 4),
    dpi: Optional[int] = None,
    fname: Optional[Path] = None,
    **kwargs,
) -> Axes:
    """Plot terminal state identificiation (TSI).

    Requires computing TSI with :meth:`tsi` first.

    Parameters
    ----------
    n_macrostates
        Maximum number of macrostates to consider. Defaults to using all.
    x_offset
        Offset of x-axis.
    y_offset
        Offset of y-axis.
    figsize
        Size of the figure.
    dpi
        Dots per inch.
    fname
        File name to save plot under.
    kwargs
        Keyword arguments for :func:`~seaborn.lineplot`.

    Returns
    -------
    Plot TSI of the kernel and an optimal identification strategy.
    """
    if n_macrostates is not None:
        df = df.loc[df["number_of_macrostates"] <= n_macrostates, :]
    df["line_style"] = "-"

    optimal_identification = df[["number_of_macrostates", "optimal_identification"]]
    optimal_identification = optimal_identification.rename(
        columns={"optimal_identification": "identified_terminal_states"}
    )
    optimal_identification["method"] = "Optimal identification"
    optimal_identification["line_style"] = "--"

    df = df[["number_of_macrostates", "identified_terminal_states", "method", "line_style"]]

    df = pd.concat([df, optimal_identification])
    df = df.drop_duplicates()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
    sns.lineplot(
        data=df,
        x="number_of_macrostates",
        y="identified_terminal_states",
        hue="method",
        style="line_style",
        drawstyle="steps-post",
        ax=ax,
        **kwargs,
    )

    ax.set_xticks(df["number_of_macrostates"].unique().astype(int))
    # Plot is generated from large to small values on the x-axis
    for label_id, label in enumerate(ax.xaxis.get_ticklabels()[::-1]):
        if ((label_id + 1) % 5 != 0) and label_id != 0:
            label.set_visible(False)
    ax.set_yticks(df["identified_terminal_states"].unique())

    x_min = df["number_of_macrostates"].min() - x_offset[0]
    x_max = df["number_of_macrostates"].max() + x_offset[1]
    y_min = df["identified_terminal_states"].min() - y_offset[0]
    y_max = df["identified_terminal_states"].max() + y_offset[1]
    ax.set(
        xlim=[x_min, x_max],
        ylim=[y_min, y_max],
        xlabel="Number of macrostates",
        ylabel="Identified terminal states",
    )

    ax.get_legend().remove()

    n_methods = len(df["method"].unique())
    handles, labels = ax.get_legend_handles_labels()
    handles[n_methods].set_linestyle("--")
    handles = handles[: (n_methods + 1)]
    labels = labels[: (n_methods + 1)]
    labels[0] = "Method"
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=(n_methods + 1), bbox_to_anchor=(0.5, -0.1))

    if fname is not None:
        format = fname.suffix[1:]
        fig.savefig(
            fname=fname,
            format=format,
            transparent=True,
            bbox_inches="tight",
        )

    return ax
