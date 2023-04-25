from typing import Optional

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_state_purity(state_purity, color="grey", fpath: Optional[str] = None, format: str = "eps"):
    """Plot purity of a given state (e.g. macrostate or terminal state)."""
    if isinstance(color, str):
        kwargs = {"color": color}
    else:
        kwargs = {"palette": dict(zip(state_purity.keys(), color))}

    df = pd.DataFrame({"Purity": state_purity.values(), "State": state_purity.keys()}).sort_values(
        "Purity", ascending=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Purity", y="State", ax=ax, **kwargs)

    if fpath is not None:
        fig.savefig(fpath, format=format, transparent=True, bbox_inches="tight")
