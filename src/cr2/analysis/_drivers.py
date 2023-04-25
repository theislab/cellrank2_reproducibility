from typing import List

import pandas as pd


def get_var_ranks(
    var_names: List[str], drivers: pd.DataFrame, macrostate: str, var_type: str, model: str, threshold: int = 100
):
    """Get ranking of a set of variables towards a given macrostate."""
    _df = drivers.loc[
        var_names, [f"{macrostate}_corr", f"{macrostate}_pval", f"Corr. rank - {macrostate}"]
    ].sort_values(by=[f"Corr. rank - {macrostate}"])

    _df["Type"] = var_type
    _df["Algorithm"] = model

    print(
        f"{var_type} towards {macrostate} for {model} in top {threshold}: "
        f"{(_df[f'Corr. rank - {macrostate}'] <= threshold).sum()} (out of {_df.shape[0]})"
    )

    return _df
