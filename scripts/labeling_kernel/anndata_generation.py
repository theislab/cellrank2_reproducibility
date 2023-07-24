import argparse
import logging
import os
import sys
from functools import reduce

from tqdm import tqdm

import pandas as pd
from scipy.sparse import csr_matrix

from anndata import AnnData

sys.path.append("./")
from paths import DATA_DIR  # isort: skip # noqa: E402


logging.basicConfig(level=logging.INFO)


def _get_args():
    python_version = sys.version_info
    python_version = python_version.major + python_version.minor / 10

    parser = argparse.ArgumentParser()

    if python_version < 3.9:
        parser.add_argument("--enforce", action="store_true")
    else:
        parser.add_argument("--enforce", action=argparse.BooleanOptionalAction)

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    if (DATA_DIR / "sceu_organoid" / "processed" / "raw.h5ad").is_file() and not args.enforce:
        logging.info(
            f" Skipping AnnData generation since target `{DATA_DIR / 'sceu_organoid' / 'processed' / 'raw.h5ad'}` already exists.\n"
            "\tTo generate the file nonetheless, either remove the existing file or pass `enforce=True`\n"
        )
        sys.exit()

    # Data loading
    FILENAME_PREFIX = "GSE128365_SupplementaryData_organoids_"
    dfs = {}
    counts = {}

    df = pd.read_csv(DATA_DIR / "sceu_organoid" / "raw" / f"{FILENAME_PREFIX}labeled_splicedUMI.csv.gz")
    ensum_id = df.pop("ID")
    ensum_id.index = df["NAME"]
    df = df.set_index("NAME").T
    df.columns.name = None

    obs_names = df.index
    var_names = df.columns

    counts["labeled_spliced"] = csr_matrix(df.values)

    for count_type in tqdm(["labeled_unspliced", "unlabeled_spliced", "unlabeled_unspliced"]):
        df = pd.read_csv(DATA_DIR / "sceu_organoid" / "raw" / f"{FILENAME_PREFIX}{count_type}UMI.csv.gz")
        df = df.set_index("NAME").T
        df.columns.name = None

        df = df.loc[obs_names, var_names].astype(float)
        counts[count_type] = csr_matrix(df.values)

    metadata = pd.read_csv(
        DATA_DIR / "sceu_organoid" / "raw" / f"{FILENAME_PREFIX}cell_metadata.csv.gz", index_col=0
    ).T.loc[obs_names, :]
    metadata[["experiment", "labeling_time"]] = metadata["treatment_id"].str.split("_").tolist()

    metadata["cell_type"] = (
        metadata["som_cluster_id"]
        .replace(
            {
                "1": "Enterocytes",
                "3": "Enteroendocrine cells",
                "4": "Enteroendocrine progenitors",
                "5": "Tuft cells",
                "6": "TA cells",
                "8": "Stem cells",
                "9": "Paneth cells",
                "10": "Goblet cells",
                "11": "Stem cells",
            }
        )
        .astype("category")
    )

    # AnnData generation
    adata = AnnData(X=reduce(lambda x, y: x + y, counts.values()), layers=counts)
    adata.obs_names = obs_names
    adata.var_names = var_names

    adata.var["ensum_id"] = ensum_id.loc[adata.var_names]
    adata.obs = metadata.loc[
        adata.obs_names,
        [
            "experiment",
            "labeling_time",
            "cell_type",
            "well_id",
            "batch_id",
            "log10_gfp",
            "som_cluster_id",
            "monocle_branch_id",
            "monocle_pseudotime",
        ],
    ]

    adata.layers["total"] = adata.X.copy()
    adata.layers["labeled"] = adata.layers["labeled_unspliced"] + adata.layers["labeled_spliced"]
    adata.layers["unlabeled"] = adata.layers["unlabeled_unspliced"] + adata.layers["unlabeled_spliced"]

    adata.obsm["X_umap_paper"] = metadata[["rotated_umap1", "rotated_umap2"]].values.astype(float)

    cell_type_colors = {
        "2": "#023fa5",
        "7": "#7d87b9",
        "11": "#bec1d4",
        "Enterocytes": "#d6bcc0",
        "Enteroendocrine cells": "#bb7784",
        "Enteroendocrine progenitors": "#8e063b",
        "Goblet cells": "#4a6fe3",
        "Paneth cells": "#8595e1",
        "Stem cells": "#b5bbe3",
        "TA cells": "#e6afb9",
        "Tuft cells": "#e07b91",
    }
    adata.uns["cell_type_colors"] = [cell_type_colors[cell_type] for cell_type in adata.obs["cell_type"].cat.categories]

    # Saving data
    os.makedirs(DATA_DIR / "sceu_organoid" / "processed", exist_ok=True)
    adata.write(DATA_DIR / "sceu_organoid" / "processed" / "raw.h5ad")
