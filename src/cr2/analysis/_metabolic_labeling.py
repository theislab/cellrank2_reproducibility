import numpy as np

import scvelo as scv


def get_consistency(X, adj_mat):
    n_obs = X.shape[0]
    X -= X.mean(axis=1)[:, None]
    X_norm = scv.core.l2_norm(X, axis=1)
    pearson_r = np.zeros(n_obs)

    for obs_id in range(n_obs):
        neighbor_ids = adj_mat[obs_id, :].nonzero()[1]
        neighbors = X[neighbor_ids, :]
        neighbors -= neighbors.mean(axis=1)[:, None]
        pearson_r[obs_id] = np.mean(
            np.einsum("ij, j", neighbors, X[obs_id]) / (scv.core.l2_norm(neighbors, axis=1) * X_norm[obs_id])[None, :]
        )

    return pearson_r


def prepare_data_for_dynamo(adata, experiment_type: str, inplace: bool = True):
    if not inplace:
        adata = adata.copy()

    adata.uns["pp"] = {
        "has_splicing": False,
        "has_labeling": True,
        "splicing_labeling": False,
        "has_protein": False,
        "tkey": "time",
        "experiment_type": experiment_type,
        "norm_method": None,
    }
    adata.uns["pca_fit"] = {}

    adata.obs["pass_basic_filter"] = True

    adata.var["pass_basic_filter"] = True
    adata.var["use_for_pca"] = True

    adata.uns["PCs"] = adata.varm["PCs"].copy()
    adata.uns["pca_mean"] = adata.X.mean(axis=0).A1

    if not inplace:
        return adata
