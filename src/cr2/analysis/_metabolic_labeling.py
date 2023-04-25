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
