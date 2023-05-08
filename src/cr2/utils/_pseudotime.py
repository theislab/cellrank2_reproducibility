def get_symmetric_transition_matrix(transition_matrix):
    """Symmetrize a given transition matrix."""
    sym_mat = (transition_matrix + transition_matrix.T) / 2

    # normalise transition matrix
    row_sums = sym_mat.sum(axis=1).A1
    sym_mat.data = sym_mat.data / row_sums[sym_mat.nonzero()[0]]

    return sym_mat
