"""Projection utilities shared by SAGE.

SAGE removes risk directions from prompt embeddings by projecting them onto
the orthogonal complement of a concept subspace.  The only linear-algebra
primitive needed for this is the orthogonal projection matrix

    P = E (E^T E)^{-1} E^T

built from the column space of E (each column is one basis vector, e.g. a
pooled concept embedding or a leave-one-out pooled prompt embedding).
"""

import torch


def projection_matrix(E_columns: torch.Tensor) -> torch.Tensor:
    """Build the orthogonal projection matrix onto span(E_columns).

    Args:
        E_columns: [D, K] tensor whose columns span the target subspace.

    Returns:
        P: [D, D] projection matrix, P = E (E^T E)^{-1} E^T (via pinverse so
        that degenerate/rank-deficient concept sets still work).
    """
    E = E_columns
    return E @ torch.pinverse(E.T @ E) @ E.T
