"""Framework-neutral autodiff-bridge helpers shared by the JAX and
PyTorch frontends (pounce#109).

The numerical core (the Rust IPM, exposed via :class:`pounce._pounce.Problem`)
and the sparsity-detection / CPR-coloring bookkeeping around it are
autodiff-framework-agnostic — they operate on plain NumPy arrays. Both
``pounce.jax`` and ``pounce.torch`` build a cyipopt-shaped problem object
from traced ``f(x)`` / ``g(x)``; the only piece that differs between them
is the *array namespace* used to evaluate the derivatives. Everything
here is pure NumPy and is imported by both adapters so the sparsity
pattern detection and column coloring live in exactly one place.

* :func:`_detect_pattern_2d_multi` / :func:`_detect_pattern_lower_multi`
  turn a set of dense probe matrices into ``(rows, cols)`` nonzero
  patterns (cyipopt convention).
* :func:`_color_columns` is the CPR (Curtis–Powell–Reid) distance-1
  greedy coloring of the column-intersection graph used to compress the
  sparse Jacobian / Hessian into one directional derivative per color
  (issue #83).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# Threshold below which a Jacobian/Hessian entry is treated as
# structurally zero during the pattern probe. Tight enough to reject
# genuine zeros from constant terms, loose enough that random probe
# values don't accidentally hit a numerical cancellation that would
# drop a real entry.
_SPARSITY_EPS = 1e-12


def _to_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _union_mask(denses) -> np.ndarray:
    """Boolean nonzero mask over one or more dense probe matrices.

    A nonzero in *any* probe is treated as structurally nonzero, so a
    value-dependent zero that a single probe happens to hit doesn't
    drop a real entry from the pattern.
    """
    mask = None
    for dense in denses:
        m = np.abs(np.asarray(dense)) > _SPARSITY_EPS
        mask = m if mask is None else (mask | m)
    return mask


def _detect_pattern_2d_multi(denses) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(_union_mask(denses))
    return rows.astype(np.int64), cols.astype(np.int64)


def _detect_pattern_lower_multi(denses) -> tuple[np.ndarray, np.ndarray]:
    """Lower-triangle sparsity pattern of a symmetric matrix, unioned
    across probes."""
    mask = _union_mask(denses)
    n = mask.shape[0]
    rows, cols = np.tril_indices(n)
    keep = mask[rows, cols]
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


def _detect_pattern_2d(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _detect_pattern_2d_multi([dense])


def _detect_pattern_lower(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower-triangle sparsity pattern of a symmetric matrix."""
    return _detect_pattern_lower_multi([dense])


def _color_columns(
    rows: np.ndarray, cols: np.ndarray, n: int,
) -> tuple[np.ndarray, int]:
    """Greedy distance-1 coloring of the column-intersection graph.

    Two columns *conflict* (must get different colors) when they share
    a nonzero row. Columns that share a color are therefore structurally
    orthogonal: a single directional derivative seeded on all of them at
    once recovers each column's entries unambiguously by row, because no
    row receives a contribution from more than one of them.

    This is the CPR (Curtis–Powell–Reid) compression used for both the
    sparse Jacobian and (on the symmetrized pattern) the sparse Hessian.
    Greedy coloring is not optimal but is cheap and gives ``k`` close to
    the maximum number of nonzeros in any row, which is what bounds the
    AD-pass count.

    Returns ``(colors, num_colors)`` where ``colors[j]`` is the color of
    column ``j`` (columns with no nonzeros get color 0).
    """
    cols_in_row: dict[int, list[int]] = defaultdict(list)
    rows_of_col: dict[int, list[int]] = defaultdict(list)
    for r, c in zip(rows.tolist(), cols.tolist()):
        cols_in_row[r].append(c)
        rows_of_col[c].append(r)

    colors = np.full(int(n), -1, dtype=np.int64)
    num_colors = 0
    for j in range(int(n)):
        forbidden: set[int] = set()
        for r in rows_of_col.get(j, ()):
            for c2 in cols_in_row[r]:
                cc = colors[c2]
                if c2 != j and cc >= 0:
                    forbidden.add(int(cc))
        c = 0
        while c in forbidden:
            c += 1
        colors[j] = c
        if c + 1 > num_colors:
            num_colors = c + 1
    # Empty matrix: still report one (unused) color so seed shapes are
    # well-defined.
    return colors, max(num_colors, 1)
