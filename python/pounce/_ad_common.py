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

* :func:`_detect_pattern_blocked` sweeps the matrix a *block of slices*
  at a time (JVPs for columns, VJPs for rows) and accumulates the
  nonzero index pairs, so detection never materializes the full ``(m, n)``
  Jacobian or ``(n, n)`` Hessian (issue #464).
* :func:`_detect_pattern_2d_multi` / :func:`_detect_pattern_lower_multi`
  turn a set of dense probe matrices into ``(rows, cols)`` nonzero
  patterns (cyipopt convention). Kept for callers that already hold a
  dense matrix; the frontends use the blocked sweep instead.
* :func:`_normalize_user_pattern` validates a caller-supplied pattern
  (``jac_pattern=`` / ``hess_pattern=``), which skips detection entirely.
* :func:`_color_columns` is the CPR (Curtis–Powell–Reid) distance-1
  greedy coloring of the column-intersection graph used to compress the
  sparse Jacobian / Hessian into one directional derivative per color
  (issue #83).
"""

from __future__ import annotations

import numpy as np

from ._pounce import DEFAULT_ACTIVE_TOL

# Single source of truth for the differentiable-solve active-set
# tolerance, re-exported from the Rust producer (the `DiffHandoff`
# contract: see ``dev-notes/diff-handoff-contract.md``). A bound or
# constraint multiplier above this is treated as active. Every JAX /
# torch backward imports ``ACTIVE_TOL`` from here rather than hardcoding
# ``1e-6``, so the producer's ``info["active_tol"]`` and every consumer's
# threshold are guaranteed to agree.
ACTIVE_TOL: float = DEFAULT_ACTIVE_TOL

# Threshold below which a Jacobian/Hessian entry is treated as
# structurally zero during the pattern probe. Tight enough to reject
# genuine zeros from constant terms, loose enough that random probe
# values don't accidentally hit a numerical cancellation that would
# drop a real entry.
_SPARSITY_EPS = 1e-12

# Peak bytes one probe block of the Jacobian / Hessian is allowed to
# occupy. Detection sweeps ``block = _PROBE_BLOCK_BYTES / (8 * width)``
# slices at a time, so build memory is bounded by this constant plus the
# detected nonzeros — not by ``O(n²)`` (issue #464). The AD pass count is
# unchanged (``jacfwd`` is itself a vmapped JVP over the identity basis);
# only the peak allocation differs.
_PROBE_BLOCK_BYTES = 32 << 20  # 32 MiB


def _to_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _probe_block_size(width: int, max_bytes: int = _PROBE_BLOCK_BYTES) -> int:
    """How many rows (or columns) of a ``width``-long slice to
    materialize per probe block, under a fixed byte budget."""
    width = int(width)
    if width <= 0:
        return 1
    return max(1, int(max_bytes) // (8 * width))


def _sorted_pattern(rows, cols) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate and sort an index pair into row-major order — the
    order ``np.nonzero`` on a dense mask produces, so a blocked or
    caller-supplied pattern is indistinguishable from a dense probe."""
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    if rows.size == 0:
        return rows, cols
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    keep = np.ones(rows.size, dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    return rows[keep], cols[keep]


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


def _detect_pattern_blocked(
    n_slices: int,
    width: int,
    eval_block,
    *,
    by_row: bool,
    lower: bool = False,
    max_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero pattern of a matrix swept a block of slices at a time.

    ``eval_block(start, stop)`` returns an iterable of dense
    ``(stop - start, width)`` arrays — one per random probe — holding
    rows ``start:stop`` of the matrix when ``by_row`` is true, or columns
    ``start:stop`` (transposed, so the swept index is first) when it is
    false. Blocks are unioned across probes and reduced to index pairs
    immediately, so peak memory is one block plus the nonzeros found so
    far rather than the whole ``O(n²)`` matrix (issue #464).

    ``lower=True`` keeps only ``row >= col``, giving the lower triangle of
    a symmetric matrix directly. ``max_bytes`` defaults to the module's
    :data:`_PROBE_BLOCK_BYTES`, read at call time so tests can shrink it
    to force a multi-block sweep on a small matrix.
    """
    if max_bytes is None:
        max_bytes = _PROBE_BLOCK_BYTES
    n_slices = int(n_slices)
    block = min(max(n_slices, 1), _probe_block_size(width, max_bytes))
    rows_acc: list[np.ndarray] = []
    cols_acc: list[np.ndarray] = []
    for start in range(0, n_slices, block):
        stop = min(start + block, n_slices)
        swept, other = np.nonzero(_union_mask(eval_block(start, stop)))
        if by_row:
            r, c = swept + start, other
        else:
            r, c = other, swept + start
        if lower:
            keep = r >= c
            r, c = r[keep], c[keep]
        rows_acc.append(r)
        cols_acc.append(c)
    if not rows_acc:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return _sorted_pattern(np.concatenate(rows_acc), np.concatenate(cols_acc))


def _as_index_array(a, what: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{what} must be a 1-D index array, got shape {arr.shape}")
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{what} must hold integer indices, got dtype {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _normalize_user_pattern(
    pattern, name: str, n_rows: int, n_cols: int, *, lower: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a caller-supplied ``(rows, cols)`` sparsity pattern.

    The pattern must be a *superset* of the true structure: extra entries
    only cost a reported zero (and possibly an extra color), but a
    missing entry is silently wrong — under ``sparse=True`` it aliases
    into a same-colored reported entry. That contract is the caller's to
    keep; nothing here evaluates the model to check it.

    With ``lower=True`` (the symmetric Hessian) an upper-triangle entry
    ``(i, j)``, ``i < j``, is folded onto its mirror ``(j, i)`` rather
    than rejected — ``H`` is symmetric, so the two say the same thing.
    """
    try:
        rows, cols = pattern
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a (rows, cols) pair of 1-D integer index arrays"
        ) from None
    rows = _as_index_array(rows, f"{name} rows")
    cols = _as_index_array(cols, f"{name} cols")
    if rows.size != cols.size:
        raise ValueError(
            f"{name}: rows and cols must have the same length, "
            f"got {rows.size} and {cols.size}"
        )
    if rows.size:
        for arr, limit, label in ((rows, n_rows, "row"), (cols, n_cols, "col")):
            lo, hi = int(arr.min()), int(arr.max())
            if lo < 0 or hi >= limit:
                raise ValueError(
                    f"{name}: {label} indices must lie in [0, {limit}), "
                    f"got [{lo}, {hi}]"
                )
    if lower:
        rows, cols = np.maximum(rows, cols), np.minimum(rows, cols)
    return _sorted_pattern(rows, cols)


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

    Greedy coloring is sequential in ``j``, but each column's neighbor
    lookup is a single vectorized gather through CSC (rows of a column)
    then CSR (columns of a row) — the per-column Python/dict loops this
    replaced dominated build time on large sparse problems (issue #464).
    """
    n = int(n)
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    colors = np.full(n, -1, dtype=np.int64)
    if rows.size == 0 or n == 0:
        # Empty matrix: still report one (unused) color so seed shapes
        # are well-defined.
        colors.fill(0)
        return colors, 1

    n_rows = int(rows.max()) + 1
    # rows_of_col[j] = csc_rows[csc_ptr[j]:csc_ptr[j+1]]
    csc_ptr, csc_rows = _group_by(cols, rows, n)
    # cols_in_row[i] = csr_cols[csr_ptr[i]:csr_ptr[i+1]]
    csr_ptr, csr_cols = _group_by(rows, cols, n_rows)

    # stamp[c] == j marks color c as forbidden for column j, which lets
    # us reuse one array across all columns instead of rebuilding a set.
    stamp = np.full(n + 1, -1, dtype=np.int64)
    num_colors = 0
    for j in range(n):
        rs = csc_rows[csc_ptr[j]:csc_ptr[j + 1]]
        if rs.size:
            starts = csr_ptr[rs]
            counts = csr_ptr[rs + 1] - starts
            neigh = csr_cols[_ragged_arange(starts, counts)]
            used = colors[neigh]
            stamp[used[used >= 0]] = j
        c = 0
        while stamp[c] == j:
            c += 1
        colors[j] = c
        if c + 1 > num_colors:
            num_colors = c + 1
    return colors, max(num_colors, 1)


def _group_by(
    major: np.ndarray, minor: np.ndarray, n_major: int,
) -> tuple[np.ndarray, np.ndarray]:
    """CSR/CSC-style grouping: ``(indptr, minor_sorted_by_major)``."""
    order = np.argsort(major, kind="stable")
    indptr = np.zeros(int(n_major) + 1, dtype=np.int64)
    np.cumsum(np.bincount(major, minlength=int(n_major)), out=indptr[1:])
    return indptr, minor[order]


def _ragged_arange(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenation of ``range(s, s + c)`` over ``zip(starts, counts)``,
    built without a Python loop."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    return offsets + np.arange(total, dtype=np.int64)
