"""Pure-NumPy tests for the shared autodiff-bridge helpers.

No JAX or PyTorch here — these cover the framework-neutral machinery in
``pounce._ad_common`` that both frontends build on: the blocked sparsity
sweep that replaced dense probing (issue #464), the CPR column coloring,
and the caller-supplied-pattern validation.
"""

from collections import defaultdict

import numpy as np
import pytest

from pounce._ad_common import (
    _color_columns,
    _detect_pattern_2d_multi,
    _detect_pattern_blocked,
    _detect_pattern_lower_multi,
    _normalize_user_pattern,
    _probe_block_size,
)


def _reference_color_columns(rows, cols, n):
    """The pre-#464 dict/set greedy coloring, kept here as the oracle the
    vectorized implementation must reproduce entry for entry."""
    cols_in_row = defaultdict(list)
    rows_of_col = defaultdict(list)
    for r, c in zip(rows.tolist(), cols.tolist()):
        cols_in_row[r].append(c)
        rows_of_col[c].append(r)
    colors = np.full(int(n), -1, dtype=np.int64)
    num_colors = 0
    for j in range(int(n)):
        forbidden = set()
        for r in rows_of_col.get(j, ()):
            for c2 in cols_in_row[r]:
                if c2 != j and colors[c2] >= 0:
                    forbidden.add(int(colors[c2]))
        c = 0
        while c in forbidden:
            c += 1
        colors[j] = c
        num_colors = max(num_colors, c + 1)
    return colors, max(num_colors, 1)


# ----- blocked sparsity sweep (issue #464) -----


@pytest.mark.parametrize("max_bytes", [8, 24, 1 << 20])
def test_detect_pattern_blocked_matches_dense_probe_464(max_bytes):
    """Sweeping the matrix a block of rows (or columns) at a time must
    give exactly the pattern a dense probe would — same indices, same
    row-major order — no matter how small the block budget is."""
    rng = np.random.default_rng(0)
    m, n = 7, 11
    A = np.where(rng.random((m, n)) < 0.4, rng.standard_normal((m, n)), 0.0)
    expected = _detect_pattern_2d_multi([A])

    by_row = _detect_pattern_blocked(
        m, n, lambda a, b: [A[a:b, :]], by_row=True, max_bytes=max_bytes,
    )
    by_col = _detect_pattern_blocked(
        n, m, lambda a, b: [A[:, a:b].T], by_row=False, max_bytes=max_bytes,
    )
    for got in (by_row, by_col):
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])


def test_detect_pattern_blocked_unions_probes_464():
    """A nonzero present in any probe survives, as with the dense union."""
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    B = np.array([[0.0, 0.0], [0.0, 2.0]])
    rows, cols = _detect_pattern_blocked(
        2, 2, lambda a, b: [A[a:b, :], B[a:b, :]], by_row=True, max_bytes=8,
    )
    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_array_equal(cols, [0, 1])


def test_detect_pattern_blocked_lower_triangle_464():
    """``lower=True`` on a symmetric matrix swept by columns reproduces
    the dense lower-triangle probe."""
    rng = np.random.default_rng(3)
    n = 9
    M = np.where(rng.random((n, n)) < 0.5, rng.standard_normal((n, n)), 0.0)
    H = M + M.T  # symmetric, with the structural zeros preserved
    expected = _detect_pattern_lower_multi([H])
    got = _detect_pattern_blocked(
        n, n, lambda a, b: [H[:, a:b].T], by_row=False, lower=True, max_bytes=24,
    )
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


def test_detect_pattern_blocked_empty_matrix_464():
    rows, cols = _detect_pattern_blocked(0, 5, lambda a, b: [], by_row=True)
    assert rows.size == 0 and cols.size == 0
    assert rows.dtype == np.int64 and cols.dtype == np.int64


def test_probe_block_size_respects_budget_464():
    # 1 MiB budget, 1000-long slices -> 131 slices of float64 per block.
    assert _probe_block_size(1000, 1 << 20) == (1 << 20) // 8000
    # Never zero, however wide the slice.
    assert _probe_block_size(10**9, 8) == 1
    assert _probe_block_size(0) == 1


# ----- CPR column coloring -----


def test_color_columns_matches_reference_464():
    """The vectorized greedy coloring must be identical to the dict/set
    implementation it replaced, not merely 'also valid'."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        m = int(rng.integers(1, 25))
        n = int(rng.integers(1, 25))
        mask = rng.random((m, n)) < rng.random()
        rows, cols = (a.astype(np.int64) for a in np.nonzero(mask))
        got = _color_columns(rows, cols, n)
        want = _reference_color_columns(rows, cols, n)
        np.testing.assert_array_equal(got[0], want[0])
        assert got[1] == want[1]


def test_color_columns_empty_pattern_464():
    colors, k = _color_columns(np.zeros(0, np.int64), np.zeros(0, np.int64), 4)
    np.testing.assert_array_equal(colors, np.zeros(4, np.int64))
    assert k == 1


def test_color_columns_banded_is_valid_and_compresses_464():
    n, half = 500, 3
    rows, cols = [], []
    for off in range(-half, half + 1):
        r = np.arange(max(0, -off), min(n, n - off))
        rows.append(r)
        cols.append(r + off)
    rows = np.concatenate(rows).astype(np.int64)
    cols = np.concatenate(cols).astype(np.int64)
    colors, k = _color_columns(rows, cols, n)
    assert k <= 2 * half + 1, f"banded pattern should compress, got {k}"
    cols_in_row = defaultdict(list)
    for r, c in zip(rows.tolist(), cols.tolist()):
        cols_in_row[r].append(c)
    for cs in cols_in_row.values():
        seen = [int(colors[c]) for c in cs]
        assert len(seen) == len(set(seen)), f"columns {cs} share a row and a color"


# ----- caller-supplied patterns (issue #464, request 1) -----


def test_normalize_user_pattern_sorts_and_dedupes_464():
    rows, cols = _normalize_user_pattern(
        ([2, 0, 2, 0], [1, 3, 1, 0]), "jac_pattern", 3, 4,
    )
    np.testing.assert_array_equal(rows, [0, 0, 2])
    np.testing.assert_array_equal(cols, [0, 3, 1])
    assert rows.dtype == np.int64 and cols.dtype == np.int64


def test_normalize_user_pattern_folds_upper_triangle_464():
    """``H`` is symmetric, so an upper-triangle entry names the same
    structural nonzero as its mirror and is folded, not rejected."""
    rows, cols = _normalize_user_pattern(
        ([0, 1, 3], [2, 1, 0]), "hess_pattern", 4, 4, lower=True,
    )
    np.testing.assert_array_equal(rows, [1, 2, 3])
    np.testing.assert_array_equal(cols, [1, 0, 0])


def test_normalize_user_pattern_accepts_empty_464():
    rows, cols = _normalize_user_pattern(([], []), "jac_pattern", 3, 4)
    assert rows.size == 0 and cols.size == 0
    assert rows.dtype == np.int64


@pytest.mark.parametrize(
    "pattern, exc, match",
    [
        ([[0, 1], [0, 1], [0, 1]], ValueError, "pair of 1-D integer index arrays"),
        # A flat (rows, cols) array unpacks into two scalars, not two arrays.
        (np.array([0, 1]), ValueError, "1-D index array, got shape"),
        (([0, 1], [0]), ValueError, "same length"),
        (([[0, 1]], [[0, 1]]), ValueError, "1-D index array"),
        (([0.0, 1.0], [0, 1]), TypeError, "integer indices"),
        (([0, 9], [0, 1]), ValueError, r"row indices must lie in \[0, 3\)"),
        (([0, 1], [0, -1]), ValueError, r"col indices must lie in \[0, 4\)"),
    ],
)
def test_normalize_user_pattern_rejects_bad_input_464(pattern, exc, match):
    with pytest.raises(exc, match=match):
        _normalize_user_pattern(pattern, "jac_pattern", 3, 4)
