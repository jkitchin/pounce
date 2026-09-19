#!/usr/bin/env python3
"""Pack a dumped KKT plus its block labels for the block-factorization prototype.

    pounce model.nl out.sol --dump kkt:10 --dump-dir dump
    python export_kkt_blocks.py dump/iter_010/kkt_solve_001.jsonl kkt.bin

The blocks are found from the matrix alone: the shared (border) columns of an
arrowhead KKT have far higher degree than the rest, so removing everything
above --degree leaves the blocks as connected components. Prints the split so a
wrong threshold is visible (one component means the border was missed).

Binary layout, little-endian: n, nnz, n_blocks (i64) | irn, jcn (i32, **0-based**
lower triangle) | vals (f64) | label per index (i32, -1 = border) | rhs, sol
(f64, the right-hand side pounce solved and the solution it got).
"""

import argparse
import json

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("out")
    ap.add_argument("--degree", type=int, default=100)
    a = ap.parse_args()
    d = json.loads(open(a.jsonl).readline())
    n = d["n"]
    irn = np.array(d["irn"], np.int64) - 1
    jcn = np.array(d["jcn"], np.int64) - 1
    vals = np.array(d["vals"], np.float64)
    A = sp.coo_matrix((np.ones(len(irn)), (irn, jcn)), shape=(n, n))
    A = (A + A.T).tocsr()
    border = np.diff(A.indptr) > a.degree
    keep = ~border
    ncomp, comp = connected_components(A[keep][:, keep], directed=False)
    lab = np.full(n, -1, np.int32)
    lab[keep] = comp.astype(np.int32)
    sizes = np.bincount(comp)
    print(f"n={n} nnz={len(irn)} border={int(border.sum())} blocks={ncomp} "
          f"sizes min {sizes.min()} max {sizes.max()}")
    with open(a.out, "wb") as f:
        np.array([n, len(irn), ncomp], np.int64).tofile(f)
        irn.astype(np.int32).tofile(f)
        jcn.astype(np.int32).tofile(f)
        vals.tofile(f)
        lab.tofile(f)
        np.asarray(d["rhs"], np.float64).tofile(f)
        np.asarray(d["sol"], np.float64).tofile(f)


if __name__ == "__main__":
    main()
