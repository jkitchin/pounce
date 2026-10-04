"""Structural sparsity of a JAX function, read off its jaxpr (gh#985).

Probing a function at a handful of points can only report the entries
that happen to be nonzero *there*: ``exp(-E/(R*T))`` underflows to an
exact 0 at ``T ~ N(0, 1)`` and a polynomial's second derivative vanishes
near 0, so a probe silently drops a structural nonzero for the whole
solve.  This module instead propagates *index sets* through the jaxpr
(a dependency / index-domain analysis).  The answer cannot depend on the
values at which the function is evaluated, so a value-dependent zero can
never remove a structural entry.

How
---
Every jaxpr variable carries, per scalar element, the set of ``x``
indices that element depends on (a boolean ``scipy.sparse`` matrix of
shape ``(size, n)``).  Primitives are handled by structure:

* element-wise ops: union of the operands' sets (broadcasting scalars);
* data movement (``slice``, ``reshape``, ``transpose``, ``concatenate``,
  ``pad``, ``squeeze``, ``broadcast_in_dim``, ``rev``, ``dynamic_slice``,
  ``gather`` with constant indices, ...): rows are moved, not merged;
* reductions / ``dot_general``: union over the contracted axes; a
  constant (closed-over) operand contributes only its *nonzero* pattern,
  so ``A @ x`` with a banded ``A`` stays banded;
* zero-derivative ops (comparisons, ``floor``, ``sign``, integer
  outputs, ``stop_gradient``) cut the dependency;
* ``scatter`` family with constant indices: each operand slot unions the
  updates that can land on it;
* ``jit`` / ``remat`` calls are analysed recursively;
* ``custom_jvp`` functions are analysed through their **JVP rule**, not
  their primal: the derivative AD uses can depend on inputs the primal
  does not (an implicit-function primal computed under ``stop_gradient``,
  a straight-through estimator around ``round``).  The rule's jaxpr is
  run in value semantics (a ``stop_gradient`` inside the rule does not
  cut) with each input tangent carrying its input's dependency.
  ``custom_vjp`` (whose backward rule is opaque Python) and any rule that
  cannot be read are bounded densely: every output element depends on
  the union of every input element's dependency, which is sound because
  a JVP/VJP is linear in the input tangents.

The **Jacobian** pattern is the dependency matrix of ``g``'s outputs.
The **Hessian** pattern is the Jacobian pattern of the *gradient program*
(``make_jaxpr(grad(L))``) — the standard index-domain-propagation
route — symmetrised and folded onto the lower triangle.

Anything this module does not know how to bound (``scan``, ``while``,
``cond``, ``sort``, ``custom_linear_solve``, convolutions, data-dependent
indexing such as a gather / scatter / ``dynamic_slice`` whose indices are
traced from ``x``, ...) makes the
analysis return ``None`` rather than guess, and the caller falls back to
the union-of-probes detector.  The same happens when an intermediate
dependency matrix would exceed ``_MAX_NNZ`` entries (a densely coupled
model), so memory stays bounded.

Cost
----
One Python-level step per jaxpr equation, each a vectorised ``numpy`` /
``scipy.sparse`` operation: ``O(#eqns + total dependency nnz)``, with *no*
AD pass and no evaluation of ``f`` or ``g``.  That is cheaper than the
``O(n)`` blocked AD sweeps of the probe path, and exact in the structural
sense (a superset of every point's true pattern).  It is a superset, not
the minimum: ``x[0] * 0`` keeps its entry, and ``cumsum`` / reductions
bound each output by the whole reduced line.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Dependency matrices larger than this (stored entries) abort the analysis.
_MAX_NNZ = 20_000_000


class _Unsupported(Exception):
    """Raised when a primitive cannot be bounded; the caller falls back."""


# ---------------------------------------------------------------------------
# primitive tables
# ---------------------------------------------------------------------------

_ELEMENTWISE = frozenset(
    """add sub mul div neg exp exp2 log log1p expm1 sin cos tan tanh logistic
    sqrt rsqrt cbrt pow integer_pow abs max min atan2 square erf erfc erf_inv
    lgamma digamma asin acos atan sinh cosh asinh acosh atanh
    convert_element_type add_any copy copy_p real imag conj rem nextafter
    reduce_precision clamp igamma igammac igamma_grad_a random_gamma_grad
    bessel_i0e bessel_i1e polygamma zeta regularized_incomplete_beta
    mul_add""".split()
)

# Derivative is zero a.e. / output not differentiable.
_ZERO_DERIVATIVE = frozenset(
    """lt le gt ge eq ne sign floor ceil round is_finite not and or xor
    argmax argmin iota stop_gradient population_count shift_left
    shift_right_logical shift_right_arithmetic clz random_bits random_seed
    random_wrap random_unwrap threefry2x32 eq_to""".split()
)

_REDUCTIONS = frozenset(
    "reduce_sum reduce_max reduce_min reduce_prod reduce_and reduce_or "
    "reduce_xor".split()
)

_CUMULATIVE = frozenset("cumsum cumprod cummax cummin cumlogsumexp".split())

_CALLS = frozenset(
    """pjit jit closed_call core_call remat checkpoint remat2 custom_jvp_call
    custom_vjp_call custom_vjp_call_jaxpr xla_call named_call""".split()
)

# User-defined derivatives: the derivative AD uses is NOT the derivative
# of the primal ``call_jaxpr`` (an implicit-function primal computed under
# ``stop_gradient``; a straight-through estimator around ``round``), so
# reading the primal's dependency would silently drop entries (gh#985
# review).  These are analysed through their JVP rule, or bounded densely.
_CUSTOM_JVP = frozenset("custom_jvp_call".split())
_CUSTOM_VJP = frozenset("custom_vjp_call custom_vjp_call_jaxpr custom_lin".split())

# Zero-derivative ops whose *value* still carries its input.  Inside a JVP
# rule the rule is evaluated, not differentiated, so ``stop_gradient(t)``
# is ``t`` there and these must propagate instead of cutting.
_VALUE_PASS = frozenset("stop_gradient floor ceil round sign".split())


def _sub_jaxpr(params):
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        j = params.get(key)
        if j is not None and (hasattr(j, "jaxpr") or hasattr(j, "eqns")):
            return j
    return None


def _is_inexact(aval) -> bool:
    dt = getattr(aval, "dtype", None)
    if dt is None:
        return False
    return np.issubdtype(dt, np.inexact)


# ---------------------------------------------------------------------------
# sparse helpers.  A "dep" is None (no dependency at all) or a csr float32
# matrix (size, n) whose stored entries are all 1.
# ---------------------------------------------------------------------------


def _binarize(M):
    """Set every stored entry to 1.  Inputs are products / sums of binary
    csr matrices, which scipy returns canonical (no duplicates)."""
    if not sp.isspmatrix_csr(M):
        M = M.tocsr()
    if not M.has_canonical_format:
        M.sum_duplicates()
    M.data = np.ones(M.data.shape, dtype=np.float32)
    if M.nnz > _MAX_NNZ:
        raise _Unsupported("dependency matrix too large")
    return M


def _selector(rows, cols, shape):
    """0/1 selector with a 1 at each ``(rows[k], cols[k])``."""
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
    )


def _take(D, ids, size):
    """Rows ``ids`` of ``D`` (``size`` rows); ``-1`` selects an empty row."""
    if D is None:
        return None
    ids = np.asarray(ids).reshape(-1)
    if ids.size == 0:
        return sp.csr_matrix((0, D.shape[1]), dtype=np.float32)
    lo, hi = int(ids.min()), int(ids.max())
    # Bound the result's nnz from the row lengths before materialising it.
    rn = np.diff(D.indptr)
    if int(rn[np.maximum(ids, 0)].sum(dtype=np.int64)) > _MAX_NNZ:
        raise _Unsupported("dependency matrix too large")
    if lo >= 0:
        if hi - lo + 1 == ids.size and (ids.size == 1 or (np.diff(ids) == 1).all()):
            R = D[lo : hi + 1]  # contiguous block
        else:
            R = D[ids]
        if R.nnz > _MAX_NNZ:
            raise _Unsupported("dependency matrix too large")
        return R
    valid = ids >= 0
    R = D[np.where(valid, ids, 0)]
    R = sp.diags(valid.astype(np.float32), format="csr") @ R
    return _binarize(R)


def _union(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return _binarize(out)


def _size(shape) -> int:
    return int(np.prod(shape, dtype=np.int64)) if len(shape) else 1


def _group_union(D, groups, size):
    """``out[r] = union_k D[groups[r, k]]`` for a ``(R, K)`` index array."""
    if D is None:
        return None
    R, K = groups.shape
    P = sp.csr_matrix(
        (
            np.ones(R * K, dtype=np.float32),
            groups.reshape(-1),
            np.arange(0, R * K + 1, K),
        ),
        shape=(R, size),
    )
    return _binarize(P @ D)


# ---------------------------------------------------------------------------
# the interpreter
# ---------------------------------------------------------------------------


class _Interp:
    def __init__(self, n):
        self.n = n
        # False: dependencies mean "first derivative AD would produce"
        # (``stop_gradient`` cuts, custom rules replace the primal).  True:
        # plain value flow, used while evaluating a custom JVP rule.
        self.value_mode = False

    # -- environment ------------------------------------------------------

    def run(self, closed, in_deps, in_known):
        """Evaluate a (Closed)Jaxpr abstractly; returns (out_deps, out_known)."""
        jaxpr = getattr(closed, "jaxpr", closed)
        consts = list(getattr(closed, "consts", ()))
        dep, known = {}, {}
        for v, c in zip(jaxpr.constvars, consts):
            dep[v] = None
            known[v] = np.asarray(c)
        for v, d, k in zip(jaxpr.invars, in_deps, in_known):
            dep[v] = d
            if k is not None:
                known[v] = k

        def get(a):
            if type(a).__name__ == "Literal":
                return None
            return dep.get(a)

        def getk(a):
            if type(a).__name__ == "Literal":
                return np.asarray(a.val)
            return known.get(a)

        for eqn in jaxpr.eqns:
            ins = [get(a) for a in eqn.invars]
            ks = [getk(a) for a in eqn.invars]
            outs, oknown = self.eqn(eqn, ins, ks)
            for v, d in zip(eqn.outvars, outs):
                dep[v] = d
            if oknown is not None:
                for v, k in zip(eqn.outvars, oknown):
                    if k is not None:
                        known[v] = k
        return (
            [get(v) for v in jaxpr.outvars],
            [getk(v) for v in jaxpr.outvars],
        )

    # -- one equation ------------------------------------------------------

    def eqn(self, eqn, ins, ks):
        prim = eqn.primitive
        name = prim.name
        outvars = eqn.outvars
        nout = len(outvars)
        avals = [v.aval for v in outvars]

        # Index arithmetic on constants: evaluate eagerly (cheap, ints only)
        # so gather / dynamic_slice indices are known.
        oknown = None
        if (
            all(d is None for d in ins)
            and all(k is not None for k in ks)
            and (
                all(not _is_inexact(a) for a in avals)
                or all(_size(tuple(a.shape)) > 1 for a in avals)
            )
            and all(_size(tuple(a.shape)) <= 4_000_000 for a in avals)
            and name not in _CALLS
        ):
            try:
                res = prim.bind(*ks, **eqn.params)
                res = res if prim.multiple_results else [res]
                oknown = [np.asarray(r) for r in res]
            except Exception:
                oknown = None

        if all(d is None for d in ins):
            # No dependence on x at all: nothing downstream to track.
            return [None] * nout, oknown
        if all(not _is_inexact(a) for a in avals):
            return [None] * nout, oknown
        if name in _ZERO_DERIVATIVE:
            if self.value_mode and name in _VALUE_PASS:
                return [self.elementwise(eqn, ins, tuple(avals[0].shape))], None
            return [None] * nout, oknown

        if name in _CUSTOM_JVP or name in _CUSTOM_VJP:
            if not self.value_mode:
                if name in _CUSTOM_JVP:
                    return self.custom_jvp(eqn, ins, ks), None
                return self.dense_union(eqn, ins), None
            if name == "custom_lin":
                return self.dense_union(eqn, ins), None
            # value mode: the primal is what gets evaluated -- fall through.

        if name in _CALLS:
            sub = _sub_jaxpr(eqn.params)
            if sub is None:
                raise _Unsupported(name)
            # custom_jvp_call / custom_vjp_call carry leading consts in
            # ``invars`` that line up with the callee's invars one-to-one.
            n_in = len(getattr(sub, "jaxpr", sub).invars)
            if n_in != len(ins):
                raise _Unsupported(name)
            od, ok = self.run(sub, ins, ks)
            return od, ok

        out_shape = tuple(avals[0].shape) if nout else ()

        if name in _ELEMENTWISE:
            return [self.elementwise(eqn, ins, out_shape)], None
        if name == "select_n":
            # operand 0 is the predicate; it cuts (derivative is zero a.e.).
            return [self.elementwise(eqn, ins[1:], out_shape, skip=1)], None
        if name in _REDUCTIONS:
            return [self.reduce(eqn, ins[0])], None
        if name in _CUMULATIVE:
            return [self.cumulative(eqn, ins[0])], None
        if name == "dot_general":
            return [self.dot_general(eqn, ins, ks)], None
        return self.movement(eqn, ins, ks)

    # -- custom derivatives ---------------------------------------------------

    def dense_union(self, eqn, ins):
        """Sound bound for an opaque derivative: a JVP / VJP is linear in
        the input tangents, so every inexact output element depends at
        most on the union of *all* input elements' dependencies."""
        live = [D for D in ins if D is not None]
        if not live:
            return [None] * len(eqn.outvars)
        cols = np.unique(np.concatenate([D.indices for D in live]))
        U = _selector(np.zeros(cols.size, dtype=np.int64), cols, (1, self.n))
        outs = []
        for v in eqn.outvars:
            if not _is_inexact(v.aval):
                outs.append(None)
                continue
            sz = _size(tuple(v.aval.shape))
            if sz * cols.size > _MAX_NNZ:
                raise _Unsupported("dependency matrix too large")
            outs.append(_take(U, np.zeros(sz, dtype=np.int64), 1))
        return outs

    def custom_jvp(self, eqn, ins, ks):
        """``custom_jvp_call``: the dependency AD sees is the tangent flow
        of the user's JVP rule, not the primal.  Evaluate the rule's jaxpr
        with the primals held constant and each input tangent carrying its
        input's dependency (value semantics: the rule is *run*, so a
        ``stop_gradient`` inside it does not cut).  Anything unexpected
        degrades to :meth:`dense_union`, never to the primal."""
        p = eqn.params
        nc = int(p.get("num_consts", 0) or 0)
        prim_ins, prim_ks = ins[nc:], ks[nc:]
        if any(D is not None for D in ins[:nc]):
            return self.dense_union(eqn, ins)
        fun = p.get("jvp_jaxpr_fun", p.get("jvp_jaxpr_thunk"))
        if fun is None:
            return self.dense_union(eqn, ins)
        zeros = [False] * len(prim_ins)
        try:
            call = getattr(fun, "call_wrapped", fun)
            jvp_jaxpr, consts, out_zeros = call(*zeros)
        except Exception:
            return self.dense_union(eqn, ins)
        jaxpr = getattr(jvp_jaxpr, "jaxpr", jvp_jaxpr)
        consts = list(getattr(jvp_jaxpr, "consts", ())) + list(consts or ())
        out_zeros = list(out_zeros)
        if (
            len(jaxpr.invars) != 2 * len(prim_ins)
            or len(consts) != len(jaxpr.constvars)
            or len(out_zeros) != len(eqn.outvars)
        ):
            return self.dense_union(eqn, ins)

        class _Closed:
            pass

        closed = _Closed()
        closed.jaxpr, closed.consts = jaxpr, consts
        in_deps = [None] * len(prim_ins) + list(prim_ins)
        in_known = list(prim_ks) + [None] * len(prim_ins)
        saved = self.value_mode
        self.value_mode = True
        try:
            od, _ = self.run(closed, in_deps, in_known)
        except _Unsupported:
            return self.dense_union(eqn, ins)
        finally:
            self.value_mode = saved
        tangents = iter(od[len(out_zeros):])
        outs = [None if z else next(tangents, None) for z in out_zeros]
        for v, D in zip(eqn.outvars, outs):
            if D is not None and D.shape[0] != _size(tuple(v.aval.shape)):
                return self.dense_union(eqn, ins)
        return outs

    # -- element-wise -------------------------------------------------------

    def elementwise(self, eqn, ins, out_shape, skip=0):
        N = _size(out_shape)
        parts = []
        for a, D in zip(eqn.invars[skip:], ins):
            if D is None:
                continue
            sh = tuple(a.aval.shape)
            if sh == out_shape:
                parts.append(D)
                continue
            if len(sh) not in (0, len(out_shape)):
                raise _Unsupported("implicit broadcast")
            # Broadcasting repeats every source row N / size(sh) times; check
            # the result's nnz before building the N-long index array.
            if D.nnz * (N // max(_size(sh), 1)) > _MAX_NNZ:
                raise _Unsupported("dependency matrix too large")
            try:
                ids = np.broadcast_to(
                    np.arange(_size(sh), dtype=np.int64).reshape(sh), out_shape
                )
            except ValueError:
                raise _Unsupported("implicit broadcast")
            parts.append(_take(D, ids, _size(sh)))
        return _union(parts)

    # -- reductions ---------------------------------------------------------

    def _lines(self, shape, axes):
        """``(R, K)`` index array: rows are the kept coordinates, columns
        enumerate the reduced axes."""
        ids = np.arange(_size(shape), dtype=np.int64).reshape(shape)
        keep = [i for i in range(len(shape)) if i not in axes]
        ids = np.transpose(ids, keep + sorted(axes))
        K = _size([shape[i] for i in axes])
        return ids.reshape(-1, K) if ids.size else ids.reshape(0, max(K, 1))

    def reduce(self, eqn, D):
        shape = tuple(eqn.invars[0].aval.shape)
        axes = tuple(int(a) % max(len(shape), 1) for a in eqn.params["axes"])
        if not axes:
            return D
        return _group_union(D, self._lines(shape, axes), _size(shape))

    def cumulative(self, eqn, D):
        # Conservative: each output depends on its whole line.
        shape = tuple(eqn.invars[0].aval.shape)
        axis = int(eqn.params["axis"]) % max(len(shape), 1)
        lines = self._lines(shape, (axis,))
        N = _size(shape)
        R = _group_union(D, lines, N)
        # Scatter the per-line unions back to every element of the line.
        owner = np.empty(N, dtype=np.int64)
        owner[lines.reshape(-1)] = np.repeat(np.arange(lines.shape[0]), lines.shape[1])
        return _take(R, owner, lines.shape[0])

    # -- dot_general --------------------------------------------------------

    def dot_general(self, eqn, ins, ks):
        (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
        lsh = tuple(eqn.invars[0].aval.shape)
        rsh = tuple(eqn.invars[1].aval.shape)
        lfree = [i for i in range(len(lsh)) if i not in lc and i not in lb]
        rfree = [i for i in range(len(rsh)) if i not in rc and i not in rb]
        B = _size([lsh[i] for i in lb])
        M = _size([lsh[i] for i in lfree])
        N = _size([rsh[i] for i in rfree])
        K = _size([lsh[i] for i in lc])
        if B * M * N > _MAX_NNZ:
            # Check before building the (B*M*N)-long index arrays below.
            raise _Unsupported("dot_general output too large")

        def arrange(shape, b, free, c, X):
            ids = np.arange(_size(shape), dtype=np.int64).reshape(shape)
            return np.transpose(ids, list(b) + list(free) + list(c)).reshape(B, X, K)

        lid = arrange(lsh, lb, lfree, lc, M)  # (B, M, K)
        rid = arrange(rsh, rb, rfree, rc, N)  # (B, N, K)
        Dl, Dr = ins
        kl, kr = ks
        both = Dl is not None and Dr is not None
        parts = []
        if Dl is not None:
            mask = None if both or kr is None else (np.asarray(kr), rid)
            parts.append(self._dot_side(Dl, lid, _size(lsh), B, M, N, K, mask, True))
        if Dr is not None:
            mask = None if both or kl is None else (np.asarray(kl), lid)
            parts.append(self._dot_side(Dr, rid, _size(rsh), B, N, M, K, mask, False))
        return _union(parts)

    @staticmethod
    def _dot_side(D, ids, size, B, X, Y, K, mask, x_is_m):
        """Dependencies reaching ``out[b, m, n]`` (flattened row-major over
        ``(B, M, N)``) through the x-dependent operand ``D``, whose index
        array ``ids`` has shape ``(B, X, K)``; ``Y`` is the partner's free
        size.  With a constant partner (``mask = (values, partner_ids)``)
        only the contracted positions where that constant is nonzero count.
        The ``B*M*N``-long index arrays are built only once the result is
        known to fit the budget."""
        E = B * X * Y
        if mask is not None and E * K <= 20_000_000:
            vals, pids = mask
            g = np.indices((B, X, Y) if x_is_m else (B, Y, X)).reshape(3, -1)
            bi, xi, yi = (g[0], g[1], g[2]) if x_is_m else (g[0], g[2], g[1])
            nz = vals.reshape(-1)[pids.reshape(-1)].reshape(pids.shape) != 0
            kmask = nz[bi, yi]  # (E, K)
            drows = ids[bi, xi]  # (E, K)
            r, k = np.nonzero(kmask)
            P = _selector(r, drows[r, k], (E, size))
            return _binarize(P @ D)
        R = _group_union(D, ids.reshape(B * X, K), size)
        # every R row is repeated Y times in the output: check first
        if R.nnz * Y > _MAX_NNZ:
            raise _Unsupported("dependency matrix too large")
        if x_is_m:  # out (b, m, n) <- R[b*M + m]
            rows = np.repeat(np.arange(B * X, dtype=np.int64), Y)
        else:  # out (b, m, n) <- R[b*N + n]
            rows = np.broadcast_to(
                (np.arange(B, dtype=np.int64)[:, None, None] * X
                 + np.arange(X, dtype=np.int64)[None, None, :]),
                (B, Y, X),
            ).reshape(-1)
        return _take(R, rows, B * X)

    # -- data movement ------------------------------------------------------

    def movement(self, eqn, ins, ks):
        """Pure row-moving primitives.  Anything else is unsupported."""
        prim, name, p = eqn.primitive, eqn.primitive.name, eqn.params
        invars = eqn.invars
        sizes = [_size(tuple(a.aval.shape)) for a in invars]
        outs = []

        def ids_of(i):
            return np.arange(sizes[i], dtype=np.int64).reshape(invars[i].aval.shape)

        def finish(id_arrays, pool_sizes):
            # pool = rows of ins concatenated; map ids -> (operand, row)
            res = []
            for ida in id_arrays:
                res.append(self._pool_take(ins, sizes, ida))
            return res

        if name in ("reshape", "squeeze", "expand_dims", "convert_element_type"):
            ida = ids_of(0)
            dims = p.get("dimensions") if name == "reshape" else None
            if dims is not None:
                # lax.reshape(x, shape, dimensions=perm) transposes first
                # (JAX emits it for ravel(order="F") and in the gradient
                # of prod(..., axis=k)).
                ida = np.transpose(ida, tuple(int(d) for d in dims))
            ida = ida.reshape(eqn.outvars[0].aval.shape)
            return [_take(ins[0], ida, sizes[0])], None
        if name == "broadcast_in_dim":
            shape = tuple(p["shape"])
            bd = tuple(p["broadcast_dimensions"])
            src = ids_of(0)
            inter = [1] * len(shape)
            for d, s in zip(bd, src.shape):
                inter[d] = s
            ida = np.broadcast_to(src.reshape(inter), shape)
            return [_take(ins[0], ida, sizes[0])], None
        if name == "transpose":
            return [_take(ins[0], np.transpose(ids_of(0), p["permutation"]), sizes[0])], None
        if name == "rev":
            return [_take(ins[0], np.flip(ids_of(0), axis=tuple(p["dimensions"])), sizes[0])], None
        if name == "slice":
            sl = tuple(
                slice(int(a), int(b), None if s is None else int(s))
                for a, b, s in zip(
                    p["start_indices"],
                    p["limit_indices"],
                    p["strides"] or [None] * len(p["start_indices"]),
                )
            )
            return [_take(ins[0], ids_of(0)[sl], sizes[0])], None
        if name == "unstack":
            axis = int(p["axis"])
            ida = ids_of(0)
            return [
                _take(ins[0], np.take(ida, i, axis=axis), sizes[0])
                for i in range(ida.shape[axis])
            ], None
        if name == "split":
            axis = int(p["axis"])
            parts = np.split(ids_of(0), np.cumsum(p["sizes"])[:-1], axis=axis)
            return [_take(ins[0], a, sizes[0]) for a in parts], None
        if name == "concatenate":
            offs = np.concatenate([[0], np.cumsum(sizes)])
            arrs = [ids_of(i) + offs[i] for i in range(len(invars))]
            ida = np.concatenate(arrs, axis=int(p["dimension"]))
            return [self._pool_take(ins, sizes, ida)], None
        if name == "tile":
            return [_take(ins[0], np.tile(ids_of(0), tuple(p["reps"])), sizes[0])], None
        if name == "stack":
            offs = np.concatenate([[0], np.cumsum(sizes)])
            arrs = [ids_of(i) + offs[i] for i in range(len(invars))]
            ida = np.stack(arrs, axis=int(p["axis"]))
            return [self._pool_take(ins, sizes, ida)], None
        if name == "pad":
            if ins[1] is not None:
                raise _Unsupported("pad with x-dependent padding value")
            ida = ids_of(0)
            cfg = [tuple(int(v) for v in c) for c in p["padding_config"]]
            if any(lo < 0 or hi < 0 for lo, hi, _ in cfg):
                # negative padding crops: do it with a cropping slice first
                crop = tuple(
                    slice(max(-lo, 0), ida.shape[d] - max(-hi, 0))
                    for d, (lo, hi, _) in enumerate(cfg)
                )
                ida = ida[crop]
                cfg = [(max(lo, 0), max(hi, 0), it) for lo, hi, it in cfg]
            for d, (lo, hi, it) in enumerate(cfg):
                if it:
                    shp = list(ida.shape)
                    L = shp[d]
                    newL = L + (L - 1) * it if L else 0
                    shp[d] = newL
                    big = np.full(shp, -1, dtype=np.int64)
                    idx = [slice(None)] * ida.ndim
                    idx[d] = slice(0, newL, it + 1)
                    big[tuple(idx)] = ida
                    ida = big
                width = [(0, 0)] * ida.ndim
                width[d] = (lo, hi)
                ida = np.pad(ida, width, constant_values=-1)
            return [_take(ins[0], ida, sizes[0])], None
        if name == "dynamic_slice":
            if any(k is None for k in ks[1:]):
                raise _Unsupported("dynamic_slice with unknown index")
            ida = ids_of(0)
            starts = [int(np.asarray(k)) for k in ks[1:]]
            sl = []
            for s, ln, dim in zip(starts, p["slice_sizes"], ida.shape):
                s = min(max(s, 0), dim - int(ln))  # XLA clamps
                sl.append(slice(s, s + int(ln)))
            return [_take(ins[0], ida[tuple(sl)], sizes[0])], None
        if name == "dynamic_update_slice":
            if any(k is None for k in ks[2:]):
                raise _Unsupported("dynamic_update_slice with unknown index")
            base = ids_of(0)
            upd = ids_of(1) + sizes[0]
            starts = [int(np.asarray(k)) for k in ks[2:]]
            sl = []
            for s, ln, dim in zip(starts, upd.shape, base.shape):
                s = min(max(s, 0), dim - ln)
                sl.append(slice(s, s + ln))
            out = base.copy()
            out[tuple(sl)] = upd
            return [self._pool_take(ins, sizes, out)], None
        if name == "gather":
            if ks[1] is None:
                raise _Unsupported("gather with unknown indices")
            # Run the gather itself on row ids; -1 marks out-of-bounds fill.
            import jax.numpy as jnp

            ida = jnp.asarray(ids_of(0).astype(np.int32))
            q = dict(p)
            q["fill_value"] = -1
            res = np.asarray(prim.bind(ida, jnp.asarray(ks[1]), **q))
            return [_take(ins[0], res.astype(np.int64), sizes[0])], None
        if name.startswith("scatter"):
            return [self._scatter(eqn, ins, ks, sizes)], None
        raise _Unsupported(name)

    def _scatter(self, eqn, ins, ks, sizes):
        """``scatter`` / ``scatter-add`` / ...: ``out[j]`` depends on
        ``operand[j]`` and on every update element that can land on ``j``.
        The landing map is computed by running the *matching gather* over
        row ids (the same construction JAX's own scatter-add transpose
        rule uses).  A collision (several updates, one slot) or an
        overwrite is covered by taking the union, so this is a superset."""
        import jax
        import jax.numpy as jnp

        if ks[1] is None:
            raise _Unsupported("scatter with unknown indices")
        p = eqn.params
        dn = p["dimension_numbers"]
        op_aval, upd_aval = eqn.invars[0].aval, eqn.invars[2].aval
        op_shape, upd_shape = tuple(op_aval.shape), tuple(upd_aval.shape)
        inserted = set(dn.inserted_window_dims) | set(
            getattr(dn, "operand_batching_dims", ())
        )
        win = iter(dn.update_window_dims)
        slice_sizes = tuple(
            1 if d in inserted else upd_shape[next(win)] for d in range(len(op_shape))
        )
        gdn = jax.lax.GatherDimensionNumbers(
            offset_dims=tuple(dn.update_window_dims),
            collapsed_slice_dims=tuple(dn.inserted_window_dims),
            start_index_map=tuple(dn.scatter_dims_to_operand_dims),
            operand_batching_dims=tuple(getattr(dn, "operand_batching_dims", ())),
            start_indices_batching_dims=tuple(
                getattr(dn, "scatter_indices_batching_dims", ())
            ),
        )
        ids = jnp.asarray(np.arange(sizes[0], dtype=np.int32).reshape(op_shape))
        tgt = np.asarray(
            jax.lax.gather(
                ids,
                jnp.asarray(ks[1]),
                gdn,
                slice_sizes,
                mode=p["mode"],
                fill_value=-1,
            )
        ).reshape(-1)
        if tgt.size != sizes[2]:
            raise _Unsupported("scatter shape")
        Dop, _, Dupd = ins
        parts = [Dop]
        if Dupd is not None:
            valid = np.nonzero(tgt >= 0)[0]
            P = _selector(tgt[valid], valid, (sizes[0], sizes[2]))
            parts.append(_binarize(P @ Dupd))
        return _union(parts)

    def _pool_take(self, ins, sizes, ida):
        """Rows of the concatenation of all operands' dependency matrices
        selected by ``ida`` (``-1`` = empty)."""
        live = [i for i, D in enumerate(ins) if D is not None]
        if not live:
            return None
        offs = np.concatenate([[0], np.cumsum(sizes)])
        pool = sp.vstack(
            [
                ins[i]
                if ins[i] is not None
                else sp.csr_matrix((sizes[i], self.n), dtype=np.float32)
                for i in range(len(ins))
            ],
            format="csr",
        )
        return _take(pool, ida, int(offs[-1]))


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def _identity_deps(n):
    return sp.identity(n, dtype=np.float32, format="csr")


def _out_matrix(deps, outvars, n):
    """Stack output dependency matrices into a single ``(total, n)`` one."""
    mats = []
    for d, v in zip(deps, outvars):
        sz = _size(tuple(v.aval.shape))
        mats.append(d if d is not None else sp.csr_matrix((sz, n), dtype=np.float32))
    return sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]


def _analyse(fn, example_args, x_pos, n):
    import jax

    closed = jax.make_jaxpr(fn)(*example_args)
    interp = _Interp(n)
    in_deps, in_known = [], []
    for i, a in enumerate(closed.jaxpr.invars):
        in_deps.append(_identity_deps(n) if i == x_pos else None)
        in_known.append(None)
    outs, _ = interp.run(closed, in_deps, in_known)
    return _out_matrix(outs, closed.jaxpr.outvars, n)


def jacobian_pattern(g, n, m, extra_args=()):
    """``(rows, cols)`` of the structural ``(m, n)`` Jacobian of ``g``, or
    ``None`` when the analysis cannot bound ``g``."""
    import jax.numpy as jnp

    try:
        D = _analyse(g, (jnp.zeros(n),) + tuple(extra_args), 0, n).tocoo()
    except _Unsupported:
        return None
    except Exception:
        return None
    if D.shape[0] != m:
        return None
    order = np.lexsort((D.col, D.row))
    return D.row[order].astype(np.int64), D.col[order].astype(np.int64)


def hessian_lower_pattern(grad_fn, example_args, n):
    """Lower-triangle ``(rows, cols)`` of the structural Hessian, read from
    the Jacobian pattern of ``grad_fn`` (a gradient program whose first
    argument is ``x``).  ``None`` when the analysis cannot bound it."""
    try:
        D = _analyse(grad_fn, example_args, 0, n)
    except Exception:
        return None
    if D.shape != (n, n):
        return None
    S = _binarize(D + D.T).tocoo()
    keep = S.row >= S.col
    r, c = S.row[keep].astype(np.int64), S.col[keep].astype(np.int64)
    order = np.lexsort((c, r))
    return r[order], c[order]
