"""Tests for in-memory `.nl` model construction (issue #469).

Three surfaces, all of which used to be Rust-only:

* ``pounce.parse_nl_text`` — the same parser ``read_nl`` uses, fed a string
  instead of a path, so a frontend that generates `.nl` never touches disk.
* ``pounce.NlExpr`` / ``pounce.build_nl_problem`` — build the expression DAG
  directly and skip `.nl` entirely. This is the only route for operators
  `.nl` cannot carry (``atan2``, ``min``/``max``, ``erf``).
* ``NlProblem.hessian_vector_product`` — matrix-free ``∇²L · v``, for models
  where materializing the Lagrangian Hessian is impractical.
"""

import math
import subprocess
import sys
import time

import numpy as np
import pytest
import scipy
import scipy.sparse as sp

import pounce

# scipy shaped 1-D sparse *arrays* inconsistently before 1.14
# (`csr_array(v_1d)` was (1, n) on 1.11 and raised on 1.13), and the
# package floor is scipy>=1.11 — so the 1-D-array cases below are gated
# rather than left to whatever CI happens to install.
_SCIPY = tuple(int(p) for p in scipy.__version__.split(".")[:2])

# A complete `.nl` body: min (x0-1)^2 + (x1-2)^2, no constraints. Same
# fixture shape the Rust reader's unit tests use, kept here as text so the
# parse-from-string path has something to chew on that never hits disk.
SIMPLE_NL = """g3 0 1 0
2 0 1 0 0
0 1
0 0
0 2 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 0 0 0 0
O0 0
o0
o5
o1
v0
n1
n2
o5
o1
v1
n2
n2
b
3
3
"""


def test_parse_nl_text_matches_read_nl_semantics(tmp_path):
    """Parsing text and parsing the same bytes off disk agree exactly."""
    p_text = pounce.parse_nl_text(SIMPLE_NL)
    nl_file = tmp_path / "simple.nl"
    nl_file.write_text(SIMPLE_NL)
    p_file = pounce.read_nl(str(nl_file))

    assert (p_text.n, p_text.m) == (p_file.n, p_file.m) == (2, 0)
    x = np.array([0.3, 1.4])
    assert p_text.objective(x) == p_file.objective(x)
    np.testing.assert_array_equal(p_text.gradient(x), p_file.gradient(x))

    # (x0-1)^2 + (x1-2)^2 at (0.3, 1.4): 0.49 + 0.36
    assert p_text.objective(x) == pytest.approx(0.85)
    np.testing.assert_allclose(p_text.gradient(x), [-1.4, -1.2])


def test_parse_nl_text_accepts_names_and_validates_length():
    """There are no sibling `.col`/`.row` files, so names come as arguments."""
    p = pounce.parse_nl_text(SIMPLE_NL, var_names=["alpha", "beta"])
    assert p.var_names == ["alpha", "beta"]
    assert p.con_names == []

    with pytest.raises(ValueError, match="var_names"):
        pounce.parse_nl_text(SIMPLE_NL, var_names=["only_one"])


def test_parse_nl_text_reports_bad_input_as_valueerror():
    with pytest.raises(ValueError, match="parse_nl_text"):
        pounce.parse_nl_text("this is not an .nl file")


# ---- NlExpr / build_nl_problem ----------------------------------------


def test_build_nl_problem_rosenbrock_matches_analytic():
    x = pounce.NlExpr.vars(2)
    rosen = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    p = pounce.build_nl_problem(n=2, objective=rosen, x0=[-1.2, 1.0])

    assert (p.n, p.m) == (2, 0)
    assert p.minimize is True

    x0, x1 = -1.2, 1.0
    assert p.objective([x0, x1]) == pytest.approx(
        (1 - x0) ** 2 + 100 * (x1 - x0**2) ** 2
    )
    np.testing.assert_allclose(
        p.gradient([x0, x1]),
        [
            -2 * (1 - x0) - 400 * x0 * (x1 - x0**2),
            200 * (x1 - x0**2),
        ],
        rtol=1e-10,
    )


def test_build_nl_problem_constraints_and_structures():
    x = pounce.NlExpr.vars(3)
    p = pounce.build_nl_problem(
        n=3,
        objective=pounce.NlExpr.sum([x[0] ** 2, x[1] ** 2, x[2] ** 2]),
        constraints=[x[0] * x[1], x[1] + x[2]],
        g_l=[1.0, 0.0],
        g_u=[1.0, 5.0],
        x_l=[-10.0] * 3,
        x_u=[10.0] * 3,
        var_names=["a", "b", "c"],
        con_names=["prod", "lin"],
    )

    assert (p.n, p.m) == (3, 2)
    assert p.var_names == ["a", "b", "c"]
    assert p.con_names == ["prod", "lin"]
    np.testing.assert_allclose(p.g_l, [1.0, 0.0])
    np.testing.assert_allclose(p.g_u, [1.0, 5.0])

    pt = [2.0, 3.0, -1.0]
    np.testing.assert_allclose(p.constraints(pt), [6.0, 2.0])

    jr, jc = p.jacobian_structure()
    jv = p.jacobian(pt)
    assert jr.shape == jc.shape == jv.shape == (p.nnz_jac,)
    dense = np.zeros((p.m, p.n))
    dense[jr, jc] = jv
    np.testing.assert_allclose(dense, [[3.0, 2.0, 0.0], [0.0, 1.0, 1.0]])

    hr, hc = p.hessian_structure()
    assert np.all(hr >= hc), "Hessian structure must be the lower triangle"
    assert p.hessian(pt).shape == (p.nnz_hess,)


def test_build_nl_problem_defaults_are_unbounded_and_zero_start():
    p = pounce.build_nl_problem(n=2, objective=pounce.NlExpr.var(0))
    np.testing.assert_allclose(p.x0, [0.0, 0.0])
    assert np.all(np.asarray(p.x_l) <= -1e19)
    assert np.all(np.asarray(p.x_u) >= 1e19)


def test_build_nl_problem_maximize_negates_objective():
    x = pounce.NlExpr.var(0)
    p = pounce.build_nl_problem(n=1, objective=x**2, minimize=False)
    assert p.minimize is False
    # The evaluator hands back the minimization form.
    assert p.objective([3.0]) == pytest.approx(-9.0)
    np.testing.assert_allclose(p.gradient([3.0]), [-6.0])


def test_build_nl_problem_rejects_out_of_range_variable():
    with pytest.raises(ValueError, match=r"Var\(4\)"):
        pounce.build_nl_problem(n=2, objective=pounce.NlExpr.var(4))

    with pytest.raises(ValueError, match="constraint 0"):
        pounce.build_nl_problem(
            n=2, objective=pounce.NlExpr.var(0), constraints=[pounce.NlExpr.var(9)]
        )


def test_build_nl_problem_rejects_mismatched_vector_lengths():
    with pytest.raises(ValueError, match="x_l"):
        pounce.build_nl_problem(n=3, objective=pounce.NlExpr.var(0), x_l=[0.0, 1.0])


def test_build_nl_problem_solves_through_minimize():
    """End to end: an in-memory model reaches the solver, not just the
    evaluators."""
    x = pounce.NlExpr.vars(2)
    p = pounce.build_nl_problem(
        n=2,
        objective=(x[0] - 1) ** 2 + (x[1] + 2) ** 2,
        x0=[5.0, 5.0],
    )
    (x, info), = pounce.solve_nlp_batch([p])
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [1.0, -2.0], atol=1e-6)


# ---- Operators `.nl` cannot carry -------------------------------------


def test_expressions_nl_cannot_export():
    """``atan2``, ``min``/``max`` and ``erf`` all evaluate here.

    A `.nl` round trip loses every one of them: writers have no two-argument
    funcall path for ``atan2``, ``min``/``max`` force a DNLP model type, and
    AMPL has no ``erf`` opcode at all.
    """
    x = pounce.NlExpr.vars(2)
    p = pounce.build_nl_problem(
        n=2,
        objective=pounce.NlExpr.sum(
            [
                pounce.NlExpr.atan2(x[0], x[1]),
                pounce.NlExpr.min(x[0], x[1]),
                pounce.NlExpr.max(x[0], x[1]),
                x[0].erf(),
            ]
        ),
    )
    a, b = 0.8, 1.5
    assert p.objective([a, b]) == pytest.approx(
        math.atan2(a, b) + min(a, b) + max(a, b) + math.erf(a)
    )


@pytest.mark.parametrize("value", [0.0, 0.5, -1.3, 2.0, -4.0])
def test_erf_value_and_derivative(value):
    x = pounce.NlExpr.var(0)
    p = pounce.build_nl_problem(n=1, objective=x.erf())
    assert p.objective([value]) == pytest.approx(math.erf(value), abs=1e-14)

    # erf'(u) = 2/sqrt(pi) * exp(-u^2)
    want = 2.0 / math.sqrt(math.pi) * math.exp(-(value**2))
    np.testing.assert_allclose(p.gradient([value]), [want], rtol=1e-12)

    # erf''(u) = -2u * erf'(u)
    hv = p.hessian([value])
    if hv.size:  # the structural entry exists for every nonconstant erf
        np.testing.assert_allclose(hv, [-2.0 * value * want], rtol=1e-9, atol=1e-12)


def test_select_and_compare_route_through_active_branch():
    x = pounce.NlExpr.vars(2)
    # f = x1^2 if x0 > 0 else x1^3 — value and derivative both follow the
    # live branch; the branch test itself contributes nothing.
    expr = pounce.NlExpr.select(
        pounce.NlExpr.compare(">", x[0], 0.0), x[1] ** 2, x[1] ** 3
    )
    p = pounce.build_nl_problem(n=2, objective=expr)

    assert p.objective([1.0, 3.0]) == pytest.approx(9.0)
    np.testing.assert_allclose(p.gradient([1.0, 3.0]), [0.0, 6.0])

    assert p.objective([-1.0, 3.0]) == pytest.approx(27.0)
    np.testing.assert_allclose(p.gradient([-1.0, 3.0]), [0.0, 27.0])


def test_compare_rejects_unknown_operator():
    with pytest.raises(ValueError, match="unknown operator"):
        pounce.NlExpr.compare("=<", pounce.NlExpr.var(0), 1.0)


# ---- NlExpr operator surface ------------------------------------------


def test_operator_coercion_both_directions():
    x = pounce.NlExpr.var(0)
    cases = {
        "add": (x + 3, 5.0),
        "radd": (3 + x, 5.0),
        "sub": (x - 3, -1.0),
        "rsub": (3 - x, 1.0),
        "mul": (x * 3, 6.0),
        "rmul": (3 * x, 6.0),
        "div": (x / 4, 0.5),
        "rdiv": (8 / x, 4.0),
        "pow": (x**3, 8.0),
        "rpow": (3**x, 9.0),
        "neg": (-x, -2.0),
        "pos": (+x, 2.0),
        "abs": (abs(-x), 2.0),
    }
    for name, (expr, want) in cases.items():
        assert expr.eval([2.0]) == pytest.approx(want), name


def test_unary_math_methods_match_stdlib():
    x = pounce.NlExpr.var(0)
    at = 0.4
    for name, ref in [
        ("sqrt", math.sqrt),
        ("exp", math.exp),
        ("log", math.log),
        ("log10", math.log10),
        ("sin", math.sin),
        ("cos", math.cos),
        ("tan", math.tan),
        ("asin", math.asin),
        ("acos", math.acos),
        ("atan", math.atan),
        ("sinh", math.sinh),
        ("cosh", math.cosh),
        ("tanh", math.tanh),
        ("asinh", math.asinh),
        ("atanh", math.atanh),
        ("erf", math.erf),
    ]:
        got = getattr(x, name)().eval([at])
        assert got == pytest.approx(ref(at), rel=1e-12), name
    assert x.acosh().eval([2.5]) == pytest.approx(math.acosh(2.5))


def test_expr_eval_gradient_and_variables():
    x = pounce.NlExpr.vars(3)
    e = x[0] * x[2]
    assert e.variables() == [0, 2]
    assert e.eval([2.0, 99.0, 5.0]) == pytest.approx(10.0)
    np.testing.assert_allclose(e.gradient([2.0, 99.0, 5.0]), [5.0, 0.0, 2.0])


def test_expr_eval_rejects_short_x():
    with pytest.raises(ValueError, match="variable 2"):
        pounce.NlExpr.var(2).eval([1.0, 2.0])


def test_expr_rejects_non_numeric_operand():
    x = pounce.NlExpr.var(0)
    with pytest.raises(TypeError):
        x + "1.0"
    with pytest.raises(TypeError):
        x * object()


def test_expr_repr_renders_small_expressions():
    x = pounce.NlExpr.vars(2)
    assert repr(x[0] + x[1]) == "NlExpr(x[0] + x[1])"
    # A large expression falls back to a node count rather than rendering
    # something unbounded.
    big = pounce.NlExpr.sum([x[0]] * 200)
    assert "nodes" in repr(big)


def test_min_max_require_operands():
    with pytest.raises(ValueError, match="at least one operand"):
        pounce.NlExpr.min()


# ---- Hessian-vector product -------------------------------------------


def _dense_hessian(p, x, lam=None, obj_factor=1.0):
    hr, hc = p.hessian_structure()
    hv = p.hessian(x, lam, obj_factor)
    dense = np.zeros((p.n, p.n))
    for i, j, v in zip(hr, hc, hv):
        dense[i, j] += v
        if i != j:
            dense[j, i] += v
    return dense


def test_hessian_vector_product_matches_dense():
    x = pounce.NlExpr.vars(3)
    p = pounce.build_nl_problem(
        n=3,
        objective=x[0] * x[1] * x[2] + (x[0] * x[1]).exp() + x[2].erf(),
        constraints=[x[0] ** 2 + x[2].sin(), x[1] * x[2]],
        g_l=[0.0, 0.0],
        g_u=[1.0, 1.0],
    )
    pt = np.array([0.3, -0.7, 1.1])
    lam = np.array([0.5, -1.25])
    obj_factor = 2.0

    dense = _dense_hessian(p, pt, lam, obj_factor)
    for v in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.4, -1.3, 2.0]),
    ):
        got = p.hessian_vector_product(pt, v, lam, obj_factor)
        np.testing.assert_allclose(got, dense @ v, rtol=1e-9, atol=1e-12)


def test_hessian_vector_product_defaults_to_objective_block():
    x = pounce.NlExpr.vars(2)
    # f = x0^2 + 3 x0 x1  ->  H = [[2, 3], [3, 0]]
    p = pounce.build_nl_problem(
        n=2, objective=x[0] ** 2 + 3 * x[0] * x[1], constraints=[x[0] * x[1]]
    )
    got = p.hessian_vector_product([0.5, 2.0], [1.0, 1.0])
    np.testing.assert_allclose(got, [5.0, 3.0], atol=1e-12)


def test_hessian_vector_product_validates_lengths():
    p = pounce.build_nl_problem(n=2, objective=pounce.NlExpr.var(0) ** 2)
    with pytest.raises(ValueError, match="v"):
        p.hessian_vector_product([1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="x"):
        p.hessian_vector_product([1.0], [1.0, 1.0])


def test_hessian_vector_product_on_read_nl_model(tmp_path):
    """The HVP is a method on every ``NlProblem``, however it was built."""
    nl_file = tmp_path / "simple.nl"
    nl_file.write_text(SIMPLE_NL)
    p = pounce.read_nl(str(nl_file))
    # min (x0-1)^2 + (x1-2)^2  ->  H = 2I
    got = p.hessian_vector_product([0.0, 0.0], [1.5, -2.5])
    np.testing.assert_allclose(got, [3.0, -5.0], atol=1e-12)


# ---- HVP against sparse and dense Hessians ----------------------------
#
# Two model shapes, because they stress different things. The chain
# objective's Hessian is tridiagonal — the sparse shape an IPM actually
# meets, where most of the matrix is structural zero and a bug that leaked
# coupling between distant variables would hide in a 3x3 test. The coupled
# objective's Hessian is completely full, so nothing is masked by a zero.


def _chain_problem(n=10):
    """Sum of (x_i * x_{i+1})^2 + exp(x_i): tridiagonal Hessian."""
    x = pounce.NlExpr.vars(n)
    terms = [(x[i] * x[i + 1]) ** 2 for i in range(n - 1)]
    terms += [x[i].exp() for i in range(n)]
    return pounce.build_nl_problem(n=n, objective=pounce.NlExpr.sum(terms))


def _coupled_problem(n=6):
    """exp(sum x_i) + (sum x_i)^3: every variable pair couples."""
    x = pounce.NlExpr.vars(n)
    s = pounce.NlExpr.sum(list(x))
    return pounce.build_nl_problem(n=n, objective=s.exp() + s**3)


def test_sparse_hessian_model_structure_is_actually_sparse():
    n = 10
    p = _chain_problem(n)
    # Tridiagonal lower triangle: n diagonal + (n-1) sub-diagonal entries,
    # far short of the n(n+1)/2 a dense Hessian would store.
    assert p.nnz_hess == 2 * n - 1 < n * (n + 1) // 2

    hr, hc = p.hessian_structure()
    assert set(zip(hr.tolist(), hc.tolist())) == (
        {(i, i) for i in range(n)} | {(i + 1, i) for i in range(n - 1)}
    )


def test_dense_hessian_model_structure_is_actually_dense():
    n = 6
    p = _coupled_problem(n)
    assert p.nnz_hess == n * (n + 1) // 2


@pytest.mark.parametrize("make", [_chain_problem, _coupled_problem])
def test_hvp_matches_both_sparse_and_dense_hessian_forms(make):
    """The HVP must agree with the sparse COO Hessian *and* with its dense
    expansion, on a sparse-Hessian and a dense-Hessian model alike."""
    p = make()
    n = p.n
    rng = np.random.default_rng(0)
    pt = rng.normal(0.0, 0.3, n)

    # Reference 1: the sparse triangle, applied as a sparse matrix.
    hr, hc = p.hessian_structure()
    hv = p.hessian(pt)
    lower = sp.coo_matrix((hv, (hr, hc)), shape=(n, n)).tocsr()
    diag = sp.diags(lower.diagonal())
    sparse_H = lower + lower.T - diag

    # Reference 2: the same thing densified.
    dense_H = sparse_H.toarray()
    np.testing.assert_allclose(dense_H, dense_H.T, atol=1e-12)

    for v in (
        np.ones(n),
        rng.normal(0.0, 1.0, n),
        np.eye(n)[0],
        np.zeros(n),
    ):
        got = p.hessian_vector_product(pt, v)
        assert got.shape == (n,)
        np.testing.assert_allclose(got, sparse_H @ v, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(got, dense_H @ v, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("make", [_chain_problem, _coupled_problem])
def test_hvp_accepts_scipy_sparse_directions(make):
    """A SciPy sparse ``v`` gives the same answer as its dense twin.

    The result is dense either way: ``H @ v`` is dense in general even when
    both factors are sparse, so a sparse return type would promise an
    economy this product does not have.
    """
    p = make()
    n = p.n
    pt = np.linspace(-0.4, 0.4, n)

    dense_v = np.zeros(n)
    dense_v[0] = 1.5
    dense_v[n // 2] = -2.0
    want = p.hessian_vector_product(pt, dense_v)

    # Version-proof forms: an explicit (n, 1) column is (n, 1) on every
    # scipy that has ever shipped these classes.
    candidates = [
        sp.csc_matrix(dense_v[:, None]),  # (n, 1) column
        sp.csr_matrix(dense_v[:, None]),
        sp.coo_matrix(dense_v[:, None]),
    ]
    # The 1-D sparse *array* API only became genuinely 1-D in scipy 1.14;
    # 1.11 shapes `csr_array(v_1d)` as (1, n) (which this API rejects as a
    # row vector) and 1.13 raises inside scipy. The package floor is
    # scipy>=1.11, so this half of the coverage is version-gated rather
    # than silently depending on CI installing the latest.
    if _SCIPY >= (1, 14):
        candidates += [sp.csr_array(dense_v), sp.coo_array(dense_v)]
        assert sp.coo_array(dense_v).shape == (n,), "expected a genuinely 1-D sparse array"

    for sparse_v in candidates:
        got = p.hessian_vector_product(pt, sparse_v)
        assert isinstance(got, np.ndarray), "the result must be dense"
        np.testing.assert_allclose(np.asarray(got).ravel(), want, rtol=1e-10)

    # An all-zero sparse direction is a legal (and cheap) input.
    zero = sp.csr_matrix((n, 1))
    np.testing.assert_allclose(
        np.asarray(p.hessian_vector_product(pt, zero)).ravel(), np.zeros(n), atol=0.0
    )


@pytest.mark.parametrize("make", [_chain_problem, _coupled_problem])
def test_hvp_block_form_dense_and_sparse(make):
    """A block of directions returns (n, k) and matches column-by-column
    single calls, whether the block arrives dense or sparse."""
    p = make()
    n = p.n
    rng = np.random.default_rng(7)
    pt = rng.normal(0.0, 0.3, n)

    V = rng.normal(0.0, 1.0, (n, 4))
    V[:, 1] = 0.0  # a skipped direction, in the middle of live ones

    block = p.hessian_vector_product(pt, V)
    assert block.shape == (n, 4)
    for c in range(4):
        single = p.hessian_vector_product(pt, V[:, c])
        np.testing.assert_allclose(block[:, c], single, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(block[:, 1], np.zeros(n), atol=0.0)

    # Same block, arriving sparse.
    sparse_block = p.hessian_vector_product(pt, sp.csc_matrix(V))
    np.testing.assert_allclose(sparse_block, block, rtol=1e-11, atol=1e-13)

    # And against the assembled Hessian.
    hr, hc = p.hessian_structure()
    lower = sp.coo_matrix((p.hessian(pt), (hr, hc)), shape=(n, n)).tocsr()
    H = lower + lower.T - sp.diags(lower.diagonal())
    np.testing.assert_allclose(block, H @ V, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("make", [_chain_problem, _coupled_problem])
def test_hvp_block_of_unit_seeds_reconstructs_the_hessian(make):
    """Densify-via-HVP: one call with the identity recovers the whole
    matrix, and it matches the sparse triangle entry for entry."""
    p = make()
    n = p.n
    pt = np.linspace(-0.3, 0.5, n)

    dense_from_hvp = p.hessian_vector_product(pt, np.eye(n))
    assert dense_from_hvp.shape == (n, n)
    np.testing.assert_allclose(dense_from_hvp, dense_from_hvp.T, rtol=1e-9, atol=1e-11)

    hr, hc = p.hessian_structure()
    hv = p.hessian(pt)
    for i, j, val in zip(hr, hc, hv):
        assert dense_from_hvp[i, j] == pytest.approx(val, rel=1e-9, abs=1e-11)
        assert dense_from_hvp[j, i] == pytest.approx(val, rel=1e-9, abs=1e-11)

    # Every entry outside the stored sparsity must be a structural zero —
    # this is the half of the check a sparse-only comparison cannot make.
    stored = set(zip(hr.tolist(), hc.tolist()))
    for i in range(n):
        for j in range(n):
            if (max(i, j), min(i, j)) not in stored:
                assert dense_from_hvp[i, j] == 0.0, f"leak at ({i}, {j})"


def test_hvp_block_with_lagrangian_multipliers():
    """Constraint blocks come along for the ride in the block form."""
    x = pounce.NlExpr.vars(3)
    p = pounce.build_nl_problem(
        n=3,
        objective=(x[0] * x[1]).exp() + x[2] ** 4,
        constraints=[x[0] * x[2], x[1] ** 3],
        g_l=[0.0, 0.0],
        g_u=[1.0, 1.0],
    )
    pt = np.array([0.4, -0.6, 1.3])
    lam = np.array([0.75, -1.5])
    V = np.array([[1.0, 0.0], [2.0, 1.0], [-3.0, 0.5]])

    block = p.hessian_vector_product(pt, V, lam, 2.0)
    dense = _dense_hessian(p, pt, lam, 2.0)
    np.testing.assert_allclose(block, dense @ V, rtol=1e-9, atol=1e-11)

    # Sparse multipliers-free call on the same block, for contrast.
    obj_only = p.hessian_vector_product(pt, sp.csc_matrix(V))
    np.testing.assert_allclose(
        obj_only, _dense_hessian(p, pt, None, 1.0) @ V, rtol=1e-9, atol=1e-11
    )


def test_hvp_rejects_wrong_shapes():
    p = _chain_problem(5)
    with pytest.raises(ValueError, match="expected length 5"):
        p.hessian_vector_product(np.zeros(5), np.ones(4))
    # A row vector is not a direction — the message says so.
    with pytest.raises(ValueError, match=r"expected shape \(5,\) or \(5, k\)"):
        p.hessian_vector_product(np.zeros(5), np.ones((1, 5)))
    with pytest.raises(ValueError, match=r"expected shape \(5,\) or \(5, k\)"):
        p.hessian_vector_product(np.zeros(5), sp.csr_matrix(np.ones((1, 5))))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        p.hessian_vector_product(np.zeros(5), np.ones((5, 2, 2)))


def test_hvp_accepts_awkward_dense_inputs():
    """Lists, integer arrays, and non-contiguous views are all directions."""
    p = _chain_problem(4)
    pt = np.array([0.1, 0.2, 0.3, 0.4])
    want = p.hessian_vector_product(pt, np.array([1.0, 0.0, 1.0, 0.0]))

    np.testing.assert_allclose(p.hessian_vector_product(pt, [1, 0, 1, 0]), want)
    np.testing.assert_allclose(
        p.hessian_vector_product(pt, np.array([1, 0, 1, 0], dtype=np.int64)), want
    )
    # A strided view: every other entry of an 8-long array.
    strided = np.array([1.0, 9.0, 0.0, 9.0, 1.0, 9.0, 0.0, 9.0])[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(p.hessian_vector_product(pt, strided), want)
    # A column of a Fortran-ordered 2-D array.
    col = np.asfortranarray(np.array([[1.0, 5.0], [0.0, 5.0], [1.0, 5.0], [0.0, 5.0]]))[
        :, 0
    ]
    np.testing.assert_allclose(p.hessian_vector_product(pt, col), want)


def test_hvp_empty_block_is_legal():
    p = _chain_problem(4)
    got = p.hessian_vector_product(np.zeros(4), np.zeros((4, 0)))
    assert got.shape == (4, 0)


# ---- Expression depth guard -------------------------------------------
#
# Every consumer of an `Expr` recurses once per nesting level — the tape
# builder, the problem assembler, the teardown when the last handle goes,
# and the `.nl` parser that produces one in the first place — so a deep
# enough tree overflows the stack. That is a SIGSEGV, not an exception: the
# interpreter dies with no traceback and nothing to catch.
#
# Two things together make it unreachable. Those walks run on a worker
# thread with a 64 MB stack, so what is survivable stops depending on the
# calling thread (8 MB on a macOS/Linux main thread, 1 MB on Windows, less
# on a `threading.Thread`); and a limit then keeps the depth well inside
# what that stack holds. Both doors share the limit: `NlExpr` enforces it
# as it builds, and `read_nl` / `parse_nl_text` enforce it on what they
# parsed, since a model that arrives already built cannot be capped as it
# is constructed.


def test_deep_chain_raises_instead_of_crashing():
    e = pounce.NlExpr.var(0)
    with pytest.raises(ValueError, match="nesting would reach depth"):
        for _ in range(pounce.NlExpr.max_depth + 10):
            e = e + 1.0


def test_depth_error_points_at_the_cheap_alternative():
    """The error has to teach: `e = e + t` in a loop is the first thing a
    modeler writes, and one flat `sum` node is the fix."""
    e = pounce.NlExpr.var(0)
    with pytest.raises(ValueError) as excinfo:
        for _ in range(pounce.NlExpr.max_depth + 10):
            e = e * 2.0
    assert "NlExpr.sum" in str(excinfo.value)


def test_depth_is_tracked_across_every_node_kind():
    x = pounce.NlExpr.vars(2)
    assert pounce.NlExpr.var(0).depth == 1
    assert pounce.NlExpr.const_(1.0).depth == 1
    assert (x[0] + x[1]).depth == 2
    assert (-x[0]).depth == 2
    assert x[0].sin().depth == 2
    assert (x[0] + x[1]).sin().depth == 3
    # n-ary nodes are ONE level regardless of width — the whole reason
    # `sum` is the answer to the depth cap.
    wide = pounce.NlExpr.sum([x[0]] * 10_000)
    assert wide.depth == 2
    assert pounce.NlExpr.min(*([x[0]] * 500)).depth == 2
    # Depth follows the deepest operand, not the first.
    deep = x[0].sin().cos().exp()
    assert deep.depth == 4
    assert (x[1] + deep).depth == 5
    assert pounce.NlExpr.select(x[0], deep, x[1]).depth == 5


def test_wide_model_is_unaffected_by_the_depth_cap():
    """200k terms through `sum` is fine; the cap only bounds nesting."""
    n = 50
    x = pounce.NlExpr.vars(n)
    p = pounce.build_nl_problem(
        n=n, objective=pounce.NlExpr.sum([xi**2 for xi in x] * 4000)
    )
    assert p.objective(np.ones(n)) == pytest.approx(n * 4000)


def test_expression_just_under_the_cap_still_tapes_and_evaluates():
    """The cap must leave a usable expression usable — build to one level
    below it and run the whole pipeline."""
    # `var(0)` is depth 1 and each `+` adds one level, so this lands
    # exactly on the cap — the deepest expression the class will build.
    adds = pounce.NlExpr.max_depth - 1
    e = pounce.NlExpr.var(0)
    for _ in range(adds):
        e = e + 1.0
    assert e.depth == pounce.NlExpr.max_depth
    assert e.eval([0.0]) == pytest.approx(adds)
    np.testing.assert_allclose(e.gradient([0.0]), [1.0])
    p = pounce.build_nl_problem(n=1, objective=e)
    assert p.objective([0.0]) == pytest.approx(adds)
    # And one more level is refused.
    with pytest.raises(ValueError, match="nesting would reach depth"):
        _ = e + 1.0


# ---- Operands are shared, not copied ----------------------------------


def test_building_an_expression_does_not_copy_its_operands():
    """A regression guard on the O(N^2) build, not a benchmark.

    Each operator used to deep-copy both operands, so accumulating in a
    loop copied the whole chain every iteration: 5 000 terms took over five
    seconds. Referencing the operands instead makes it a few milliseconds,
    so the budget can be loose enough that machine speed does not matter
    and still catch a return to copying.
    """
    started = time.perf_counter()
    e = pounce.NlExpr.var(0)
    for _ in range(5_000):
        e = e + 1.0
    p = pounce.build_nl_problem(n=1, objective=e)
    elapsed = time.perf_counter() - started
    assert p.objective([1.0]) == pytest.approx(5_001.0)
    assert elapsed < 1.0, f"building 5 000 terms took {elapsed:.2f}s"


def test_a_reused_subexpression_evaluates_like_a_written_out_copy():
    """Reusing a Python name shares the subtree now rather than copying it.
    The tape emits a shared body once and sums the adjoint contributions
    from every reference, which is the same function and the same
    derivatives.

    Values match exactly — the forward sweep does the identical arithmetic,
    just once. Derivatives are compared to rounding: the reverse sweep sums
    one slot's contributions where the copy sums two, and floating-point
    addition does not promise the same order gives the same last bit."""

    def model(reuse):
        x = pounce.NlExpr.vars(2)
        if reuse:
            t = x[0] * x[1] + x[0].sin()
            obj, con = t * t + t, t.exp()
        else:
            # The same thing with nothing shared: a fresh `t` per use.
            def t():
                return x[0] * x[1] + x[0].sin()

            obj, con = t() * t() + t(), t().exp()
        return pounce.build_nl_problem(n=2, objective=obj, constraints=[con])

    shared, copied = model(True), model(False)
    pt = [0.7, -1.3]
    assert shared.objective(pt) == copied.objective(pt)
    np.testing.assert_array_equal(shared.constraints(pt), copied.constraints(pt))
    np.testing.assert_allclose(shared.gradient(pt), copied.gradient(pt), rtol=1e-14)
    np.testing.assert_allclose(shared.jacobian(pt), copied.jacobian(pt), rtol=1e-14)

    rows, cols = shared.hessian_structure()
    np.testing.assert_array_equal(rows, copied.hessian_structure()[0])
    np.testing.assert_array_equal(cols, copied.hessian_structure()[1])
    lam = np.array([0.6])
    np.testing.assert_allclose(
        shared.hessian(pt, lam), copied.hessian(pt, lam), rtol=1e-14
    )


def test_a_shared_subexpression_is_not_expanded():
    """`e = e * e` doubles the *size* of a copied expression every step, so
    40 steps is a trillion nodes and no machine builds it. Referencing the
    operand keeps it 40 nodes, and every walk over the result — the tape
    build, the variable scan, the assembly — visits a shared body once
    rather than following each of the 2^40 paths to it."""
    e = pounce.NlExpr.var(0)
    for _ in range(40):
        e = e * e
    # x ** (2 ** 40), whose derivative at 1 is 2 ** 40.
    assert e.eval([1.0]) == 1.0
    np.testing.assert_allclose(e.gradient([1.0]), [2.0**40])
    p = pounce.build_nl_problem(n=1, objective=e)
    assert p.objective([1.0]) == 1.0
    np.testing.assert_allclose(p.gradient([1.0]), [2.0**40])
    np.testing.assert_allclose(p.hessian([1.0]), [2.0**40 * (2.0**40 - 1)])


def test_a_loop_built_objective_still_tapes_per_term():
    """Sharing must not cost the summand split. Operand references are
    inlined when the model is assembled, so `from_expressions` sees the
    plain `+` chain and gives each term its own tape — the same problem the
    flat `sum` spelling produces, entry for entry."""
    n = 60
    x = pounce.NlExpr.vars(n)
    terms = [(x[i] * x[i + 1]) ** 2 for i in range(n - 1)]
    looped = terms[0]
    for t in terms[1:]:
        looped = looped + t
    by_loop = pounce.build_nl_problem(n=n, objective=looped)
    by_sum = pounce.build_nl_problem(n=n, objective=pounce.NlExpr.sum(terms))

    pt = np.linspace(0.1, 1.0, n)
    assert by_loop.objective(pt) == pytest.approx(by_sum.objective(pt))
    for got, want in zip(by_loop.hessian_structure(), by_sum.hessian_structure()):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(by_loop.hessian(pt), by_sum.hessian(pt), rtol=1e-14)


# ---- Depth reached through the parser, not the builder -----------------


def _deep_nl(depth):
    """`.nl` text whose objective is `((v0 + 1) + 1) ...`, `depth` deep."""
    body = "o0\n" * depth + "v0\n" + "n1\n" * depth
    return (
        "g3 0 1 0\n1 0 1 0 0\n0 1\n0 0\n0 1 0\n0 0 0 1\n0 0 0 0 0\n"
        f"0 0\n0 0\n0 0 0 0 0\nO0 0\n{body}x0\nr\nb\n3\nk0\n"
    )


def test_parsed_nl_deeper_than_the_caller_stack_would_take():
    """A `.nl` file arrives already built, so the construction-time cap
    cannot see it — this used to segfault at depth 4 000, and predates the
    `NlExpr` surface entirely."""
    for depth in (3_000, 5_000, pounce.NlExpr.max_depth - 1):
        p = pounce.parse_nl_text(_deep_nl(depth))
        assert p.objective([0.0]) == pytest.approx(float(depth))


def test_read_nl_takes_the_same_depth_from_a_file(tmp_path):
    path = tmp_path / "deep.nl"
    path.write_text(_deep_nl(5_000))
    p = pounce.read_nl(str(path))
    assert p.objective([0.0]) == pytest.approx(5_000.0)


def test_both_doors_enforce_the_same_depth_limit():
    """The inconsistency the issue called out: `NlExpr` refused depth 1 001
    while the parser accepted 3 000 and crashed on 4 000. One limit now,
    enforced wherever an expression can enter."""
    limit = pounce.NlExpr.max_depth

    with pytest.raises(ValueError) as parsed:
        pounce.parse_nl_text(_deep_nl(limit + 1))
    assert str(limit) in str(parsed.value)
    assert "o54" in str(parsed.value), "the parser's error names the flat form"

    e = pounce.NlExpr.var(0)
    with pytest.raises(ValueError) as built:
        for _ in range(limit + 10):
            e = e + 1.0
    assert str(limit) in str(built.value)


# Deep enough that the walks need more stack than the thread below has.
_SMALL_STACK_SCRIPT = """
import threading
import pounce

threading.stack_size(256 * 1024)
out = []


def deep_nl(depth):
    body = "o0\\n" * depth + "v0\\n" + "n1\\n" * depth
    return (
        "g3 0 1 0\\n1 0 1 0 0\\n0 1\\n0 0\\n0 1 0\\n0 0 0 1\\n0 0 0 0 0\\n"
        "0 0\\n0 0\\n0 0 0 0 0\\nO0 0\\n" + body + "x0\\nr\\nb\\n3\\nk0\\n"
    )


def work():
    e = pounce.NlExpr.var(0)
    for _ in range(1999):
        e = e + 1.0
    assert e.variables() == [0]
    assert e.eval([1.0]) == 2000.0
    p = pounce.build_nl_problem(n=1, objective=e)
    assert p.objective([1.0]) == 2000.0
    assert p.variant(x0=[2.0]).objective([1.0]) == 2000.0
    del p, e

    # The parser reaches the same recursion from the other side.
    q = pounce.parse_nl_text(deep_nl(2000))
    assert q.objective([0.0]) == 2000.0
    del q
    out.append(True)


t = threading.Thread(target=work)
t.start()
t.join()
assert out == [True]
"""


def test_deep_work_survives_a_small_caller_stack():
    """The limit is only half the fix: the walks whose frames are too big
    to fit run on a worker thread with a stack sized for them, so what is
    safe does not depend on the caller's stack. 256 KB is well under any
    platform's main thread, and a 2 000-deep tape build wants roughly twice
    that on its own.

    Out-of-process because the failure being guarded is a SIGSEGV, which
    would take the test session down with it rather than fail a test.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SMALL_STACK_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)


# ---- Other NlExpr surface details -------------------------------------


def test_numpy_array_times_expr_raises_in_both_directions():
    """Without `__array_ufunc__ = None`, the reflected form silently
    returns a dtype=object ndarray of NlExpr instead of raising."""
    x = pounce.NlExpr.var(0)
    with pytest.raises(TypeError):
        x * np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        np.array([1.0, 2.0]) * x
    with pytest.raises(TypeError):
        np.array([1.0, 2.0]) + x


def test_expr_supports_copy_and_deepcopy():
    import copy

    x = pounce.NlExpr.vars(2)
    e = x[0] * x[1] + 3.0
    for clone in (copy.copy(e), copy.deepcopy(e)):
        assert clone is not e
        assert clone.depth == e.depth
        assert clone.eval([2.0, 5.0]) == pytest.approx(e.eval([2.0, 5.0]))


def test_expr_is_not_picklable_and_says_so():
    """Not supported — pinned so it stays a clean TypeError rather than
    silently producing a broken object."""
    import pickle

    with pytest.raises((TypeError, NotImplementedError, pickle.PicklingError)):
        pickle.dumps(pounce.NlExpr.var(0))


def test_build_nl_problem_rejects_imported_function_expressions():
    """There is no way to construct an `Expr::Funcall` from Python today,
    so this pins the *reachable* surface: nothing in the builder produces
    one, and the Rust-side guard covers the path that could."""
    x = pounce.NlExpr.vars(2)
    p = pounce.build_nl_problem(n=2, objective=x[0] * x[1])
    assert p.objective([2.0, 3.0]) == pytest.approx(6.0)


# ---- Strided / awkward x and lam --------------------------------------


def test_evaluators_accept_strided_x():
    """`x` gets the same treatment the docstring promises for `v` — a
    non-contiguous view must not produce a bare numpy contiguity error."""
    p = _chain_problem(4)
    want_pt = np.array([0.1, 0.2, 0.3, 0.4])
    strided = np.array([0.1, 9.0, 0.2, 9.0, 0.3, 9.0, 0.4, 9.0])[::2]
    assert not strided.flags["C_CONTIGUOUS"]

    assert p.objective(strided) == pytest.approx(p.objective(want_pt))
    np.testing.assert_allclose(p.gradient(strided), p.gradient(want_pt))
    np.testing.assert_allclose(p.hessian(strided), p.hessian(want_pt))
    np.testing.assert_allclose(
        p.hessian_vector_product(strided, np.ones(4)),
        p.hessian_vector_product(want_pt, np.ones(4)),
    )


def test_wrong_length_x_names_the_argument():
    p = _chain_problem(4)
    with pytest.raises(ValueError, match="objective: x"):
        p.objective(np.zeros(3))


# ---- Documented HVP semantics -----------------------------------------


def test_hvp_does_not_propagate_nan_through_structural_zeros():
    """Deliberate, and worth pinning so no future change claims exact
    `H @ v` equivalence.

    AD never multiplies by a structural zero — the term is not in the tape
    at all — so a NaN in one component of `v` stays confined to the
    variables actually coupled to it. A dense `H @ v` computes `0 * nan`
    and smears NaN across every row.
    """
    x = pounce.NlExpr.vars(4)
    # Block diagonal: {x0, x1} and {x2, x3} never interact.
    p = pounce.build_nl_problem(
        n=4, objective=(x[0] * x[1]).exp() + (x[2] * x[3]).exp()
    )
    pt = np.array([0.1, 0.2, 0.3, 0.4])
    v = np.array([np.nan, 0.0, 1.0, 0.0])

    got = p.hessian_vector_product(pt, v)
    assert np.isnan(got[0]) and np.isnan(got[1]), "the coupled block sees the NaN"
    assert np.all(np.isfinite(got[2:])), "the uncoupled block must stay clean"

    # The dense product, for contrast: 0 * nan = nan, everywhere.
    dense = _dense_hessian(p, pt)
    assert np.all(np.isnan(dense @ v))
