# Global Optimization

Most of POUNCE settles a problem at a **local** optimum (the NLP filter-IPM and
SQP) or exploits convexity so that local *is* global (the convex/conic IPM).
For a genuinely **nonconvex** problem, the path to a *certified* global optimum
that **ships in this release** is for polynomials:

- **The SOS / Lasserre hierarchy** (`pounce-convex`) — for **polynomial**
  problems, via a single semidefinite program. Callable from Rust
  (`sos_minimize`) and Python (`pounce.sos_minimize`).

It returns a result that is *certified*: a lower bound together with a moment
certificate that, when exact, pins the global minimum and recovers its
minimizer(s).

> A second path — general-purpose **spatial branch-and-bound** (`pounce-global`)
> for factorable nonconvex NLPs with `exp`/`ln`/trig — is **in development on
> the `feature/global` branch and is not part of this release**. It is described
> at the end of this chapter for context, but there is no `pounce-global` crate
> in the shipped workspace, no `pounce.minimize_global` Python entry point, and
> no `--solver global` CLI route here.

## The SOS / Lasserre path (polynomials)

When the objective and constraints are **polynomials**, the
sum-of-squares / moment approach in `pounce-convex` certifies the global
minimum from a *single* semidefinite program — no branching — by searching for
the largest `γ` such that `p(x) − γ` lies in the Putinar cone (a sum of squares
plus constraint multipliers). The SDP is solved by POUNCE's own convex conic
interior-point method; flat truncation of the resulting moment matrix certifies
when the bound is exact, and a **facial-reduction** step recovers every global
minimizer — even when the optimum is attained at several points.

From Python, a polynomial is a **dict mapping an exponent tuple to its
coefficient** (the all-zeros key is the constant term):

```python
from pounce.sos import sos_minimize

# x**4 - 2 x**2 + 3  ->  global minimum 2, attained at BOTH x = +1 and x = -1
r = sos_minimize({(4,): 1.0, (2,): -2.0, (0,): 3.0})
r.lower_bound       # ≈ 2.0
r.is_exact          # True — flat-truncation certificate: the bound is the minimum
r.minimizers        # both x = +1 and x = -1
```

Constraints are polynomials too, passed as `inequalities` (`g_i(x) ≥ 0`) and
`equalities` (`h_j(x) = 0`); raise the relaxation `order` to tighten the bound
(the Lasserre hierarchy) at the cost of a larger SDP. A runnable walkthrough —
double well, a constrained problem, and a 2-D example — is in
[`18_sos_global_optimization.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/18_sos_global_optimization.ipynb).

The same solver from Rust:

```rust
use pounce_convex::{sos_minimize, PolyProblem, Polynomial};
# use pounce_feral::FeralSolverInterface;
# use pounce_linsol::SparseSymLinearSolverInterface;
# fn backend() -> Box<dyn SparseSymLinearSolverInterface> { Box::new(FeralSolverInterface::new()) }
// x⁴ − 2x² + 3 → global minimum 2 at x = ±1.
let p = Polynomial::new(1, vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 3.0)]);
let sol = sos_minimize(&PolyProblem::new(p), None, backend);
// sol.lower_bound ≈ 2; when the moment matrix is flat, sol.minimizers holds
// the global minimizer(s) — here both x = +1 and x = −1.
```

The full treatment lives in the `pounce_convex::sos` module documentation.

**When SOS fits:** polynomials of modest degree and dimension — one SDP,
recovers all global minimizers, but the SDP grows with the relaxation order.
For general factorable problems (`exp`/`ln`/trig), or polynomials where the SDP
would be too large, the tool is spatial branch-and-bound — which is still in
development (below).

## Spatial branch-and-bound (in development)

> **Not in this release.** Everything in this section describes the
> `pounce-global` crate as it exists on the `feature/global` branch. It is not
> in the shipped workspace, and the Rust snippets below will not compile against
> the published crates. There is no Python or CLI binding for it in this
> release. The section is kept for design context and to set expectations for
> what the general nonconvex path will look like.

### The problem

```text
minimize    f(x)
subject to  cl_j ≤ g_j(x) ≤ cu_j        (j = 0 … m−1)
            x_lo ≤ x ≤ x_hi
```

`f` and the `g_j` are **factorable** — built from `+ − × ÷`, integer powers,
`√`, `exp`, `ln`, `|·|`, `sin`, and `cos`. A bounded box is required (the
relaxation needs finite bounds).

### The idea

Branch-and-bound brackets the global optimum between a **lower bound** (valid
over a region) and an **upper bound** (the value of some feasible point), then
subdivides the search region until the two meet. The whole game is making the
lower bound tight enough, fast enough.

For each node — a box `[lo, hi]` — the solver:

1. **Tightens the box.** Feasibility-based bound tightening (FBBT) propagates
   interval bounds through each constraint; **optimization-based** bound
   tightening (OBBT) then minimizes and maximizes each variable over the
   relaxation (with an incumbent cutoff). Either may prove the box empty, in
   which case it is pruned.
2. **Computes a lower bound.** A convex *relaxation* of the problem over the
   box — built so that it underestimates `f` and contains every feasible point
   — is solved as a linear program through `pounce-convex`. Its optimum is a
   valid lower bound. Crucially the relaxation is **exact in the limit of a
   zero-width box**, so as branching shrinks boxes the bound converges to the
   truth.
3. **Improves the incumbent.** Feasible points are probed (the relaxation
   solution, the box center) and polished with a local NLP solve
   (`pounce-algorithm`), giving a sharp upper bound.
4. **Branches.** The variable with the largest **relaxation violation** (the
   one whose nonconvexity is driving the gap) is split at the relaxation point
   — falling back to the widest box side when nothing is violated — and the two
   child boxes join a best-first frontier ordered by node lower bound.

The search stops when the frontier's lowest bound meets the incumbent within
tolerance — at which point the incumbent is the certified global optimum.

```rust,ignore
// On the `feature/global` branch — not in this release.
use pounce_global::{expr::var, solve_global, GlobalProblem, GlobalOptions, GlobalStatus};
use pounce_feral::FeralSolverInterface;

// Six-hump camel — six local minima, two global (value ≈ −1.0316).
let x = var(0);
let y = var(1);
let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4) + (1.0 / 3.0) * x.clone().powi(6)
    + x.clone() * y.clone() - 4.0 * y.clone().powi(2) + 4.0 * y.powi(4);

let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
let sol = solve_global(&prob, &GlobalOptions::default(),
                       || Box::new(FeralSolverInterface::new()));

assert_eq!(sol.status, GlobalStatus::Optimal);
// sol.objective ≈ −1.0316  (a certified global minimum, not just a local one)
// sol.lower_bound brackets it; sol.gap() is the optimality gap; sol.nodes the
// branch-and-bound node count.
```

Constraints use the same expression DSL — `.ge`, `.le`, `.equality`, and
`.subject_to(g, lo, hi)`; an infeasible problem returns
`GlobalStatus::Infeasible` with a proof:

```rust,ignore
let obj = var(0) + var(1);
let g = var(0) * var(1);
// min x + y  s.t.  x·y ≥ 4 on [1,5]²  → 4 at (2,2)
let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);
```

### The relaxation suite

The lower bound is everything, and POUNCE's is built term by term over the
factorable expression tape (the same `FbbtTape` representation FBBT uses), with
the techniques a state-of-the-art global solver uses:

| Component | Role |
|---|---|
| **Tight univariate envelopes** | The exact convex/concave hull of each atom (`xⁿ`, `√`, `exp`, `ln`, `sin`, `cos`, `|·|`): secant + tangent cuts on a convex/concave arc, the *tangent-from-the-endpoint* construction for single-inflection arcs (odd powers across 0, trig over a sub-π box), and slope-sampled supporting lines for trig over wider boxes. |
| **McCormick** | The exact convex hull of each bilinear product. |
| **Sandwich cuts** | After the LP solve, tangent cuts are added at the solution for loose atoms and the LP re-solved — tightening the bound *without* branching. |
| **OBBT** | Optimization-based bound tightening: the single biggest box reducer. |
| **αBB** | A convex underestimator of the *whole* objective, from a rigorous interval-Hessian spectral shift (`α ≥ max(0, −½λ_min)`), complementing the term-wise relaxation. |
| **RLT** | Level-1 reformulation-linearization: each affine constraint times each variable bound factor, linearized with shared product columns. |
| **Multilinear** | A 3-way product `x·y·z` is relaxed by intersecting all three bilinear groupings, not just the one nested grouping. |

Each is a verified global under/over-estimator — so any of them can be turned
on or off without affecting correctness, only the bound's tightness (and the
node count). On the six-hump camel, the envelope engine alone certifies in 287
nodes; adding sandwich cuts brings it to ~220, and OBBT to ~60.

### Tuning

`GlobalOptions` exposes the gap tolerances and every relaxation knob:

| Field | Default | Meaning |
|---|---|---|
| `abs_gap`, `rel_gap` | `1e-6` | stop when `ub − lb` clears either tolerance |
| `feas_tol` | `1e-6` | constraint tolerance for accepting an incumbent |
| `box_tol` | `1e-7` | stop branching a box this narrow |
| `max_nodes` | `5000` | node budget (else `NodeLimit`, with bound + incumbent) |
| `local_solve_iters` | `50` | IPM iteration cap for the NLP upper-bound polish (`0` off) |
| `sandwich_rounds` | `4` | cutting-plane rounds per node (`0` off) |
| `obbt_passes` | `2` | OBBT sweeps per node (`0` off — costly: `2n` LP solves/pass) |
| `alphabb_cuts` | `1` | αBB tangent planes added to the objective (`0` off) |
| `rlt` | `true` | level-1 RLT cuts |
| `multilinear` | `true` | multi-grouping trilinear relaxation |
| `branching` | `MostViolation` | branching rule: `Widest`, `MostViolation`, or `Reliability` |
| `parallel` | `false` | run OBBT's `2n` solves on a thread pool (deterministic) |
| `threads` | `1` | `> 1` runs the parallel node pool (non-deterministic order) |
| `fbbt` | — | FBBT configuration |

The branching rule (`BranchRule`) chooses the variable to split: `Widest` (box
geometry), `MostViolation` (the variable whose nonconvexity drives the
relaxation gap — the default), or `Reliability` (pseudocosts learned from child
solves, with strong branching until a variable's pseudocost is reliable — the
MILP/MINLP SOTA rule). Because OBBT tightens every node here, the relaxation is
usually tight enough that the rule is second-order; reliability is most useful
on larger problems where variable choice dominates the node count.

The defaults aim for robustness on small problems. OBBT dominates the per-node
cost; turn `obbt_passes` down (or off) on larger problems where the LP solves
outweigh the node savings.

There are two opt-in forms of parallelism:

- **`parallel = true`** parallelizes OBBT's `2n` independent solves per pass on a
  thread pool — *deterministically* (the same nodes and optimum as serial, only
  faster). On a 7-variable problem it cut wall-clock ≈2.3× on 14 cores; the
  speedup is sub-linear because the relaxation build, sandwich cuts, αBB, RLT,
  the local NLP solve, and branching remain serial within a node.
- **`threads > 1`** runs the **node pool**: workers pull whole frontier nodes
  and process them concurrently (OBBT stays serial inside each worker). This is
  coarser-grained and the larger speedup, but **non-deterministic** — the
  certified optimum and gap are unchanged, yet the node count varies run to run
  (parallel best-first explores some nodes a serial run would have pruned). On a
  small 5-variable problem it was ≈2.6× on 14 cores (≈40 nodes — too few to
  saturate the cores); it scales further as the tree widens.

### Honest limits

On the `feature/global` branch, `pounce-global` is a complete, correct
*continuous* global solver. It is not yet at commercial-solver scale (and, as
noted, not yet wired into a shipped release):

- **Continuous only** — no integer branching (MINLP).
- **Branching** offers widest, most-violation (default), and reliability
  (pseudocost + strong branching) rules; with OBBT every node the rule is
  usually second-order here, so it is a tunable knob rather than a fixed win.
- Atoms outside the supported set, `sin`/`cos` over a box spanning more than a
  few full periods, and division by an interval straddling zero fall back to the
  (valid but weak) interval box bound, which branching sharpens. (`sin`/`cos`
  over a box wider than π but within a few periods now gets a valid sloped
  relaxation rather than the bare box.)

For the classes it does cover, the answer is global and certified.
