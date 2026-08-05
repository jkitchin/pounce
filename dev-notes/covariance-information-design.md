# Covariance and information: design notes

This document records the design of pyomo-pounce's parameter
uncertainty subsystem as it stands: `covariance()` and
`information()`, the declaration surface that feeds them, the `wrt=`
block selection both accessors take, and `retain_kkt()`. Everything
is computed from ONE ordinary solve: the solver factorizes the KKT
matrix to solve the NLP, the subsystem keeps that factorization, and
every question below is a backsolve or a Hessian product against
it; nothing here uses a second solve, finite differencing, or a
perturbed re-solve.

User-facing documentation lives in `docs/src/sensitivity.md`; demo
notebooks 31 and 32 demonstrate the whole surface. This file records
the constructions and why they are what they are.

Throughout, `Σ_i = z_i/s_i` summed over whichever bound sides exist
is the barrier diagonal, $H$ the Lagrangian Hessian, $W = H + \Sigma$
the factor's barrier-augmented block, $F$ the free members of the
block, $A$ the pinned ones, and
$S = H_{AA} - H_{AF} H_{FF}^{-1} H_{FA}$ the reduction onto $A$.

## The declaration surface

- `declare_fitted(vars)`: free variables become the fitted
  parameters, the default block.
- `declare_residual(container, group=)`: indexed variables holding
  the residuals, one member per data point; residual count and SSR
  are derived, never passed. Groups partition residuals into noise
  populations; distinct groups get their own variances and switch
  the covariance to the heteroscedastic sandwich.
- `declare_sens_param(params)`: pinned inputs for
  `gradient()`/`estimate()`; shares the session, out of scope here.
- `retain_kkt(model)`: keep the factorization with nothing declared,
  for problems where every question arrives as an explicit `wrt=`.
- `release_kkt(model)`: drop the held factorization now, freeing its
  memory; declarations and the retain flag still apply to the next
  solve.

| setup | factor kept | `covariance(model)` | `covariance(model, wrt=T)` |
|---|---|---|---|
| nothing | no | error | error |
| `declare_fitted(S)` | yes | over S | over T |
| `retain_kkt()` only | yes | error, no default | over T |
| `retain_kkt()` + `declare_fitted(S)` | yes | over S | over T |

One more rule completes retention: a `Covariance` or `Information` result whose
`conditioned_on` has not been read keeps the session, and with it the
factor, alive until first access, as does a live `Gradient` (it reads
the factor on every lookup); `release_kkt` drops the model's hold, not
an outstanding result's.

Explicit call-time forms (`solve(m, fitted=..., residuals=...)`) are
deliberately solve-local, so repeated solves of one model do not
accumulate declarations. The registry is deepcopy-aware:
`model.clone()` carries declarations and the retain flag; the
session, which holds handles into one converged factorization, stays
behind.

## Two spaces of rows

- FULL-X: `.col` order over all the model's variables. Everything the
  session hands around, and everything the activity report and
  `row_normal` speak, is full-x.
- VAR-X: the factor's x block. A variable with equal bounds is
  removed from the solve (`fixed_variable_treatment =
  make_parameter`) and every later column moves up.

The spaces coincide exactly when no variable is fixed, which makes
confusing them invisible until it silently reads a NEIGHBORING
variable's numbers. `Solver.primal_rows` maps full-x to factor rows
(`None` for removed variables, on which the accessors raise an
error naming the variable); `session.scatter_x` maps factor-space vectors
back. Every factor index routes through this pair, and the two
spaces are kept in separately named variables (`rows` vs `krows`)
wherever both appear.

## The block and its covariance

`wrt=` selects the block; omitted, it is the declared fitted set and
the accessors behave exactly as before `wrt=` existed. `wrt=`
accepts the following forms, normalized to an ordered list; a
repeated member is an error, since a duplicated coordinate makes the
block singular by construction:

- a Var, scalar or indexed (every member);
- an indexed slice (`m.x[2, :]`);
- a `(Var, iterable)` pair; a tuple of two Vars is a two-member
  block, not a pair;
- data objects, or a list mixing any of the above.

Unit backsolves at the block's factor rows give the K-inverse
columns, and

    M[i, j] = (K^-1)[krows[i], krows[j]]

symmetrized, is the block of the inverse KKT matrix, in natural
units by the `kkt_solve` contract (the backsolver unscales when NLP
scaling was active), so everything built on M composes without scale
factors. The homoscedastic Lagrangian covariance is `2 sigma^2 M`
with the dispositions below; the 2 comes from the objective being a
plain sum of squares. The K-inverse block IS the marginal
covariance: everything outside the block adjusts rather than being
held fixed, and each call re-reduces onto its own argument.

Sigma is a property of the fit, never of the block:

- declared residuals: `SSR / (n - n_fit)` per group;
- `sigma_sq=`: a float, or a per-group dict, when known;
- `n_data=`: SSR taken from the solve-time objective value, recorded
  at the solve so later writes into the model cannot silently
  rescale the covariance.

Both ESTIMATION routes divide by `n - n_fit`, so under `retain_kkt()`
alone there is no fit to take degrees of freedom from: with nothing
declared fitted they raise rather than divide by `n`, which would
bias every variance low by `n/(n - p)`. Retain-only therefore needs
`sigma_sq=`.

A sub-block's marginal therefore equals the corresponding entries of
the default answer exactly.

## Gauss-Newton

`hessian="gauss-newton"` replaces the exact reduced Hessian with the
expected information `2 J^T J`. J is recovered exactly from the same
backsolves by an identity: the residual rows of the K-inverse
columns are `J M`, so

    Z_r inv(M) = J

and the factor's barrier weight cancels regardless of its size. The
product is formed over the whole block and sliced afterwards, so
pinned members still have rows when S is assembled. Gauss-Newton is
structurally positive semidefinite, which MHE arrival costs and
consumers wanting scipy's `curve_fit` numbers need; the Lagrangian default is the observed
information, exact at the solution. They agree for linear models and
in the small-residual limit, and differ by O(residual x curvature)
otherwise. The heteroscedastic sandwich uses the same recovered
per-group Jacobians in both modes.

## Tangent recovery: the exact reduced Hessian

`information()` returns the reduced Hessian over the block, natural
units, no sigma anywhere, so for a homoscedastic fit covariance
equals `2 sigma^2 inv(information)` on the free block. The x blocks
of the K-inverse columns are `T M`: each column satisfies the
linearized equalities and has the block as its own coordinates, so

    T = Zx inv(M),    R = T^T H T,

with H applied through `Solver.hessian_vec` (exact Lagrangian
Hessian times a user-space vector, natural units, one product per
block column; factor-space tangents scatter to full-x first). The
barrier weight cancels multiplicatively inside the recovery instead
of being subtracted off. Measurement puts two limits on its accuracy:

- machine-exact for equality and variable-bound activity, including
  pinned variables at `Sigma/q ~ 3e10` where a subtraction loses ten
  digits;
- a binding INEQUALITY row couples through its slack barrier with a
  large but finite weight, leaving roughly 1e-6 relative residue at
  practical mu and degrading as mu tightens (the pinned combination
  drives M toward singularity).

The recovery requires the square estimation structure, checked as
`n_var - n_eq == n_params` before any tangent is formed. The
Lagrangian form routes over block shapes as follows:

| block | construction |
|---|---|
| parameterizes the constraint manifold (size = degrees of freedom) | direct tangent, `R = T^T H T` |
| proper sub-block of a square fitted set | Schur complement of the exact tangent R over the fitted block: free outside members profiled out, pinned ones conditioned on; no covariance inverted, a pinned member costs no digits |
| other within-count EXPLICIT block | corrected reduction off the factor; benign for free coordinates, whose slice carries no barrier term |
| default fitted block outside the square structure | raises an error suggesting `hessian="gauss-newton"`, which does not need the structure |
| rank-deficient | no information matrix exists; raises an error pointing to `covariance()` |

Fitted-level binding rows decline the Schur route with a warning:
their projection does not compose simply with marginalization.

## Activity classification

The Rust core classifies every bounded variable and inequality row
of the converged solve, and both accessors take membership from it:

```
classify(i):                          # a bounded variable, or an inequality row
    Σ = z/s, summed over whichever sides exist
    q = |H_ii|                                  variable: curvature in that coordinate
        |∇gᵢᵀ H ∇gᵢ| / ‖∇gᵢ‖⁴                   row: see below

    if q < sqrt(eps_machine) * max(1, max_j |H_jj|):
                                      return unidentified, sign of q's value
    r = Σ / q

    if μ > 1e-4:                      # the μ-edges thin toward the band
                                      # (they meet it at μ = 1e-2); only
                                      # the two clear calls are made
        return inactive         if r < 1e-1
        return strongly active  if r > 1e1
        return ambiguous

    return inactive         if r < √μ
    return strongly active  if r > 1/√μ
    return weakly active    if 1e-1 ≤ r ≤ 1e1
    return ambiguous                  # in a gap between the band and an edge
```

- The row denominator carries the fourth power so `r` is invariant to
  rescaling the row: `d → c·d` sends `Σ → Σ/c²` while curvature along
  the unit normal is unchanged, and `‖∇g‖⁴` restores the balance.
  This also absorbs the solver's per-row `d_scale`.
- The report is USER-SPACE indexed (a removed fixed variable reports
  `fixed` at its own index, an equality row `equality`) and exports
  `var_sigma`, `row_sigma`, `row_normal(j)`, and `hessian_vec(v)` in
  natural units; classification itself runs on the solver's scaled
  quantities, where the ratio is invariant.
- It requires `bound_relax_factor = 0` and checks it rather than
  documenting it. Two more conditions are checked on every call:
  `s·z` away from `μ` (off the central path, or a relaxed bound),
  and `Σ_i/|H_ii| > 100μ` on an inactive variable (contamination:
  barrier curvature surviving where none should). The contamination
  threshold is μ-relative because inactive means `r = O(μ)`; a fixed
  floor is structurally dead there.

Block members are classified at the REDUCED level, because a fitted
parameter in the residual-variable idiom (the misfit carried in
declared residual variables) has zero raw curvature: the
effective curvature is `q_red = |diag(inv(M)) - Σ|`, clamped to a
cancellation floor rather than rejected (a huge Σ cancelling inside
q_red would otherwise misfile a strongly active entry), and the same
edges make the call. Variables OUTSIDE the block are classified per
candidate as a singleton block by the identical rule, one backsolve
giving `(K^-1)_ii`, behind a cheap `Σ > √μ` prefilter so only
near-bound variables pay; this is scale-invariant where any absolute
Σ threshold is not.

## Dispositions

The table gives what each accessor returns for a block member,
given the classification. Both return a matrix over the whole block; the
columns are the row member $i$ gets in each:

| status | `s` | `z` | `Σ` as `μ → 0` | $i$ in | `covariance()` row | `information()` row |
|---|---|---|---|---|---|---|
| inactive | `O(1)` | `→ 0` | `μ/s² → 0` | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |
| strongly active | `→ 0` | `O(1)` | `z²/μ → ∞` | $A$ | $0$ | $S_{iA}$ |
| weakly active | `→ 0` | `→ 0` | finite, `O(1)` | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |
| ambiguous | n/a | n/a | ratio in a band gap | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |
| unidentified | n/a | n/a | curvature below scale | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |

The `s` and `z` columns say what each regime looks like, not how it
is detected: weak activity is the case where both vanish together,
and classification runs on `Σ/q` rather than on either alone. The
remaining dispositions do not fit a table row:

- A weakly active member is kept free at its TRUE variance: its own
  barrier weight is subtracted from the reduced block, so it reports
  the curvature q rather than the factor's 2q.
- Cross blocks between $F$ and $A$ are zero; S is conditional on the
  rest of the pinned set. Zero variance is not zero information,
  which is why the two accessors' pinned rows differ.
- A strongly active general row whose normal is supported on the
  block pins a DIRECTION: the free block is reduced on the null
  space of the binding normals and pushed back, singular by the
  number of binding rows, identically in both accessors; the
  projection annihilates the row's barrier weight exactly. The
  conditional information along the pinned combination is reported
  in the warning (tangent route inside the square structure, factor
  subtraction outside it).
- A row whose support leaves the block cannot be represented by a
  restricted normal: it is warned about and left unprojected, with
  the raw classification as its reported status.
- An indefinite Lagrangian information block is returned as computed
  with a warning naming Gauss-Newton: raising an error would
  withhold the
  finding that the point is not a minimum or the model is
  over-parameterized.
- Diagnostics name what they touch: fitted parameters on the default
  block, block members and variables outside the block under an
  explicit `wrt=`, every message prefixed by the accessor.

## Rank and singularity guards

Whether LAPACK raises on a structurally singular system is
build-dependent, so no structural condition is guarded by catching
`LinAlgError` alone:

- Explicit blocks are gated by count (more coordinates than the fit
  has degrees of freedom) and then by a rank test on M
  (dependence detectable in floating point; a duplicated design
  point is the canonical case), each path with its own message.
- A rank-deficient block is the trajectory-band case: `covariance()`
  returns the homoscedastic Lagrangian marginal `2 sigma^2 M`, the
  confidence band on the fitted trajectory (observation noise added
  on top makes it a prediction band), with membership handling
  bypassed; `information()`,
  per-group noise, and Gauss-Newton raise, since the latter two
  profile Jacobians through `inv(M)`.
- The singular free block inside the S computation is rank-gated the
  same way.
- `_SingularBlock`, a dedicated exception from the shared inversion
  helper, is the last resort behind all of these, so no rescue path
  does control flow on message text.
- A held factor carrying inertia-correction perturbations warns that
  the answer is regularized rather than exact: the isotropic delta_w
  lands on the free block and survives projection.

## The reporting surface

- `Covariance`: matrix keyed by the block's data objects (either
  index order), `std_err`, `correlation` (exactly-zero-variance
  entries report correlation 0), `sigma_sq` as used, `eigen()`.
- `Information`: matrix, `params`, `eigen()` with eigenvalues
  ascending; a near-zero eigenvalue is a direction the data does not
  determine and its eigenvector names the combination.
- Both: `conditioned_on`, the strongly active variables OUTSIDE the
  block. Their barrier weight stays in the held factor and drives
  the coupling through them to zero as mu falls, so the block's
  numbers are conditional on those bounds, not marginal over them.
  Computed on first access and cached; until then the pending
  computation keeps the session alive.

## Validation

- Suites anchor to closed-form values, not the implementation:
  `2 X^T X` for the linear model's information, restricted least
  squares for the pinned dispositions, the hat matrix for the
  trajectory's confidence band, the inverse identity between the accessors.
- Load-bearing constructions are mutation-verified: a broken
  tangent, a slice-first Gauss-Newton, a block-sized degrees of
  freedom, and a re-widened tuple guard each fail a named test.
- Two fixture axes are always exercised because their absence has
  repeatedly hidden bugs: objective scaling engaged with `df != 1`
  asserted, and an inert fixed variable ahead of the block in `.col`
  order.
- The structural error paths are tested through fixtures that reach
  them
  deterministically (bit-identical coordinates, not
  near-singularity), so the tests do not inherit LAPACK's
  build-dependence.
