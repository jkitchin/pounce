# Auxiliary-Equality Preprocessing

POUNCE's auxiliary-equality preprocessing pass identifies small,
self-contained equality sub-systems in an NLP and solves them
*before* the IPM starts. Variables determined by those sub-systems
are pinned to their values; the equality rows are dropped from the
problem the IPM sees. The IPM then handles the reduced problem,
which is smaller, often better-conditioned, and sometimes solvable
in zero iterations.

The pass is a port of [ripopt PR #32][ripopt32] by
[David Bernal Neira][bernalde] to pounce's TNLP wrapper. It lives
entirely inside `pounce-presolve` and is enabled by setting two
options:

```sh
pounce problem.nl presolve=yes presolve_auxiliary=yes
```

[ripopt32]: https://github.com/jkitchin/ripopt/pull/32
[bernalde]: https://github.com/bernalde

## What it does, in words

For each call to the inner TNLP, the wrapper:

1. **Builds a bipartite graph** between equality constraint rows and
   variables, using the Jacobian sparsity.
2. **Finds a maximum matching** (Hopcroft-Karp).
3. **Runs a Dulmage-Mendelsohn partition**, slicing the graph into
   three pieces: overdetermined, underdetermined, and *square* (the
   piece where rows and variables pair up one-to-one).
4. **Decomposes the square piece** into independent connected
   components, and each component into an ordered sequence of
   *blocks* via Tarjan SCC.
5. **Classifies** each block by how it's coupled to the rest of the
   problem: pure equality, objective-coupled, inequality-coupled, or
   both.
6. **Solves** each pure-equality (or, with `aggressive` coupling,
   objective-coupled) block via a small dense-LU Newton step and
   verifies the full-space residual is within tolerance.
7. **Applies** accepted blocks by clamping the fixed variables'
   bounds (`x_l = x_u = value`) and dropping the dropped rows.
8. After the IPM finishes, **recovers** the Lagrange multipliers for
   the dropped rows via a small dense-LU stationarity solve, and
   hands the user back a complete full-space KKT solution.

If the model has no eliminable structure, the pass is a tested no-op
and the IPM runs as usual.

## When it helps

The pass is most valuable when an NLP contains:

- Algebraic auxiliary variables that appear in one or two linear
  constraints with no other coupling (common in process-engineering
  and energy-system models).
- Internal chains where one variable is defined as a function of
  another (e.g. `T_out = T_in + delta_T` with `T_in` already known).
- Mass-balance equalities that form a small square block on a subset
  of stream variables.

ripopt reports `gaslib11_steady` going from 204 / 200 vars / cons to
140 / 136 vars / cons under this pass, and `tutorial_flow_density`
going from 6–7 IPM iterations to 0.

## Coupling classes

Every candidate block is classified by what it touches:

| Class                                | Touches inequality? | Touches objective grad? | Eliminated under `safe`? | Eliminated under `aggressive`? |
|--------------------------------------|---------------------|--------------------------|--------------------------|--------------------------------|
| `PureEquality`                       | no                  | no                       | yes                      | yes                            |
| `ObjectiveCoupled`                   | no                  | yes                      | no                       | yes (postsolve candidate)      |
| `InequalityCoupled`                  | yes                 | no                       | no                       | no                             |
| `ObjectiveAndInequalityCoupled`      | yes                 | yes                      | no                       | no                             |

`safe` is the default. Inequality-coupled blocks are never
eliminated in v1 — fixing such a variable could violate the
inequality.

## Options

See [Solver Options → NLP Presolve](options.md#nlp-presolve) for the
full list. The two switches you most often touch are:

| Option                          | Default | Effect                                  |
|---------------------------------|---------|-----------------------------------------|
| `presolve_auxiliary`            | `no`    | Master switch. Off → pass is a no-op.   |
| `presolve_auxiliary_coupling`   | `safe`  | `none` / `safe` / `aggressive` policy.  |

## Diagnostics

The pass populates an
[`AuxiliaryPreprocessingDiagnostics`](https://docs.rs/pounce-presolve)
struct on every call. From Rust:

```rust
use pounce_presolve::{wrap_with_presolve, PresolveOptions};

let opts = PresolveOptions { enabled: true, auxiliary: true, ..PresolveOptions::defaults() };
let wrapped = wrap_with_presolve(inner, opts)?;
// ... run a solve ...
// Access via the typed handle returned by PresolveTnlp::new:
//   let diag = typed.auxiliary_diagnostics();
//   println!("{diag}");
```

The `Display` impl produces output like:

```text
auxiliary-preprocessing: 1 of 1 candidate block(s) eliminated, fixing 2 variable(s) and dropping 2 row(s) in 0 ms
  max block dim: 2, max residual: 0.000e0
  coupling: pure=1, obj=0, ineq=0, both=0
```

Per-stage timings (`stage_time_ms.incidence_ms` /
`matching_ms` / `dm_ms` / `components_ms` / `btf_ms` /
`block_solve_ms` / `residual_check_ms`) and per-class accept counts
are also available.

From the command line, set `presolve_auxiliary_diagnostics=yes` to
have the same summary emitted to stderr automatically after every
Phase-0 pass:

```sh
pounce problem.nl presolve=yes presolve_auxiliary=yes \
  presolve_auxiliary_diagnostics=yes
```

## Limitations (v1)

Both linear and nonlinear blocks are eliminated. The linear path
reuses the pre-fetched Jacobian; the nonlinear path drives Newton
through TNLP callbacks.

Fixed variables are assumed to be interior to their original bounds
at the optimum; postsolve sets their bound multipliers to zero
implicitly. Lifting this assumption — handling the case where a
fixed variable is at an *original* bound — is a known follow-up.

The pass currently runs once, at the start of the solve. Iterative
re-elimination (running the pass again on the reduced problem) is
not supported in v1.

### Interaction with the rest of presolve

The auxiliary pass runs **before** the existing bound-tightening
phase (`presolve_bound_tightening=yes`). The two phases interact at
the bounds: aux clamps `x_l[i] = x_u[i] = value` for variables it
fixes; bound tightening then propagates the remaining constraints.
The orchestrator filters out aux-dropped rows before tightening
runs, so they can't propagate contradictions back over the clamps.
If `tighten_bounds` still flags infeasibility — for example because
an aux-fixed value disagrees with a kept-row's bound — the
orchestrator rolls back the aux pass for that solve and re-runs
tightening on the unfiltered rows. A one-line warning lands on
stderr when this happens.

### Interaction with sensitivity / reduced-Hessian post-processing

When the input `.nl` file carries sensitivity suffixes
(`sens_init_constr` / `sens_state_*`) or the CLI is invoked with
`--compute-reduced-hessian`, the entire presolve layer — including
auxiliary preprocessing — is **silently disabled**. The user sees a
single warning on stderr (`pounce: disabling presolve — ...`) and
the solve proceeds without any presolve transformation. This is
because the existing sensitivity / reduced-Hessian code paths
assume the IPM's variable and row indices match the user's
original `.nl`. Lifting this restriction is tracked separately
([pounce#19](https://github.com/jkitchin/pounce/issues/19)).

### Caveat: nonconvex problems can land at a different local optimum

When the auxiliary pass eliminates a block, it pins the block's
variables to a specific feasible point of the equality system —
the one Newton converges to from the probe point. On *convex*
problems this is the unique local optimum and the IPM would reach
the same values anyway. On *nonconvex* problems with multiple
feasible solutions to the equality system, the auxiliary pass may
fix variables to a feasible point in a different basin of
attraction than where the un-presolved IPM would eventually settle.
The full-space objective then differs between
`presolve_auxiliary=yes` and `presolve_auxiliary=no`, both
solutions remain feasible and locally optimal.

The vendored `gaslib11_steady.nl` benchmark in
`crates/pounce-cli/tests/fixtures/aux_presolve/` exhibits exactly
this — `presolve_auxiliary=yes` converges to objective ≈ 1.825e-02
while the un-presolved path settles at ≈ 3.286e-02. Both points
satisfy the model's KKT conditions; aux just lands in a different
basin. The regression test for `gaslib11_steady` deliberately does
NOT assert objective parity for this reason; the test name and
comments document the constraint.

If matching the un-presolved path's local optimum is important
for your workflow, leave `presolve_auxiliary=no` until iterative
re-elimination or a multiple-basin-aware policy lands (tracked on
[pounce#53](https://github.com/jkitchin/pounce/issues/53)).

## Worked example

Run any of these to see the pipeline in action:

```sh
cargo run -p pounce-presolve --example pipeline_demo
cargo run -p pounce-presolve --example phase0_via_tnlp
```

The first runs the algorithmic pipeline directly on a hand-crafted
problem and prints each stage's output. The second wraps a real
TNLP with `presolve_auxiliary=yes` and exercises the end-to-end
elimination + multiplier recovery.

## References

- Issue tracking the port: [pounce#53][pounce53].
- Upstream: [ripopt PR #32][ripopt32] by David Bernal Neira
  ([@bernalde][bernalde]). The
  `tutorial_flow_density{,_perturbed}.nl` and `gaslib11_steady.nl`
  fixtures vendored into `crates/pounce-cli/tests/fixtures/aux_presolve/`
  originate from that ripopt PR.
- Design notes: `dev-notes/auxiliary-equality-preprocessing.md` in
  the pounce repo.

[pounce53]: https://github.com/jkitchin/pounce/issues/53
