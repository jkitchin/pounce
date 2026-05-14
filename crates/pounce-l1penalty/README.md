# pounce-l1penalty

Thierry-Biegler (2020) ℓ₁-exact penalty-barrier TNLP wrapper for POUNCE.

Internal crate. Wraps a user [`TNLP`](../pounce-nlp) so that the IPM
solves the augmented problem

```
min   f(x) + ρ · 1ᵀ(p + n)
s.t.  c(x) − p + n = g_target,
      x_L ≤ x ≤ x_U,   p ≥ 0,   n ≥ 0
```

instead of the original. The augmented NLP automatically satisfies LICQ
on the slack variables `(p, n)`, which is the property that makes the
standard interior-point machinery (filter LS, inertia correction,
fraction-to-boundary) work on degenerate / MPCC-like cases that the
stock filter line search thrashes on.

## Status

- **Phase 1 (this crate today):** TNLP wrapper with fixed `ρ`,
  default-off behind `SolverOptions::l1_exact_penalty_barrier`. Solution
  back-projection (truncate `x`, recompute `f(x*)` and `c(x*)`) lives
  here; multiplier mapping refines in Phase 2.
- **Phase 2:** algorithm-side wiring tightens, including correct
  multiplier reporting.
- **Phase 3:** Byrd-Nocedal-Waltz dynamic ρ + honest infeasibility
  upgrade.
- **Phase 3.5:** opt-in auto-fallback on `Restoration_Failed` /
  `Infeasible_Problem_Detected` / `Solved_To_Acceptable_Level` /
  `Maximum_Iterations_Exceeded` / `Not_Enough_Degrees_Of_Freedom`.
- **Phase 4:** MPCC paper reproduction (`benchmarks/mpcc/`).

See [pounce#10](https://github.com/jkitchin/pounce/issues/10) for the
full plan.

## Algorithmic reference

Thierry, D. & Biegler, L.T. (2020). *"The ℓ₁ Exact Penalty-Barrier
Phase for Degenerate Nonlinear Programming Problems in Ipopt"*,
IFAC-PapersOnLine.

ripopt 0.8.0's `src/l1_penalty_barrier_nlp.rs` (commit `7847bba9`) is
the canonical port source.

## License

EPL-2.0.
