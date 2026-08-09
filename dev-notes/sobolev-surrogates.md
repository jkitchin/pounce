# Sobolev surrogates from POUNCE solves — thermodynamics, and how to train them

**Status: research note, nothing started.** Expands §7 of
`dev-notes/wgpu-batched-solve.md`: using POUNCE to manufacture training data
labelled with *both* solutions and sensitivities, the way an ML interatomic
potential trains on DFT energies *and* forces. Two halves — the equilibrium
application in detail (§1), and the training methodology that applies to all of
§7's applications (§2). They are not independent: the thermodynamics supplies
the sharpest available answer to the training question.

## 1. Thermodynamic equilibrium surrogates

### 1.1 The problem

Constrained Gibbs free-energy minimization:

```text
min_n   G(n, T, P) = Σ_i n_i μ_i(T, P, n)
s.t.    A n = b(z)        element balance (A = formula matrix)
        n ≥ 0
```

with `μ_i = μ_i°(T) + RT ln a_i`; `a_i = γ_i x_i` in a liquid (NRTL / UNIQUAC /
UNIFAC) or `φ_i y_i P/P°` in a vapour (Peng–Robinson, SRK). The surrogate learns
`(T, P, z) ↦ n*` and/or `G*`.

This is a nonconvex NLP with equalities, bounds, and `ln` terms — it exercises
`pounce-algorithm` and the sIPOPT path, not just the QP path.

### 1.2 Why this is the right target

**Volume.** Flash calculations sit in the innermost loop of every process
simulator. A 50-stage column inside a Newton loop inside a design optimization
is millions of flashes. It is *the* hot spot in process simulation, so
amortization has somewhere to pay off.

**The parameter space is small and bounded.** For a fixed component set,
`(T, P, z)` is ~3–10 dimensions with `z` on a simplex. Surrogates work in low
dimensions. Contrast molecular configuration space, where MLIPs need enormous
datasets and clever equivariance just to cover the domain.

**The derivatives are named physical quantities.** This is the feature that
makes it better than every other candidate in §7:

| sensitivity | is |
|---|---|
| `∂G*/∂T` | `−S`, entropy |
| `∂G*/∂P` | `V`, volume |
| `∂²G*/∂T²` | `−C_p/T`, heat capacity |
| `∂G*/∂b_k` | `λ_k`, the element potential (the element-balance multiplier) |
| `∂ ln K/∂(1/T)` | `−ΔH°/R`, van 't Hoff — reaction enthalpy |
| `∂n*/∂z` | K-value / distribution sensitivities |

So the "forces" can be validated against **independently measured data** —
calorimetry for `C_p`, PVT for `V`, tabulated `ΔH°` — not merely against the
solver that produced them. MLIP practitioners have no such channel; their forces
are only ever as good as the DFT. Here a systematic error in the labels is
detectable from outside the pipeline.

Note also that `λ_k` falls out as a *dual* variable. The element potentials are
exactly what the RAND-family equilibrium literature works in, so POUNCE's
multipliers are already the quantity the domain wants.

**Thermodynamic consistency is an exact, checkable constraint.** Gibbs–Duhem
(`Σ x_i d ln γ_i = 0` at fixed `T,P`) and the Maxwell relations are *identities*
that any true `G` satisfies. A surrogate that models the scalar `G*` and obtains
everything else by autodiff satisfies them **by construction**. A surrogate with
independent heads for `S`, `V`, `n*` will violate them. This is the domain
handing us a decisive answer to §2's question, and it is checkable numerically
on held-out points.

There is precedent: Rittig, Felton, Lapkin & Mitsos, *Gibbs–Duhem-informed
neural networks for binary activity coefficient prediction* (Digital Discovery,
2023; [arXiv:2306.07937](https://arxiv.org/abs/2306.07937)) put Gibbs–Duhem in
the loss as a regularizer and report better consistency *and* generalization.
Also HANNA (hard-constraint NN for consistent activity coefficients). The open
part: that work learns *properties* (`γ`) and still runs the flash iteratively.
Learning the **equilibrium solution map itself**, with solver-generated
sensitivities, is the gap.

### 1.3 Honest caveats

**Phase stability and multiple solutions.** Equilibrium flash is nonconvex with
trivial and spurious solutions; getting the right one requires tangent-plane /
Michelsen stability analysis. POUNCE solves nonconvex problems locally by
default, so the generator needs multi-start plus a stability check, and the
dataset needs the *stable* root, not whatever the solver found. This is real
work, not a detail.

**The solution map is genuinely discontinuous.** When a phase appears or
vanishes, `n*` jumps. This is the active-set-change pathology of §2.4 in its most
severe form (`n_i` hits zero, a phase leaves), and it is *worse* than the generic
case because the discontinuity is in the solution, not just its derivative. The
surrogate must be structured around it — classify the phase configuration first,
regress within it — which is also how the domain already thinks.

**A thermo package is a hard dependency.** `μ°(T)`, `γ`, `φ` have to come from
somewhere: Cantera (ideal-gas/reacting), CoolProp or `thermo` (EOS/activity).
That is a modelling commitment, and the surrogate inherits every bias of the
chosen model. The surrogate approximates *that model's* equilibrium, not
nature's — worth stating plainly in any paper.

**Cubic EOS root selection is discrete**, which injects another
non-smoothness independent of the optimization.

### 1.4 Staging

1. **Ideal-gas chemical equilibrium** — Cantera `equilibrate('TP')` as the
   reference. Well-behaved, near-convex, unambiguous solution. Proves the
   pipeline and the Sobolev machinery end to end.
2. **Single-phase nonideal** — activity model, still one phase, no stability
   problem. Introduces real nonlinearity.
3. **Multiphase VLE** — where the value is, where the discontinuities are, and
   where the phase-classification architecture earns its keep.

Do not start at 3.

## 2. Training on values *and* derivatives

### 2.1 Do not train them independently

The derivative head must **be** the derivative of the value head, obtained by
autodiff with respect to the *inputs*, not a second output head fit to
derivative labels. Loss:

```text
L = w_V ‖ V̂(p) − V(p) ‖²  +  w_D ‖ ∇_p V̂(p) − ∂V/∂p ‖²
```

where `∇_p V̂` is computed by backprop through the network. Training therefore
needs **double backprop** (a gradient of a loss that already contains a
gradient), which costs roughly 2–3× per step and more memory. That cost is the
price of the whole idea.

Independent heads look tempting — they are cheaper and often score *better* on
derivative error alone — and the MLIP field ran this experiment for us. Direct
force-prediction models (Orb-v2, direct Orb-v3, GemNet-dT) held state of the art
on OC20/OC22/Matbench-Discovery while being non-conservative, and then failed in
actual use: Bigi et al., *The dark side of the forces*
([arXiv:2412.11569](https://arxiv.org/abs/2412.11569), ICML 2025) catalogues
ill-defined geometry-optimization convergence and MD instability, and reported
energy drift figures put direct Orb-v3 at ~78 meV/ps against ~9 meV/ps for the
conservative variant of the same model. The field has swung back to conservative
models, with direct prediction retained mainly as a *pre-training* strategy.

The lesson transfers: a derivative head that is not actually a derivative gives
up the structural guarantees, and the structural guarantees are the reason to
have derivative labels at all.

### 2.2 Model the scalar; differentiate for everything else

The strong version of the design. For a parametric problem where `p` enters the
objective linearly,

```text
V(p) = min_x { f(x) + pᵀx  :  x ∈ C }
```

the envelope (Danskin) theorem gives `∇_p V(p) = x*(p)` exactly. So a **single
scalar network** `V̂` yields:

| quantity | from `V̂` |
|---|---|
| optimal value | `V̂(p)` |
| **solution** `x*(p)` | `∇_p V̂` — first derivative |
| **sensitivity** `∂x*/∂p` | `∇²_p V̂` — second derivative |

Three label types, one network, consistency free. And it enforces a property the
true solution map genuinely has: `∂x*/∂p` is a Hessian of `V`, hence
**symmetric** (and negative semidefinite, since `V` is concave as a min of
affine functions of `p`). Independent heads violate that symmetry; this
construction cannot.

The thermodynamic case is the same structure one level up: model `G*(T,P,b)`,
and `S`, `V`, `C_p`, and the element potentials `λ` all fall out by
differentiation — which is exactly why Gibbs–Duhem and the Maxwell relations
hold by construction (§1.2).

**Where it does not apply:** a general vector-valued solution map `x*(p)` is not
the gradient of any potential, so there is no conservativity constraint to
exploit and you are doing ordinary Jacobian matching. And for MPC the control
`u*(x₀)` is not `∇V` — you recover it by a short-horizon solve using `V̂` as
terminal cost (application (a) of the parent note), not by differentiating.
Check which case you are in before assuming the scalar trick is available.

### 2.3 Weighting

**Non-dimensionalize before weighting anything.** `V` and `∂V/∂p` carry
different physical units, so raw `w_V`/`w_D` are not comparable numbers. Z-score
each target block against dataset statistics and weight the normalized
residuals. Skipping this makes every subsequent tuning decision meaningless.

**Mind the counting.** One solve gives 1 value label and up to `n × n_p`
derivative labels. Weighting equally *per label* lets derivatives dominate by
orders of magnitude. That is usually right early — derivative labels carry local
shape information while the value is a single global number — but training on
derivatives alone determines the value only up to a constant, and the constant
matters whenever solutions are compared across parameter points.

**Anneal.** Standard MLIP practice, and it follows from the above: start
derivative-heavy to learn the shape, finish value-heavy to fix the offsets.

**Better: learn the weights.** Kendall & Gal-style homoscedastic uncertainty
weighting treats each term as a Gaussian likelihood with learned noise:

```text
L = Σ_j [ L_j / (2 σ_j²) + log σ_j ]
```

with `σ_j` free parameters. Principled, self-tuning, and it removes a
hyperparameter sweep that otherwise has to be repeated per problem family. This
is the recommended default over hand-tuned `w_V`/`w_D`.

### 2.4 Where the DFT analogy breaks — and it matters

DFT forces are smooth almost everywhere (the exception, conical intersections,
is rare and known). **`∂x*/∂p` is not.** It is piecewise smooth with jumps at
active-set changes, and at degenerate points — weakly active constraints, LICQ
failure, strict-complementarity failure — it may not exist at all. Training a
smooth network on labels that are genuinely discontinuous will produce smoothing
artefacts exactly at the boundaries, which is where control decisions flip.

POUNCE is unusually well placed to help here, because it *knows* when a solve is
near such a point and can say so:

- `crates/pounce-sensitivity/src/activity.rs` classifies every bounded variable
  and finite-bounded inequality row into one of five statuses from the ratio of
  barrier curvature to model curvature, and explicitly reports `unidentified`
  when the classification is not clean;
- the complementarity products `sᵢzᵢ` and the smallest active multiplier are
  available directly from the converged iterate;
- `pounce-sensitivity`'s boundcheck path already deals with steps that leave the
  box.

So the generator can **emit a degeneracy flag alongside every sample**. Three
usable responses: downweight or drop flagged samples; use Huber rather than L2
on the derivative term so boundary samples cannot dominate; or model the
piecewise structure explicitly (classify the active set, regress within region)
— which for the thermo case is the same architecture the phase-boundary problem
demands anyway (§1.3).

This is a genuine advantage of generating data with an instrumented solver
rather than a black box, and it should be designed in from the start rather than
discovered when the surrogate misbehaves near constraint boundaries.

### 2.5 Cost, and a fortunate alignment

Full Jacobian matching needs `n` VJPs (or `n_p` JVPs) per sample per step, and
storing `n × n_p` floats per sample. Both are avoidable by **random projection**:
match `vᵀ ∂x̂/∂p` against `vᵀ ∂x*/∂p` for a fresh random `v` each step. Unbiased,
one VJP per step, and storage drops to `k × n` for `k` sampled directions.

This lines up with POUNCE better than it has any right to.
`SensApplication::parametric_step(Δp, dx)` is **already directional** — it
applies `Bᵀ` to `Δp` and does one backsolve against the converged KKT factor
(`crates/pounce-sensitivity/src/sens_app.rs:211`). It computes a directional
derivative for a *given* `Δp`; it does not form a Jacobian. So:

- generating `k` random directional sensitivities costs `k` backsolves against a
  factorization you already have,
- the natural storage format is `k` directions and their responses,
- and that is precisely the format sketched Jacobian matching consumes.

The efficient generation format and the efficient training format are the same
object. Storing full Jacobians would be doing extra work to produce data the
training loop would then have to project back down.

## 3. What to build first

1. Ideal-gas equilibrium generator (Cantera reference), emitting
   `(T, P, z) → G*, n*, λ`, plus `k` random directional sensitivities and a
   per-sample degeneracy flag from `activity.rs`.
2. Scalar `Ĝ` network, everything else by autodiff, uncertainty-weighted loss.
3. Validate the *derivatives* against independent data — `C_p` from calorimetry,
   `V` from PVT — and check Gibbs–Duhem residuals on held-out points. If the
   derivative labels are right and the consistency identities hold to tolerance,
   the method is working; if not, no amount of value-fitting accuracy matters.

Only then decide whether generation throughput justifies §4's GPU kernel.

## References

- Rittig, J. G., Felton, K. C., Lapkin, A. A., Mitsos, A. *Gibbs–Duhem-informed
  neural networks for binary activity coefficient prediction.* Digital Discovery
  (2023). [arXiv:2306.07937](https://arxiv.org/abs/2306.07937).
- Bigi, F., et al. *The dark side of the forces: assessing non-conservative
  force models for atomistic machine learning.* ICML 2025.
  [arXiv:2412.11569](https://arxiv.org/abs/2412.11569).
- Czarnecki, W. M., et al. *Sobolev Training for Neural Networks.* NeurIPS 2017.
- Kendall, A., Gal, Y., Cipolla, R. *Multi-task learning using uncertainty to
  weigh losses.* CVPR 2018.
- Michelsen, M. L. *The isothermal flash problem.* Fluid Phase Equilibria (1982).
  (Phase stability / tangent plane.)
- Pirnay, H., López-Negrete, R., Biegler, L. T. *Optimal sensitivity based on
  IPOPT.* Math. Prog. Comp. 4(4) (2012). (Basis of `pounce-sensitivity`.)

## Related

- `dev-notes/wgpu-batched-solve.md` — parent note; §7 the use case, §9 packaging.
- [#561](https://github.com/jkitchin/pounce/issues/561) — the `pounce-rs` facade
  does not cover `pounce-sensitivity`, which this work depends on.
