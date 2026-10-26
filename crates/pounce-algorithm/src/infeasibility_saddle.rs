//! Testing a local-infeasibility verdict for a nearby point with less violation.
//!
//! # Why this exists
//!
//! `Infeasible_Problem_Detected` claims the solve stopped at a local minimizer
//! of the constraint violation. Every gate that issues it checks that claim to
//! first order only: the gradient of the violation vanishes, or a step along
//! its negative does not help (`IpoptCq::infeasibility_descent_available`).
//! A first-order test cannot tell a minimizer from a saddle or a maximizer,
//! and nonconvex models have saddles of the violation that the iterates are
//! attracted to.
//!
//! Vanderbei `discs` is the measured case. It places 12 discs subject to
//! tangency equalities `(x_i - x_j)^2 + (y_i - y_j)^2 = (r_i + r_j)^2`. A
//! solve that perturbs the trajectory (a different FERAL ordering, `mc64`
//! scaling, or a different symbolic analysis) can end with discs 8 and 9 at
//! the same centre. There the row `cons1[9,8]` reads `0 = 4`. Its gradient
//! with respect to the centres is exactly zero, and the radii sit on their
//! lower bound of 1, so every first-order test passes and the solve reports
//! infeasibility. Upstream Ipopt 3.14.19 with MUMPS reports the same verdict
//! at default options. But separating the two discs by `delta` lowers that
//! row's violation to `4 - |delta|^2`: the point is a maximizer of the
//! violation along every separating direction.
//!
//! # What is tested
//!
//! The max-norm violation `theta(x) = max_i dist(g_i(x), [g_l_i, g_u_i])`, at
//! seeded random points near the verdict's point and clipped into the
//! variable box. If one of them has less violation, beyond a margin that
//! evaluation noise cannot reach, the verdict's point is not a local
//! minimizer of `theta`.
//!
//! The norm is deliberate. At the `discs` point the ℓ₁ violation *is* locally
//! minimal: moving one disc breaks its satisfied tangency rows at first order,
//! and the sum of those exceeds the second-order gain on the coincident pair.
//! Under the max norm those first-order breaks stay below the violated row's
//! 4.0 for small steps, so the gain shows. A random probe can therefore
//! refute a degenerate row's minimality without knowing which row it is.
//!
//! # Direction
//!
//! The probe changes no verdict by itself. It only decides whether the
//! second-opinion ladder spends one more solve, started from the point the
//! probe found, and that solve is promoted only if it converges, which no
//! infeasible model can do. A genuinely infeasible model whose verdict point
//! is a strict local minimizer of `theta` produces no escape point and pays
//! nothing; one whose verdict point is a saddle pays one re-solve and keeps
//! its verdict. Any failure to evaluate declines to produce a point.

use pounce_common::types::Number;
use pounce_nlp::tnlp::{BoundsInfo, TNLP};
use std::cell::RefCell;
use std::rc::Rc;

/// The second-opinion rung that re-solves from [`SaddleEscape::x`].
pub const SADDLE_ESCAPE_RUNG_LABEL: &str = "restart_from_infeasibility_saddle";

/// Relative step sizes, largest first. Each variable moves by
/// `step * max(1, |x_j|) * u_j` with `u_j` uniform in `[-1, 1)`.
///
/// Largest first because the re-solve starts from the returned point, and a
/// point barely off the saddle has a violation gradient of the same small
/// size, which a fresh barrier start can walk straight back from. The smaller
/// steps are there for models whose first-order breaks of the satisfied rows
/// outgrow the second-order gain at the larger ones.
const STEPS: [Number; 3] = [1e-2, 1e-3, 1e-4];

/// Random directions per step, each probed with both signs.
const DIRECTIONS: usize = 8;

/// Doublings along the direction that found a lower violation, at most. Ten
/// reach `1024` probe steps, about `10x` the variable's own scale from the
/// largest step.
const EXTRAPOLATIONS: usize = 10;

/// A trial counts only if it lowers `theta` by this fraction of its value at
/// the verdict's point. Rounding moves a row evaluation by a few ulps of the
/// row's terms; `1e-9` sits far above that and far below the gain on the
/// measured case (`2.5e-5` of `theta` at the largest step on `discs`).
const MARGIN: Number = 1e-9;

/// Fixed seed: the probe has to reproduce on every platform and every run.
const SEED: u64 = 0x5AD0_1E5C_A9E0_0001;

/// A point near a local-infeasibility verdict's point with less max-norm
/// violation, showing the verdict's point is not a local minimizer of it.
#[derive(Debug, Clone)]
pub struct SaddleEscape {
    /// The point, in the user's variable order, inside the variable box.
    pub x: Vec<Number>,
    /// Max-norm violation at the verdict's point (clipped into the box).
    pub theta_at_verdict: Number,
    /// Max-norm violation at [`Self::x`].
    pub theta_escaped: Number,
    /// The relative step that found it, one of [`STEPS`].
    pub step: Number,
    /// How far [`Self::x`] lies along the probe's direction, in probe steps:
    /// `1` for the probe point itself, `2^k` after `k` accepted doublings.
    pub multiple: Number,
}

/// Probe near `x_verdict` for a point with less max-norm constraint
/// violation.
///
/// Returns `None` when the model has no constraints, when the verdict's
/// point has no measurable violation, when no probe lowers it beyond
/// [`MARGIN`], or when the model cannot be evaluated. `None` means "no escape
/// point found", never "the verdict is confirmed".
///
/// `lower_bound_inf` / `upper_bound_inf` are the solver's
/// `nlp_lower_bound_inf` / `nlp_upper_bound_inf`: a bound at or beyond them is
/// absent, both for clipping and for measuring a row's violation.
pub fn infeasibility_saddle_escape(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    x_verdict: &[Number],
    lower_bound_inf: Number,
    upper_bound_inf: Number,
) -> Option<SaddleEscape> {
    let info = tnlp.borrow_mut().get_nlp_info()?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    if n == 0 || m == 0 || x_verdict.len() != n {
        return None;
    }
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !tnlp.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return None;
    }
    let lo_present = |b: Number| b.is_finite() && b > lower_bound_inf;
    let hi_present = |b: Number| b.is_finite() && b < upper_bound_inf;
    for j in 0..n {
        // A crossed box is presolve's business, as in `infeasibility_refutation`.
        if lo_present(x_l[j]) && hi_present(x_u[j]) && x_l[j] > x_u[j] {
            return None;
        }
    }
    let clip = |j: usize, v: Number| -> Number {
        let mut v = v;
        if lo_present(x_l[j]) && v < x_l[j] {
            v = x_l[j];
        }
        if hi_present(x_u[j]) && v > x_u[j] {
            v = x_u[j];
        }
        v
    };

    let mut g = vec![0.0; m];
    let mut theta = |x: &[Number]| -> Option<Number> {
        if !tnlp.borrow_mut().eval_g(x, true, &mut g) {
            return None;
        }
        let mut worst: Number = 0.0;
        for i in 0..m {
            let v = g[i];
            if !v.is_finite() {
                return None;
            }
            let below = if lo_present(g_l[i]) { g_l[i] - v } else { 0.0 };
            let above = if hi_present(g_u[i]) { v - g_u[i] } else { 0.0 };
            worst = worst.max(below).max(above);
        }
        Some(worst)
    };

    // The reported point may sit a rounding distance outside the box
    // (`honor_original_bounds` projects, the IPM's own iterate does not), and
    // every trial below is clipped, so the reference is clipped too: otherwise
    // the clip alone could read as a gain.
    if x_verdict.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let base: Vec<Number> = (0..n).map(|j| clip(j, x_verdict[j])).collect();
    let theta0 = theta(&base)?;
    if !(theta0 > 0.0) {
        return None;
    }
    let threshold = theta0 * (1.0 - MARGIN);

    let mut state = SEED;
    let mut u = vec![0.0; n];
    let mut trial = vec![0.0; n];
    for &step in &STEPS {
        let mut best: Option<(Number, Vec<Number>)> = None;
        for _ in 0..DIRECTIONS {
            // A variable within one step of a present bound stays put, so
            // the probe moves on the face of the box the point sits on and
            // never clips. Clipping would break the symmetry the two-sided
            // test relies on: on `discs` the radii sit on their lower bound,
            // a step that grows one worsens the violated row at first order,
            // and the step that would shrink it is clipped to nothing. A
            // minimizer over the box is a minimizer over that face, so
            // fixing these variables cannot manufacture a decrease.
            for (j, uj) in u.iter_mut().enumerate() {
                let draw = unit_symmetric(&mut state);
                let h = step * base[j].abs().max(1.0);
                let near_lo = lo_present(x_l[j]) && base[j] - x_l[j] < h;
                let near_hi = hi_present(x_u[j]) && x_u[j] - base[j] < h;
                *uj = if near_lo || near_hi { 0.0 } else { draw };
            }
            // Both signs must lower the violation. A first-order decrease
            // along `+u` is an increase along `-u`, so this rejects what a
            // one-sided test cannot: a verdict point that is a minimizer of
            // the solver's own measure (ℓ₁ or ℓ₂) but not of the max norm,
            // and one converged only to tolerance. A decrease even in the
            // step is curvature, which is the claim being refuted.
            let mut pair: [Option<(Number, Vec<Number>)>; 2] = [None, None];
            for (k, sign) in [1.0, -1.0].into_iter().enumerate() {
                for j in 0..n {
                    let h = step * base[j].abs().max(1.0);
                    trial[j] = clip(j, base[j] + sign * h * u[j]);
                }
                match theta(&trial) {
                    Some(t) if t < threshold => pair[k] = Some((t, trial.clone())),
                    _ => break,
                }
            }
            let [Some(plus), Some(minus)] = pair else {
                continue;
            };
            let lower = if plus.0 <= minus.0 { plus } else { minus };
            if best.as_ref().is_none_or(|(b, _)| lower.0 < *b) {
                best = Some(lower);
            }
        }
        if let Some((mut t, mut x)) = best {
            // Carry on along the direction that found it, doubling while the
            // violation keeps falling. A point one probe step off the saddle
            // has a violation gradient of the same small size, and a fresh
            // barrier start walks straight back from it: measured on `discs`,
            // a restart `1e-2` off the coincident pair re-collapsed it and
            // ended infeasible again in 31 iterations.
            let dir: Vec<Number> = (0..n).map(|j| x[j] - base[j]).collect();
            let mut multiple = 1.0;
            for _ in 0..EXTRAPOLATIONS {
                let next = 2.0 * multiple;
                for j in 0..n {
                    trial[j] = clip(j, base[j] + next * dir[j]);
                }
                match theta(&trial) {
                    Some(tt) if tt < t => {
                        t = tt;
                        x.copy_from_slice(&trial);
                        multiple = next;
                    }
                    _ => break,
                }
            }
            return Some(SaddleEscape {
                x,
                theta_at_verdict: theta0,
                theta_escaped: t,
                step,
                multiple,
            });
        }
    }
    None
}

/// SplitMix64, as in `pounce_nlp::start_conditioner`: seedable and identical
/// on every platform.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[-1, 1)` from 53 random bits.
fn unit_symmetric(state: &mut u64) -> Number {
    let bits = splitmix64(state) >> 11;
    (bits as Number) / ((1u64 << 53) as Number) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_nlp::tnlp::{
        IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    };

    const LO_INF: Number = -1e19;
    const UP_INF: Number = 1e19;

    /// Rows given as closures of `x`, with bounds.
    struct Rows {
        lo: Vec<Number>,
        hi: Vec<Number>,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
        g: fn(&[Number], &mut [Number]),
        /// Multiplies every row value; bounds are scaled by the caller.
        scale: Number,
    }

    impl Rows {
        fn scaled(mut self, s: Number) -> Self {
            self.scale = s;
            self
        }
    }

    impl TNLP for Rows {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: self.lo.len() as i32,
                m: self.g_l.len() as i32,
                nnz_jac_g: 0,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&self.lo);
            b.x_u.copy_from_slice(&self.hi);
            b.g_l.copy_from_slice(&self.g_l);
            b.g_u.copy_from_slice(&self.g_u);
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            false
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f.fill(0.0);
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            (self.g)(x, g);
            for v in g.iter_mut() {
                *v *= self.scale;
            }
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
    }

    fn probe(t: Rows, x: &[Number]) -> Option<SaddleEscape> {
        let rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(t));
        infeasibility_saddle_escape(&rc, x, LO_INF, UP_INF)
    }

    /// Two unit discs on a line that must touch, `(a - b)^2 = 4`, stopped
    /// with both centres at 0: the `discs` trap in one dimension. Separating
    /// them lowers the violation, so the probe must find a point.
    #[test]
    fn coincident_tangent_discs_are_a_saddle_of_the_violation() {
        let rows = Rows {
            lo: vec![LO_INF, LO_INF],
            hi: vec![UP_INF, UP_INF],
            g_l: vec![4.0],
            g_u: vec![4.0],
            scale: 1.0,
            g: |x, g| g[0] = (x[0] - x[1]).powi(2),
        };
        let e = probe(rows, &[0.0, 0.0]).expect("separating the discs lowers the violation");
        assert_eq!(e.theta_at_verdict, 4.0);
        assert!(e.theta_escaped < 4.0);
        assert_eq!(e.step, STEPS[0]);
    }

    /// The same trap with a satisfied tangency beside it, which every move of
    /// `a` breaks at first order. The ℓ₁ violation is locally minimal here;
    /// the max norm is not, and that is the one the probe uses.
    #[test]
    fn a_satisfied_row_broken_at_first_order_does_not_hide_the_saddle() {
        let rows = Rows {
            lo: vec![LO_INF; 3],
            hi: vec![UP_INF; 3],
            g_l: vec![4.0, 4.0],
            g_u: vec![4.0, 4.0],
            // a = x0, b = x1 coincide; c = x2 sits tangent to a.
            scale: 1.0,
            g: |x, g| {
                g[0] = (x[0] - x[1]).powi(2);
                g[1] = (x[0] - x[2]).powi(2);
            },
        };
        assert!(probe(rows, &[0.0, 0.0, 2.0]).is_some());
    }

    /// `x^2 + 1 = 0` at `x = 0` is a strict local minimizer of the violation
    /// of a genuinely infeasible model. No probe can lower it.
    #[test]
    fn a_strict_minimizer_of_the_violation_produces_no_point() {
        let rows = Rows {
            lo: vec![LO_INF],
            hi: vec![UP_INF],
            g_l: vec![0.0],
            g_u: vec![0.0],
            scale: 1.0,
            g: |x, g| g[0] = x[0] * x[0] + 1.0,
        };
        assert!(probe(rows, &[0.0]).is_none());
    }

    /// `x >= 0.7` over `x ∈ [0, 0.6]`, stopped at the upper bound: the only
    /// descent leaves the box, and clipping must not count it.
    #[test]
    fn descent_that_only_exists_outside_the_box_is_not_counted() {
        let rows = Rows {
            lo: vec![0.0],
            hi: vec![0.6],
            g_l: vec![0.7],
            g_u: vec![UP_INF],
            scale: 1.0,
            g: |x, g| g[0] = x[0],
        };
        assert!(probe(rows, &[0.6]).is_none());
    }

    /// The `discs` shape: the tangency row also depends on two radii pinned
    /// at their lower bound, and growing either worsens the row at first
    /// order. The probe must move on the face the point sits on.
    #[test]
    fn variables_on_a_bound_stay_put_so_the_saddle_still_shows() {
        let rows = Rows {
            lo: vec![LO_INF, LO_INF, 1.0, 1.0],
            hi: vec![UP_INF; 4],
            g_l: vec![0.0],
            g_u: vec![0.0],
            scale: 1.0,
            // (a - b)^2 - (r1 + r2)^2 = 0 at a = b = 0, r1 = r2 = 1.
            g: |x, g| g[0] = (x[0] - x[1]).powi(2) - (x[2] + x[3]).powi(2),
        };
        let e = probe(rows, &[0.0, 0.0, 1.0, 1.0]).expect("the centres can separate");
        assert_eq!(&e.x[2..], &[1.0, 1.0], "the radii stay on their bound");
    }

    /// Two contradictory rows of different scale, `x = 0` and `2x = 2`. The
    /// ℓ₁ violation is minimized at `x = 1`, where the max norm still falls
    /// to the left at first order — and rises to the right. That is a norm
    /// mismatch, not a saddle, and the two-sided test must not take it.
    #[test]
    fn a_first_order_decrease_on_one_side_only_is_not_a_saddle() {
        let rows = Rows {
            lo: vec![LO_INF],
            hi: vec![UP_INF],
            g_l: vec![0.0, 2.0],
            g_u: vec![0.0, 2.0],
            scale: 1.0,
            g: |x, g| {
                g[0] = x[0];
                g[1] = 2.0 * x[0];
            },
        };
        assert!(probe(rows, &[1.0]).is_none());
    }

    /// A point with no violation is not an infeasibility verdict's point.
    #[test]
    fn a_feasible_point_produces_no_point() {
        let rows = Rows {
            lo: vec![LO_INF, LO_INF],
            hi: vec![UP_INF, UP_INF],
            g_l: vec![4.0],
            g_u: vec![4.0],
            scale: 1.0,
            g: |x, g| g[0] = (x[0] - x[1]).powi(2),
        };
        assert!(probe(rows, &[0.0, 2.0]).is_none());
    }

    /// The probe is seeded and must give the same point on every call.
    #[test]
    fn the_probe_is_deterministic() {
        let mk = || Rows {
            lo: vec![LO_INF, LO_INF],
            hi: vec![UP_INF, UP_INF],
            g_l: vec![4.0],
            g_u: vec![4.0],
            scale: 1.0,
            g: |x, g| g[0] = (x[0] - x[1]).powi(2),
        };
        let a = probe(mk(), &[0.0, 0.0]).unwrap();
        let b = probe(mk(), &[0.0, 0.0]).unwrap();
        assert_eq!(a.x, b.x);
    }

    /// Row scaling cancels out of the ratio: the trap is found, and the
    /// strict minimizer is not, at every row scaling.
    #[test]
    fn the_verdict_does_not_depend_on_row_scaling() {
        for k in -8..=8 {
            let s = 10f64.powi(k);
            let saddle = Rows {
                lo: vec![LO_INF, LO_INF],
                hi: vec![UP_INF, UP_INF],
                g_l: vec![4.0 * s],
                g_u: vec![4.0 * s],
                scale: 1.0,
                g: |x, g| g[0] = (x[0] - x[1]).powi(2),
            }
            .scaled(s);
            assert!(
                probe(saddle, &[0.0, 0.0]).is_some(),
                "saddle at scale 10^{k}"
            );
            let minimizer = Rows {
                lo: vec![LO_INF],
                hi: vec![UP_INF],
                g_l: vec![0.0],
                g_u: vec![0.0],
                scale: 1.0,
                g: |x, g| g[0] = x[0] * x[0] + 1.0,
            }
            .scaled(s);
            assert!(
                probe(minimizer, &[0.0]).is_none(),
                "minimizer at scale 10^{k}"
            );
        }
    }
}
