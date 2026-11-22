//! Lifting the live iterate through a presolve layer.
//!
//! The algorithm installs the iterate it hands the intermediate callback
//! ([`pounce_nlp::live_iterate`]) in the space of the problem it solves —
//! the reduced one, when presolve is in the stack. Each presolve wrapper
//! lifts it one layer on its way through `intermediate_callback`, so the
//! user reads their own variables and rows.
//!
//! The lift *is* the wrapper's own `finalize_solution`, run against a
//! stand-in inner TNLP that forwards every evaluation and keeps the
//! `Solution` it is handed instead of finalizing. Reusing the code rather
//! than restating it is the point: the iterate read at the last fire is the
//! solution `finalize_solution` reports because it is computed by the same
//! function — eliminated columns restored, dropped rows re-evaluated, their
//! multipliers recovered — and a later change to that recovery reaches the
//! live iterate without anyone having to remember it.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::Number;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::live_iterate::{self, IterateScaling, UserIterate};
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint, TNLP,
};

/// The inner TNLP's last `eval_grad_f` / `eval_g` / `eval_jac_g` values a
/// presolve layer obtained while the algorithm iterated, each with the
/// inner-space `x` it was taken at. Recorded only when the user's TNLP
/// wants the live iterate ([`TNLP::wants_current_iterate`]).
///
/// The lift reruns the layer's `finalize_solution` every fire, and that
/// re-evaluates the inner problem — `g` for the dropped rows, `∇f` and `J`
/// for the multiplier recovery — at the inner image of the current
/// iterate. The algorithm evaluated exactly that point a moment before
/// (its convergence test needs `∇f`, `J`, `g` at the iterate), through
/// this layer, so those calls are answered from here: same values, no
/// evaluation. A record taken at any other `x` is not used.
///
/// Each entry is shared (`Rc`), so the per-fire clone a lift takes is a
/// refcount bump rather than a copy of the full Jacobian (gh#1238).
#[derive(Clone, Debug, Default)]
pub(crate) struct EvalRecord {
    on: Option<bool>,
    grad_f: Option<Rc<Recorded>>,
    g: Option<Rc<Recorded>>,
    jac: Option<Rc<Recorded>>,
}

/// One recorded evaluation: the `x` it was taken at, and its values.
type Recorded = (Vec<Number>, Vec<Number>);

/// Overwrite `slot` with `(x, v)`, reusing its buffers when no lift still
/// holds them.
fn store(slot: &mut Option<Rc<Recorded>>, x: &[Number], v: &[Number]) {
    if let Some(rc) = slot
        && let Some((at, vals)) = Rc::get_mut(rc)
    {
        at.clear();
        at.extend_from_slice(x);
        vals.clear();
        vals.extend_from_slice(v);
        return;
    }
    *slot = Some(Rc::new((x.to_vec(), v.to_vec())));
}

impl EvalRecord {
    /// Whether to record, asked of `inner` once.
    pub(crate) fn enabled(&mut self, inner: &Rc<RefCell<dyn TNLP>>) -> bool {
        *self
            .on
            .get_or_insert_with(|| inner.borrow().wants_current_iterate())
    }
    pub(crate) fn grad_f(&mut self, x: &[Number], v: &[Number]) {
        store(&mut self.grad_f, x, v);
    }
    pub(crate) fn g(&mut self, x: &[Number], v: &[Number]) {
        store(&mut self.g, x, v);
    }
    pub(crate) fn jac(&mut self, x: &[Number], v: &[Number]) {
        store(&mut self.jac, x, v);
    }
}

/// `rec`'s values, into `out`, when they were taken at `x` and fit.
fn serve(rec: &Option<Rc<Recorded>>, x: &[Number], out: &mut [Number]) -> bool {
    match rec.as_deref() {
        Some((at, v)) if at.as_slice() == x && v.len() == out.len() => {
            out.copy_from_slice(v);
            true
        }
        _ => false,
    }
}

/// Forwards the evaluations a `finalize_solution` makes of its inner TNLP —
/// answering those the layer's [`EvalRecord`] already holds — and captures
/// the solution it would have handed on.
struct Capture {
    inner: Rc<RefCell<dyn TNLP>>,
    captured: Option<UserIterate>,
    record: EvalRecord,
}

impl TNLP for Capture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.borrow_mut().get_starting_point(sp)
    }
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        if serve(&self.record.grad_f, x, g) {
            return true;
        }
        // `true`, not `new_x`: a call answered from the record never
        // reached the inner TNLP, so its idea of the current point may be
        // another one.
        let _ = new_x;
        let ok = self.inner.borrow_mut().eval_grad_f(x, true, g);
        if ok {
            self.record.grad_f(x, g);
        }
        ok
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        if serve(&self.record.g, x, g) {
            return true;
        }
        let ok = self.inner.borrow_mut().eval_g(x, true, g);
        if ok {
            self.record.g(x, g);
        }
        ok
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match (x, mode) {
            (Some(x), SparsityRequest::Values { values }) => {
                if serve(&self.record.jac, x, values) {
                    return true;
                }
                let ok = self.inner.borrow_mut().eval_jac_g(
                    Some(x),
                    true,
                    SparsityRequest::Values { values },
                );
                if ok {
                    self.record.jac(x, values);
                }
                ok
            }
            (x, mode) => self.inner.borrow_mut().eval_jac_g(x, new_x, mode),
        }
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner
            .borrow_mut()
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.captured = Some(UserIterate {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            g: sol.g.to_vec(),
            lambda: sol.lambda.to_vec(),
            ..Default::default()
        });
    }
}

/// Where each entry of a presolve layer's (outer, reduced) problem sits in
/// the problem it wraps: `cols[k]` is the inner column outer column `k`
/// copies, `rows[k]` the inner row outer row `k` copies. Inner columns and
/// rows absent from these are the ones the layer removed.
pub(crate) struct LiftMaps<'a> {
    pub cols: &'a [usize],
    pub rows: &'a [usize],
    /// The inner problem's declared variable bounds, for a column this
    /// layer removed (empty: none removed).
    pub declared_x_l: &'a [Number],
    pub declared_x_u: &'a [Number],
    /// The layer's record of the inner evaluations the algorithm's
    /// iteration made.
    pub record: EvalRecord,
}

/// The algorithm's variable bounds carried to the inner shape: a kept
/// column keeps the bound the algorithm measures its slack to (whatever this
/// layer made of the declared one — the slack the algorithm has *is* to that
/// bound); a removed column, which the algorithm never held, its declared
/// bound — the only one there is, unwidened.
fn lift_alg_bounds(
    outer: &[Number],
    maps: &LiftMaps<'_>,
    declared: &[Number],
    n_in: usize,
) -> Vec<Number> {
    if outer.len() != maps.cols.len() {
        return Vec::new();
    }
    let mut out: Vec<Number> = (0..n_in)
        .map(|j| declared.get(j).copied().unwrap_or(Number::NAN))
        .collect();
    for (k, &j) in maps.cols.iter().enumerate() {
        if j < n_in {
            out[j] = outer[k];
        }
    }
    out
}

/// The outer iterate's [`IterateScaling`] carried to the inner shape: a
/// kept column or row keeps its factor, a removed one — which the
/// algorithm never scaled — gets `1` (and is marked as not held by the
/// algorithm). This is what defines the `scaled = TRUE` readers under
/// presolve: the scaled values of the problem the algorithm solved where it
/// has the entry, the unscaled lifted value (times the objective factor,
/// for a multiplier) where it does not.
fn lift_scaling(
    s: &IterateScaling,
    maps: &LiftMaps<'_>,
    n_in: usize,
    m_in: usize,
) -> Option<IterateScaling> {
    if s.x.len() != maps.cols.len() || s.g.len() != maps.rows.len() {
        return None;
    }
    let mut out = IterateScaling {
        obj: s.obj,
        x: vec![1.0; n_in],
        x_in_alg: vec![false; n_in],
        g: vec![1.0; m_in],
    };
    for (k, &j) in maps.cols.iter().enumerate() {
        if j < n_in {
            out.x[j] = s.x[k];
            out.x_in_alg[j] = s.x_in_alg.get(k).copied().unwrap_or(true);
        }
    }
    for (k, &i) in maps.rows.iter().enumerate() {
        if i < m_in {
            out.g[i] = s.g[k];
        }
    }
    Some(out)
}

/// `∇f + Jᵀλ − z_L + z_U` of `c`'s inner problem at `it`.
fn inner_lagrangian_gradient(c: &mut Capture, it: &UserIterate) -> Option<Vec<Number>> {
    let info = c.get_nlp_info()?;
    let (n, nnz) = (info.n as usize, info.nnz_jac_g as usize);
    if it.x.len() != n {
        return None;
    }
    let mut grad = vec![0.0; n];
    if !c.eval_grad_f(&it.x, true, &mut grad) {
        return None;
    }
    let (mut irow, mut jcol, mut vals) = (vec![0; nnz], vec![0; nnz], vec![0.0; nnz]);
    if nnz > 0
        && (!c.eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) || !c.eval_jac_g(
            Some(&it.x),
            false,
            SparsityRequest::Values { values: &mut vals },
        ))
    {
        return None;
    }
    let off = match info.index_style {
        pounce_nlp::tnlp::IndexStyle::Fortran => 1,
        pounce_nlp::tnlp::IndexStyle::C => 0,
    };
    let idx = |v: &[pounce_common::types::Index]| -> Vec<usize> {
        v.iter()
            .map(|&k| usize::try_from(k - off).unwrap_or(usize::MAX))
            .collect()
    };
    Some(live_iterate::lagrangian_gradient(
        it,
        &grad,
        &idx(&irow),
        &idx(&jcol),
        &vals,
    ))
}

/// The installed live iterate lifted through `this` — `this`'s own
/// `finalize_solution` with `inner` (reached through `slot`) swapped for a
/// [`Capture`] for the duration. `None` when no iterate is installed or it
/// is not shaped like `this`'s problem (`outer_dims`), in which case the
/// caller forwards the callback with the iterate untouched.
///
/// The caller saves and restores whatever state its `finalize_solution`
/// records about the *final* solution, which this run must not overwrite.
pub(crate) fn lift<T: TNLP>(
    this: &mut T,
    slot: fn(&mut T) -> &mut Rc<RefCell<dyn TNLP>>,
    outer_dims: (usize, usize),
    maps: LiftMaps<'_>,
) -> Option<UserIterate> {
    let it = live_iterate::current_user_iterate()?;
    let (n, m) = outer_dims;
    if it.x.len() != n
        || it.z_l.len() != n
        || it.z_u.len() != n
        || it.g.len() != m
        || it.lambda.len() != m
    {
        return None;
    }
    let original = Rc::clone(slot(this));
    let capture = Rc::new(RefCell::new(Capture {
        inner: Rc::clone(&original),
        captured: None,
        record: maps.record.clone(),
    }));
    let as_dyn: Rc<RefCell<dyn TNLP>> = capture.clone();
    *slot(this) = as_dyn;
    this.finalize_solution(
        Solution {
            // Unread by the lift; the solve has not ended.
            status: SolverReturn::Success,
            x: &it.x,
            z_l: &it.z_l,
            z_u: &it.z_u,
            g: &it.g,
            lambda: &it.lambda,
            obj_value: Number::NAN,
        },
        &IpoptData::default(),
        &IpoptCq::default(),
    );
    *slot(this) = original;
    let mut lifted = capture.borrow_mut().captured.take()?;
    // `∇f + Jᵀλ − z_L + z_U` of the inner problem at the lifted iterate,
    // through the capture, so from the record or `finalize_solution`'s own
    // evaluations wherever those were at this point.
    lifted.grad_lag_x =
        inner_lagrangian_gradient(&mut capture.borrow_mut(), &lifted).unwrap_or_default();
    let (n_in, m_in) = (lifted.x.len(), lifted.g.len());
    lifted.scaling = it
        .scaling
        .as_ref()
        .and_then(|s| lift_scaling(s, &maps, n_in, m_in));
    lifted.x_l_alg = lift_alg_bounds(&it.x_l_alg, &maps, maps.declared_x_l, n_in);
    lifted.x_u_alg = lift_alg_bounds(&it.x_u_alg, &maps, maps.declared_x_u, n_in);
    Some(lifted)
}
