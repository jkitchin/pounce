//! Expression-domain checks.
//!
//! Two halves, with different evidence:
//!
//! * **Over the bounds** (`expression_domains`, `row_enclosures`) — an
//!   interval walk of every row's expression tree over the variable box.
//!   Each operator with a restricted domain (`log`, `sqrt`, division,
//!   fractional powers, `acos`, …) has its argument's enclosure compared
//!   against that domain. The enclosure is an *over*-approximation (interval
//!   arithmetic forgets correlations: `x - x` encloses to `[l-u, u-l]`), so:
//!   - an argument enclosure entirely outside the domain is a **proof** that
//!     every point in the box fails (`expression_domain_violation`, error);
//!   - one that merely reaches outside is only a **possible** violation
//!     (`possible_expression_domain_violation`, warning, heuristic) — the
//!     bounds do not exclude it, but the true range may;
//!   - one that touches the boundary exactly (`log(x)` with `x ≥ 0`) is
//!     reported because the solver relaxes bounds (`bound_relax_factor`) and
//!     can evaluate a hair outside them.
//!
//!   The same walk encloses each whole row; a row whose enclosure misses its
//!   bounds can never hold (`row_cannot_be_satisfied`, a proof).
//!
//! * **At a point** (`nonfinite_localization`) — every row that evaluates to
//!   NaN/Inf, or whose derivative does, is walked at that point to the
//!   deepest operator that produced the non-finite value from finite inputs.
//!   That operator, its inputs, and the cause (argument outside the domain,
//!   a pole, overflow) are reported. This is a numerical observation about
//!   one point; it says nothing about any other point.

use super::report::{Basis, Confidence, Finding, Report, Severity, fmt_num, interval, num};
use super::{DiagnoseOptions, PointEval, cname, lo_eff, up_eff};
use crate::nl_reader::{BinOp, Expr, NlBody, NlProblem, UnaryOp, eval_expr, render_expression};
use pounce_common::types::Number;
use pounce_presolve::fbbt::Interval;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};

/// Rows with more quadratic variables than this skip the enclosure test:
/// rebuilding the tree of a huge recognized quadratic is exactly the cost the
/// parser avoided by not building it.
const QUAD_ENCLOSURE_MAX_VARS: usize = 5_000;

/// Longest expression text quoted in a finding.
const EXPR_TEXT_MAX: usize = 160;

fn short(s: String) -> String {
    if s.chars().count() <= EXPR_TEXT_MAX {
        s
    } else {
        let t: String = s.chars().take(EXPR_TEXT_MAX).collect();
        format!("{t}…")
    }
}

/// Widen an interval a little outward, for functions evaluated with
/// round-to-nearest libm calls (a few ULP of error).
fn widen(lo: Number, hi: Number) -> Interval {
    let w = |v: Number| v.abs() * 1e-15 + 1e-300;
    Interval::new(lo - w(lo), hi + w(hi))
}

/// Image of `a` under a nondecreasing function.
fn mono(a: Interval, f: impl Fn(Number) -> Number) -> Interval {
    if a.is_empty() {
        return Interval::EMPTY;
    }
    let (lo, hi) = (f(a.lo), f(a.hi));
    if lo.is_nan() || hi.is_nan() {
        return Interval::ENTIRE;
    }
    widen(lo, hi)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    /// Every point of the enclosure is outside the domain.
    Certain,
    /// The enclosure reaches outside the domain.
    Possible,
    /// The enclosure touches the domain boundary, where the value or the
    /// derivative is infinite.
    Boundary,
}

#[derive(Debug, Clone)]
struct DomainEvent {
    kind: EventKind,
    op: &'static str,
    /// What the domain is, in words.
    domain: &'static str,
    arg: Interval,
    expr: String,
    /// Inside a branch of an `if-then-else`, which may guard it.
    guarded: bool,
    /// The value (not just the derivative) is infinite at the boundary.
    value_pole: bool,
}

/// `atan(a / b)` with `b`'s enclosure containing zero.
#[derive(Debug, Clone)]
struct AtanRatio {
    expr: String,
    denom: Interval,
}

struct IvWalker<'a> {
    lo: &'a [Number],
    hi: &'a [Number],
    names: &'a [String],
    tol: Number,
    cache: HashMap<*const Expr, Interval>,
    events: Vec<DomainEvent>,
    atan_ratios: Vec<AtanRatio>,
    opaque: bool,
}

impl<'a> IvWalker<'a> {
    fn render(&self, e: &Expr) -> String {
        short(render_expression(e, self.names))
    }

    /// Check `arg` against `[dlo, dhi]` (`open_lo`: the lower end itself is
    /// outside the domain, e.g. `log` at 0).
    #[allow(clippy::too_many_arguments)]
    fn check(
        &mut self,
        node: &Expr,
        op: &'static str,
        domain: &'static str,
        arg: Interval,
        dlo: Number,
        dhi: Number,
        open_lo: bool,
        guarded: bool,
        value_pole: bool,
    ) {
        if arg.is_empty() {
            return;
        }
        let tlo = self.tol * (1.0 + dlo.abs().min(1e300));
        let thi = self.tol * (1.0 + dhi.abs().min(1e300));
        let below_all = arg.hi < dlo - tlo || (open_lo && arg.hi <= dlo);
        let above_all = arg.lo > dhi + thi;
        let kind = if below_all || above_all {
            Some(EventKind::Certain)
        } else if arg.lo < dlo - tlo || arg.hi > dhi + thi {
            Some(EventKind::Possible)
        } else if (arg.lo <= dlo + tlo && dlo.is_finite())
            || (arg.hi >= dhi - thi && dhi.is_finite())
        {
            Some(EventKind::Boundary)
        } else {
            None
        };
        if let Some(kind) = kind {
            let expr = self.render(node);
            self.events.push(DomainEvent {
                kind,
                op,
                domain,
                arg,
                expr,
                guarded,
                value_pole,
            });
        }
    }

    fn walk(&mut self, e: &Expr, guarded: bool) -> Interval {
        match e {
            Expr::Const(c) => Interval::point(*c),
            Expr::Var(j) => Interval::new(lo_eff(self.lo[*j]), up_eff(self.hi[*j])),
            Expr::Cse(body) => {
                let key = Arc_ptr(body);
                if let Some(iv) = self.cache.get(&key) {
                    return *iv;
                }
                let iv = self.walk(body, guarded);
                self.cache.insert(key, iv);
                iv
            }
            Expr::Sum(args) => args.iter().fold(Interval::point(0.0), |acc, a| {
                acc.add(self.walk(a, guarded))
            }),
            Expr::Binary(op, a, b) => {
                let ia = self.walk(a, guarded);
                let ib = self.walk(b, guarded);
                match op {
                    BinOp::Add => ia.add(ib),
                    BinOp::Sub => ia.sub(ib),
                    BinOp::Mul => ia.mul(ib),
                    BinOp::Div => {
                        if ib.contains_zero() {
                            let kind_pt = ib.lo == 0.0 && ib.hi == 0.0;
                            let expr = self.render(e);
                            self.events.push(DomainEvent {
                                kind: if kind_pt {
                                    EventKind::Certain
                                } else if ib.lo == 0.0 || ib.hi == 0.0 {
                                    EventKind::Boundary
                                } else {
                                    EventKind::Possible
                                },
                                op: "division",
                                domain: "denominator ≠ 0",
                                arg: ib,
                                expr,
                                guarded,
                                value_pole: true,
                            });
                        }
                        ia.div(ib)
                    }
                    BinOp::Pow => self.pow(e, a, b, ia, ib, guarded),
                    BinOp::Atan2 => Interval::new(-std::f64::consts::PI, std::f64::consts::PI),
                    BinOp::CEntropy => {
                        self.check(
                            e,
                            "centropy",
                            "first argument ≥ 0",
                            ia,
                            0.0,
                            Number::INFINITY,
                            false,
                            guarded,
                            false,
                        );
                        self.check(
                            e,
                            "centropy",
                            "second argument > 0",
                            ib,
                            0.0,
                            Number::INFINITY,
                            true,
                            guarded,
                            true,
                        );
                        Interval::ENTIRE
                    }
                }
            }
            Expr::Unary(op, a) => {
                let ia = self.walk(a, guarded);
                use std::f64::consts::{FRAC_PI_2, PI};
                match op {
                    UnaryOp::Neg => ia.neg(),
                    UnaryOp::Exp => ia.exp(),
                    UnaryOp::Abs => ia.abs(),
                    UnaryOp::Sin => ia.sin(),
                    UnaryOp::Cos => ia.cos(),
                    UnaryOp::Sqrt => {
                        self.check(
                            e,
                            "sqrt",
                            "argument ≥ 0",
                            ia,
                            0.0,
                            Number::INFINITY,
                            false,
                            guarded,
                            false,
                        );
                        ia.sqrt()
                    }
                    UnaryOp::Log | UnaryOp::Log10 => {
                        let name = if *op == UnaryOp::Log { "log" } else { "log10" };
                        self.check(
                            e,
                            name,
                            "argument > 0",
                            ia,
                            0.0,
                            Number::INFINITY,
                            true,
                            guarded,
                            true,
                        );
                        let l = ia.ln();
                        if *op == UnaryOp::Log10 {
                            l.mul(Interval::new(
                                0.434_294_481_903_251_7,
                                0.434_294_481_903_251_9,
                            ))
                        } else {
                            l
                        }
                    }
                    UnaryOp::Acos | UnaryOp::Asin => {
                        let name = if *op == UnaryOp::Acos { "acos" } else { "asin" };
                        self.check(
                            e,
                            name,
                            "argument in [-1, 1]",
                            ia,
                            -1.0,
                            1.0,
                            false,
                            guarded,
                            false,
                        );
                        if *op == UnaryOp::Acos {
                            widen(0.0, PI)
                        } else {
                            widen(-FRAC_PI_2, FRAC_PI_2)
                        }
                    }
                    UnaryOp::Acosh => {
                        self.check(
                            e,
                            "acosh",
                            "argument ≥ 1",
                            ia,
                            1.0,
                            Number::INFINITY,
                            false,
                            guarded,
                            false,
                        );
                        if ia.hi >= 1.0 {
                            widen(0.0, ia.hi.acosh())
                        } else {
                            Interval::ENTIRE
                        }
                    }
                    UnaryOp::Atanh => {
                        self.check(
                            e,
                            "atanh",
                            "argument in (-1, 1)",
                            ia,
                            -1.0,
                            1.0,
                            false,
                            guarded,
                            true,
                        );
                        Interval::ENTIRE
                    }
                    UnaryOp::XLogX => {
                        self.check(
                            e,
                            "x·log(x)",
                            "argument ≥ 0",
                            ia,
                            0.0,
                            Number::INFINITY,
                            false,
                            guarded,
                            false,
                        );
                        if ia.lo >= 0.0 {
                            let f = |v: Number| if v == 0.0 { 0.0 } else { v * v.ln() };
                            widen((-(-1.0f64).exp()).min(f(ia.lo)), f(ia.lo).max(f(ia.hi)))
                        } else {
                            Interval::ENTIRE
                        }
                    }
                    UnaryOp::Atan => {
                        if let Expr::Binary(BinOp::Div, _, den) = strip_cse(a) {
                            // Re-walk only for the enclosure: the events in
                            // `den` were recorded when `a` was walked.
                            let (ne, na) = (self.events.len(), self.atan_ratios.len());
                            let id = self.walk(den, guarded);
                            self.events.truncate(ne);
                            self.atan_ratios.truncate(na);
                            if id.contains_zero() {
                                let expr = self.render(e);
                                self.atan_ratios.push(AtanRatio { expr, denom: id });
                            }
                        }
                        mono(ia, Number::atan)
                    }
                    UnaryOp::Tanh => mono(ia, Number::tanh),
                    UnaryOp::Sinh => mono(ia, Number::sinh),
                    UnaryOp::Asinh => mono(ia, Number::asinh),
                    UnaryOp::Cosh => mono(ia.abs(), Number::cosh),
                    UnaryOp::Erf => widen(-1.0, 1.0),
                    UnaryOp::Tan => Interval::ENTIRE,
                }
            }
            Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
                self.walk(a, guarded);
                self.walk(b, guarded);
                Interval::new(0.0, 1.0)
            }
            Expr::Not(a) => {
                self.walk(a, guarded);
                Interval::new(0.0, 1.0)
            }
            Expr::Cond { cond, then_, else_ } => {
                self.walk(cond, guarded);
                let t = self.walk(then_, true);
                let f = self.walk(else_, true);
                t.hull(f)
            }
            Expr::MinList(args) => {
                let mut it = args
                    .iter()
                    .map(|a| self.walk(a, guarded))
                    .collect::<Vec<_>>()
                    .into_iter();
                let first = it.next().unwrap_or(Interval::ENTIRE);
                it.fold(first, Interval::min)
            }
            Expr::MaxList(args) => {
                let mut it = args
                    .iter()
                    .map(|a| self.walk(a, guarded))
                    .collect::<Vec<_>>()
                    .into_iter();
                let first = it.next().unwrap_or(Interval::ENTIRE);
                it.fold(first, Interval::max)
            }
            Expr::Funcall { .. } => {
                self.opaque = true;
                Interval::ENTIRE
            }
        }
    }

    fn pow(
        &mut self,
        node: &Expr,
        a: &Expr,
        b: &Expr,
        ia: Interval,
        ib: Interval,
        guarded: bool,
    ) -> Interval {
        let _ = a;
        if let Some(p) = super::statics::const_fold(b) {
            if p.fract() == 0.0 && p.abs() < 1e9 {
                if p >= 0.0 {
                    return ia.pow_uint(p as u32);
                }
                if ia.contains_zero() {
                    let kind = if ia.lo == 0.0 && ia.hi == 0.0 {
                        EventKind::Certain
                    } else if ia.lo == 0.0 || ia.hi == 0.0 {
                        EventKind::Boundary
                    } else {
                        EventKind::Possible
                    };
                    let expr = self.render(node);
                    self.events.push(DomainEvent {
                        kind,
                        op: "negative power",
                        domain: "base ≠ 0",
                        arg: ia,
                        expr,
                        guarded,
                        value_pole: true,
                    });
                }
                return Interval::point(1.0).div(ia.pow_uint((-p) as u32));
            }
            // Non-integer constant exponent: base must be ≥ 0 (> 0 if p < 0).
            if p < 0.0 {
                self.check(
                    node,
                    "fractional power",
                    "base > 0",
                    ia,
                    0.0,
                    Number::INFINITY,
                    true,
                    guarded,
                    true,
                );
            } else {
                self.check(
                    node,
                    "fractional power",
                    "base ≥ 0",
                    ia,
                    0.0,
                    Number::INFINITY,
                    false,
                    guarded,
                    false,
                );
            }
            let base = ia.intersect(Interval::new(0.0, Number::INFINITY));
            if base.is_empty() {
                return Interval::EMPTY;
            }
            let f = |v: Number| v.powf(p);
            return if p > 0.0 {
                mono(base, f)
            } else if base.lo == 0.0 {
                widen(f(base.hi), Number::INFINITY)
            } else {
                widen(f(base.hi), f(base.lo))
            };
        }
        if let Some(c) = super::statics::const_fold(a) {
            if c > 0.0 {
                return ib.mul(Interval::point(c.ln())).exp();
            }
            return Interval::ENTIRE;
        }
        // Variable base and exponent: x^y is defined for x > 0.
        self.check(
            node,
            "variable power",
            "base > 0",
            ia,
            0.0,
            Number::INFINITY,
            true,
            guarded,
            false,
        );
        if ia.lo > 0.0 {
            ib.mul(ia.ln()).exp()
        } else {
            Interval::ENTIRE
        }
    }
}

#[allow(non_snake_case)]
fn Arc_ptr(b: &std::sync::Arc<Expr>) -> *const Expr {
    std::sync::Arc::as_ptr(b)
}

fn strip_cse(e: &Expr) -> &Expr {
    match e {
        Expr::Cse(b) => strip_cse(b),
        other => other,
    }
}

/// `expression_domains` + `row_enclosures`.
pub fn check_domains(prob: &NlProblem, opts: &DiagnoseOptions, report: &mut Report) {
    let mut walker = IvWalker {
        lo: &prob.x_l,
        hi: &prob.x_u,
        names: &prob.var_names,
        tol: opts.domain_tol,
        cache: HashMap::new(),
        events: Vec::new(),
        atan_ratios: Vec::new(),
        opaque: false,
    };

    // (row label, events, atan ratios) per row; the objective first.
    let mut per_row: Vec<(String, Vec<DomainEvent>, Vec<AtanRatio>)> = Vec::new();
    let mut enclosures: Vec<(usize, Interval)> = Vec::new();
    let mut skipped_quad = 0usize;
    let mut opaque_rows = 0usize;

    if prob.num_obj > 0 {
        if let NlBody::Tree(e) = &prob.obj_nonlinear {
            walker.walk(e, false);
        }
        per_row.push((
            "objective".to_string(),
            std::mem::take(&mut walker.events),
            std::mem::take(&mut walker.atan_ratios),
        ));
    }
    for k in 0..prob.m {
        walker.opaque = false;
        let body = match &prob.con_nonlinear[k] {
            NlBody::Tree(e) => Some(walker.walk(e, false)),
            NlBody::Quad(q) if q.vars.len() <= QUAD_ENCLOSURE_MAX_VARS => {
                let e = prob.con_expr(k);
                Some(walker.walk(&e, false))
            }
            NlBody::Quad(_) => {
                skipped_quad += 1;
                None
            }
        };
        if walker.opaque {
            opaque_rows += 1;
        }
        if let Some(mut iv) = body {
            for &(j, a) in &prob.con_linear[k] {
                let xv = Interval::new(lo_eff(prob.x_l[j]), up_eff(prob.x_u[j]));
                iv = iv.add(xv.mul(Interval::point(a)));
            }
            enclosures.push((k, iv));
        }
        per_row.push((
            cname(prob, k),
            std::mem::take(&mut walker.events),
            std::mem::take(&mut walker.atan_ratios),
        ));
    }

    {
        let mut sink = report.ran("expression_domains");
        for (row, events, atans) in &per_row {
            for ev in events {
                sink.push(domain_finding(row, ev));
            }
            for a in atans {
                sink.push(
                    Finding::new("expression_domains", "atan_ratio_may_need_atan2", Severity::Info,
                        Basis::Heuristic, Confidence::Medium,
                        format!("In {row}, `{}` divides by a quantity whose range over the bounds includes 0.", a.expr))
                    .why("atan(y/x) is undefined at x = 0 and loses the quadrant when x < 0; atan2(y, x) has neither problem.")
                    .next("If the ratio is an angle from two coordinates, use atan2.")
                    .cons(vec![row.clone()])
                    .evidence(json!({"expression": a.expr, "denominator_range": interval(a.denom.lo, a.denom.hi)})),
                );
            }
        }
    }

    report.sections.insert(
        "expression_domains",
        json!({"rows_with_imported_functions": opaque_rows,
               "large_quadratic_rows_not_enclosed": skipped_quad}),
    );
    let mut sink = report.ran("row_enclosures");
    for (k, iv) in enclosures {
        let (gl, gu) = (lo_eff(prob.g_l[k]), up_eff(prob.g_u[k]));
        if iv.is_empty() || iv.is_entire() || gl > gu {
            continue;
        }
        let tol = opts.feas_tol;
        let miss_low = iv.hi < gl - tol * (1.0 + gl.abs());
        let miss_high = iv.lo > gu + tol * (1.0 + gu.abs());
        if miss_low || miss_high {
            sink.push(
                Finding::new("row_enclosures", "row_cannot_be_satisfied", Severity::Error,
                    Basis::Proof, Confidence::Certain,
                    format!("Over the variable bounds, constraint {} can only take values in [{}, {}], which misses its bounds [{}, {}].",
                        cname(prob, k), fmt_num(iv.lo), fmt_num(iv.hi), fmt_num(gl), fmt_num(gu)))
                .why("No point inside the variable bounds satisfies this row, so the problem is infeasible.")
                .next("Check the row's right-hand side and the bounds of the variables it uses; a sign or unit error is the usual cause.")
                .cons(vec![cname(prob, k)])
                .evidence(json!({"enclosure": interval(iv.lo, iv.hi), "bounds": interval(gl, gu),
                    "method": "outward-rounded interval arithmetic over the variable box"})),
            );
        }
    }
}

fn domain_finding(row: &str, ev: &DomainEvent) -> Finding {
    let arg = interval(ev.arg.lo, ev.arg.hi);
    let ev_json = json!({"operator": ev.op, "domain": ev.domain, "argument_range": arg,
        "expression": ev.expr, "inside_if_then_else": ev.guarded,
        "method": "interval enclosure over the variable bounds"});
    let base = match ev.kind {
        EventKind::Certain => Finding::new("expression_domains", "expression_domain_violation",
            if ev.guarded { Severity::Warning } else { Severity::Error }, Basis::Proof, Confidence::Certain,
            format!("In {row}, `{}` is outside its domain ({}) at every point within the variable bounds: the argument ranges over [{}, {}].",
                ev.expr, ev.domain, fmt_num(ev.arg.lo), fmt_num(ev.arg.hi)))
            .why(if ev.guarded {
                "It sits inside an if-then-else branch, so a guard may keep it from being evaluated; if that branch is ever taken the evaluation fails."
            } else {
                "Evaluating this row anywhere in the box produces NaN or Inf, so a solve fails with Invalid_Number_Detected."
            })
            .next("Check the bounds of the variables inside the argument, and the sign conventions of the expression."),
        EventKind::Possible => Finding::new("expression_domains", "possible_expression_domain_violation",
            if ev.guarded { Severity::Info } else { Severity::Warning }, Basis::Heuristic, Confidence::Medium,
            format!("In {row}, the variable bounds do not keep `{}` inside its domain ({}): the argument's enclosure is [{}, {}].",
                ev.expr, ev.domain, fmt_num(ev.arg.lo), fmt_num(ev.arg.hi)))
            .why("If an iterate reaches the part of the box outside the domain, the evaluation returns NaN/Inf and the line search has to back off; at the start point it aborts the solve. Interval enclosures can overstate the range, so this is a risk, not a certainty.")
            .next("Add bounds on the variables inside the argument that keep it in the domain, if the model's meaning allows them."),
        EventKind::Boundary => Finding::new("expression_domains", "domain_boundary_reachable",
            if ev.value_pole && !ev.guarded { Severity::Warning } else { Severity::Info },
            Basis::Heuristic, Confidence::Medium,
            format!("In {row}, the bounds allow `{}` to reach the edge of its domain ({}): the argument's enclosure is [{}, {}].",
                ev.expr, ev.domain, fmt_num(ev.arg.lo), fmt_num(ev.arg.hi)))
            .why(if ev.value_pole {
                "The value is infinite at the edge. The solver relaxes bounds slightly (bound_relax_factor, default 1e-8), so it can evaluate at or just past the edge."
            } else {
                "The derivative is infinite at the edge (the value is finite). The solver relaxes bounds slightly (bound_relax_factor), so an iterate can land there and produce an infinite Jacobian entry."
            })
            .next("Move the bound a little inside the domain (e.g. x ≥ 1e-8 rather than x ≥ 0) if the model allows it."),
    };
    base.cons(vec![row.to_string()]).evidence(ev_json)
}

// ---------------------------------------------------------------------------
// At a point: localize NaN/Inf to an operator.
// ---------------------------------------------------------------------------

/// Why-text for a non-finite value the interior clamp moves the start away
/// from.
const CLAMP_CLEARS: &str = "The solver's interior clamp (bound_push) moves this start inside the bounds, to a point where this is finite, so a default solve from here does not evaluate it. A warm start with a small bound_push, or an iterate that reaches this point, does.";

/// The deepest operator that produced a non-finite value from finite inputs.
#[derive(Debug, Clone)]
struct Culprit {
    expr: String,
    op: String,
    inputs: Vec<Number>,
    output: Number,
    cause: &'static str,
    detail: String,
    /// Address of the node, to group rows sharing it.
    key: usize,
}

struct PtWalker<'a> {
    x: &'a [Number],
    names: &'a [String],
    cache: HashMap<*const Expr, (Number, Option<Culprit>)>,
    unavailable: bool,
}

fn op_name(e: &Expr) -> String {
    match e {
        Expr::Binary(op, _, _) => format!("{op:?}").to_lowercase(),
        Expr::Unary(op, _) => format!("{op:?}").to_lowercase(),
        Expr::Sum(_) => "sum".into(),
        Expr::MinList(_) => "min".into(),
        Expr::MaxList(_) => "max".into(),
        _ => "expression".into(),
    }
}

/// Why `e`, with these finite inputs, produced `out`.
fn classify(e: &Expr, inputs: &[Number], out: Number) -> (&'static str, String) {
    let a = inputs.first().copied().unwrap_or(Number::NAN);
    let b = inputs.get(1).copied().unwrap_or(Number::NAN);
    match e {
        Expr::Unary(UnaryOp::Sqrt, _) if a < 0.0 => ("domain", format!("square root of {a} < 0")),
        Expr::Unary(UnaryOp::Log | UnaryOp::Log10, _) if a < 0.0 => {
            ("domain", format!("logarithm of {a} < 0"))
        }
        Expr::Unary(UnaryOp::Log | UnaryOp::Log10, _) if a == 0.0 => {
            ("pole", "logarithm of 0".into())
        }
        Expr::Unary(UnaryOp::Acos | UnaryOp::Asin, _) if a.abs() > 1.0 => (
            "domain",
            format!("inverse sine/cosine of {a}, outside [-1, 1]"),
        ),
        Expr::Unary(UnaryOp::Acosh, _) if a < 1.0 => ("domain", format!("acosh of {a} < 1")),
        Expr::Unary(UnaryOp::Atanh, _) if a.abs() >= 1.0 => (
            if a.abs() == 1.0 { "pole" } else { "domain" },
            format!("atanh of {a}"),
        ),
        Expr::Unary(UnaryOp::XLogX, _) if a < 0.0 => ("domain", format!("x·log(x) at {a} < 0")),
        Expr::Binary(BinOp::Div, _, _) if b == 0.0 => ("pole", format!("division of {a} by zero")),
        Expr::Binary(BinOp::Pow, _, _) if a < 0.0 && b.fract() != 0.0 => (
            "domain",
            format!("negative base {a} raised to the non-integer power {b}"),
        ),
        Expr::Binary(BinOp::Pow, _, _) if a == 0.0 && b < 0.0 => {
            ("pole", format!("zero raised to the negative power {b}"))
        }
        Expr::Binary(BinOp::CEntropy, _, _) if a < 0.0 || b <= 0.0 => {
            ("domain", format!("centropy({a}, {b}) needs a ≥ 0, b > 0"))
        }
        _ if out.is_infinite() => (
            "overflow",
            format!("finite inputs {inputs:?} overflow to {out}"),
        ),
        _ => (
            "invalid_operation",
            format!("finite inputs {inputs:?} give {out}"),
        ),
    }
}

impl<'a> PtWalker<'a> {
    fn leaf_node(e: &Expr, kids: &[Number]) -> Option<Expr> {
        let c = |i: usize| Box::new(Expr::Const(kids[i]));
        Some(match e {
            Expr::Unary(op, _) => Expr::Unary(*op, c(0)),
            Expr::Binary(op, _, _) => Expr::Binary(*op, c(0), c(1)),
            _ => return None,
        })
    }

    fn walk(&mut self, e: &Expr) -> (Number, Option<Culprit>) {
        match e {
            Expr::Const(c) => (*c, None),
            Expr::Var(j) => (self.x[*j], None),
            Expr::Cse(body) => {
                let key = std::sync::Arc::as_ptr(body);
                if let Some(v) = self.cache.get(&key) {
                    return v.clone();
                }
                let v = self.walk(body);
                self.cache.insert(key, v.clone());
                v
            }
            Expr::Funcall { .. } => {
                self.unavailable = true;
                (Number::NAN, None)
            }
            Expr::Cond { cond, then_, else_ } => {
                let (c, cul) = self.walk(cond);
                if cul.is_some() {
                    return (Number::NAN, cul);
                }
                if c != 0.0 {
                    self.walk(then_)
                } else {
                    self.walk(else_)
                }
            }
            _ => {
                let kids: Vec<&Expr> = match e {
                    Expr::Unary(_, a) | Expr::Not(a) => vec![a],
                    Expr::Binary(_, a, b)
                    | Expr::Compare(_, a, b)
                    | Expr::And(a, b)
                    | Expr::Or(a, b) => vec![a, b],
                    Expr::Sum(v) | Expr::MinList(v) | Expr::MaxList(v) => v.iter().collect(),
                    _ => vec![],
                };
                let mut vals = Vec::with_capacity(kids.len());
                for k in kids {
                    let (v, cul) = self.walk(k);
                    if cul.is_some() {
                        return (v, cul);
                    }
                    vals.push(v);
                }
                if self.unavailable {
                    return (Number::NAN, None);
                }
                let out = match e {
                    Expr::Sum(_) => vals.iter().sum(),
                    Expr::MinList(_) => vals.iter().copied().fold(Number::INFINITY, Number::min),
                    Expr::MaxList(_) => {
                        vals.iter().copied().fold(Number::NEG_INFINITY, Number::max)
                    }
                    Expr::Compare(..) | Expr::And(..) | Expr::Or(..) | Expr::Not(..) => 0.0,
                    _ => match Self::leaf_node(e, &vals) {
                        Some(n) => eval_expr(&n, &[]),
                        None => Number::NAN,
                    },
                };
                if !out.is_finite() && vals.iter().all(|v| v.is_finite()) {
                    let (cause, detail) = classify(e, &vals, out);
                    let cul = Culprit {
                        expr: short(render_expression(e, self.names)),
                        op: op_name(e),
                        inputs: vals,
                        output: out,
                        cause,
                        detail,
                        key: e as *const Expr as usize,
                    };
                    return (out, Some(cul));
                }
                (out, None)
            }
        }
    }
}

/// Operators whose *derivative* is infinite at a point where their value is
/// finite: `acos`/`asin` at ±1, `acosh(1)`, `x·log(x)` at 0.
///
/// `sqrt` at 0 and `x^p` at 0 (`0 < p < 1`) are deliberately absent: the
/// evaluator returns a zero slope there by convention, so they never put an
/// Inf in the Jacobian, and naming them would blame the wrong operator.
fn derivative_poles(
    e: &Expr,
    x: &[Number],
    names: &[String],
    out: &mut Vec<(String, String)>,
    seen: &mut HashMap<*const Expr, ()>,
) {
    let val = |a: &Expr| -> Option<Number> {
        let mut w = PtWalker {
            x,
            names,
            cache: HashMap::new(),
            unavailable: false,
        };
        let (v, c) = w.walk(a);
        (c.is_none() && !w.unavailable).then_some(v)
    };
    let mut push = |node: &Expr, why: String| {
        out.push((short(render_expression(node, names)), why));
    };
    match e {
        Expr::Cse(b) => {
            let key = std::sync::Arc::as_ptr(b);
            if seen.insert(key, ()).is_none() {
                derivative_poles(b, x, names, out, seen);
            }
            return;
        }
        Expr::Unary(op, a) => {
            if let Some(v) = val(a) {
                match op {
                    UnaryOp::Acos | UnaryOp::Asin if v.abs() == 1.0 => {
                        push(e, format!("acos/asin has an infinite derivative at {v}"))
                    }
                    UnaryOp::Acosh if v == 1.0 => {
                        push(e, "acosh has an infinite derivative at 1".into())
                    }
                    UnaryOp::XLogX if v == 0.0 => {
                        push(e, "x·log(x) has an infinite derivative at 0".into())
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    for k in children(e) {
        derivative_poles(k, x, names, out, seen);
    }
}

fn children(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Unary(_, a) | Expr::Not(a) => vec![a],
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            vec![a, b]
        }
        Expr::Sum(v) | Expr::MinList(v) | Expr::MaxList(v) => v.iter().collect(),
        Expr::Cond { cond, then_, else_ } => vec![cond, then_, else_],
        Expr::Cse(b) => vec![b],
        _ => vec![],
    }
}

/// `nonfinite_localization`: at the evaluation point, name the operator
/// behind every non-finite row value or derivative.
///
/// `clamped` is the same point after the solver's interior clamp
/// (`bound_push`), when that moves it. A non-finite value that the clamp
/// removes is still reported — a warm start or an iterate can land on it —
/// but as a warning, since a default solve from here never evaluates it.
pub fn check_nonfinite(
    prob: &NlProblem,
    pt: &PointEval,
    clamped: Option<&PointEval>,
    report: &mut Report,
) {
    // Is row `k` (None: the objective) finite at the clamped point?
    let value_ok_after_clamp = |k: Option<usize>| -> bool {
        let Some(c) = clamped else { return false };
        match k {
            None => c.f.map(|v| v.is_finite()).unwrap_or(false),
            Some(k) => c.g_ok && c.g.get(k).map(|v| v.is_finite()).unwrap_or(false),
        }
    };
    let deriv_ok_after_clamp = |k: Option<usize>| -> bool {
        let Some(c) = clamped else { return false };
        match k {
            None => c.grad_ok && c.grad.iter().all(|v| v.is_finite()),
            Some(k) => {
                c.jac_ok
                    && c.jac_irow
                        .iter()
                        .zip(&c.jac_val)
                        .all(|(&r, v)| r != k || v.is_finite())
            }
        }
    };
    let x = &pt.x;
    let names = &prob.var_names;
    let mut walker = PtWalker {
        x,
        names,
        cache: HashMap::new(),
        unavailable: false,
    };
    // Group rows by culprit node.
    let mut by_culprit: BTreeMap<usize, (Culprit, Vec<(String, Option<usize>)>)> = BTreeMap::new();
    let mut unexplained: Vec<(String, Option<usize>)> = Vec::new();
    let mut unavailable_rows = 0usize;

    let mut rows: Vec<(String, Option<usize>, std::borrow::Cow<'_, Expr>, Number)> = Vec::new();
    if prob.num_obj > 0 {
        rows.push((
            "objective".into(),
            None,
            prob.obj_expr(),
            pt.f.unwrap_or(Number::NAN),
        ));
    }
    let g_bad: Vec<usize> = (0..prob.m)
        .filter(|&k| !pt.g_ok || !pt.g.get(k).copied().unwrap_or(Number::NAN).is_finite())
        .collect();
    for &k in &g_bad {
        rows.push((
            cname(prob, k),
            Some(k),
            prob.con_expr(k),
            pt.g.get(k).copied().unwrap_or(Number::NAN),
        ));
    }
    for (label, idx, e, v) in &rows {
        if v.is_finite() && label == "objective" {
            continue;
        }
        walker.unavailable = false;
        let (_, cul) = walker.walk(e);
        match cul {
            Some(c) => by_culprit
                .entry(c.key)
                .or_insert_with(|| (c, Vec::new()))
                .1
                .push((label.clone(), *idx)),
            None if walker.unavailable => unavailable_rows += 1,
            // The nonlinear part is finite; the linear part overflowed or
            // the evaluator failed.
            None => unexplained.push((label.clone(), *idx)),
        }
    }

    // Derivative poles: rows whose value is finite but whose Jacobian row is not.
    let mut jac_rows: BTreeMap<usize, ()> = BTreeMap::new();
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        if !pt.jac_val[k].is_finite() && pt.g.get(r).copied().unwrap_or(Number::NAN).is_finite() {
            jac_rows.insert(r, ());
        }
    }
    let grad_bad = pt.grad_ok
        && pt.grad.iter().any(|v| !v.is_finite())
        && pt.f.map(|v| v.is_finite()).unwrap_or(false);

    let mut sink = report.ran("nonfinite_localization");
    for (_, (c, rows)) in by_culprit {
        let cleared = rows.iter().all(|(_, k)| value_ok_after_clamp(*k));
        let rows: Vec<String> = rows.into_iter().map(|(l, _)| l).collect();
        let shown: Vec<String> = rows.iter().take(10).cloned().collect();
        let more = rows.len().saturating_sub(shown.len());
        let sev = if cleared {
            Severity::Warning
        } else {
            Severity::Error
        };
        sink.push(
            Finding::new("nonfinite_localization", match c.cause {
                "domain" => "nonfinite_from_domain_violation",
                "pole" => "nonfinite_from_pole",
                "overflow" => "nonfinite_from_overflow",
                _ => "nonfinite_from_invalid_operation",
            }, sev, Basis::Numerical, Confidence::Certain,
                format!("At {}, `{}` produces {} from finite inputs ({}); {} row(s) are non-finite because of it: {}{}.",
                    pt.label, c.expr, c.output, c.detail, rows.len(), shown.join(", "),
                    if more > 0 { format!(" and {more} more") } else { String::new() }))
            .why(if cleared { CLAMP_CLEARS } else { "A solve started here aborts with Invalid_Number_Detected; a solve that reaches here has to cut its step back." })
            .next(match c.cause {
                "domain" | "pole" => "Check the point's provenance (a start outside the operator's domain is a start problem, not a model problem), then the bounds that should keep the argument in the domain.",
                _ => "Look for a term that grows too fast (exp of a large argument, a high power); rescaling the variables or reformulating usually fixes overflow.",
            })
            .cons(rows)
            .evidence(json!({"point": pt.label, "operator": c.op, "expression": c.expr,
                "inputs": c.inputs.iter().map(|v| num(*v)).collect::<Vec<_>>(),
                "output": num(c.output), "cause": c.cause})),
        );
    }
    for (label, k) in unexplained {
        let cleared = value_ok_after_clamp(k);
        sink.push(
            Finding::new("nonfinite_localization", "nonfinite_unlocalized", if cleared { Severity::Warning } else { Severity::Error },
                Basis::Numerical, Confidence::High,
                format!("At {}, {label} is non-finite, but no single operator in its expression produced the non-finite value from finite inputs.", pt.label))
            .why(if cleared { CLAMP_CLEARS } else { "A solve started here aborts with Invalid_Number_Detected." })
            .next("Check the point itself for NaN/Inf entries and the row's linear coefficients for overflow.")
            .cons(vec![label]),
        );
    }
    if unavailable_rows > 0 {
        sink.push(
            Finding::new("nonfinite_localization", "nonfinite_in_imported_function", Severity::Error,
                Basis::Numerical, Confidence::Medium,
                format!("At {}, {unavailable_rows} non-finite row(s) call an imported (external) function, which this check cannot evaluate step by step.", pt.label))
            .why("A solve started here aborts with Invalid_Number_Detected.")
            .next("Evaluate the external function at this point in isolation."),
        );
    }

    let mut deriv_rows: Vec<(String, Option<usize>, std::borrow::Cow<'_, Expr>)> = jac_rows
        .keys()
        .map(|&k| (cname(prob, k), Some(k), prob.con_expr(k)))
        .collect();
    if grad_bad {
        deriv_rows.insert(0, ("objective".into(), None, prob.obj_expr()));
    }
    for (label, k, e) in deriv_rows {
        let cleared = deriv_ok_after_clamp(k);
        let sev = if cleared {
            Severity::Warning
        } else {
            Severity::Error
        };
        let mut poles = Vec::new();
        derivative_poles(&e, x, names, &mut poles, &mut HashMap::new());
        if poles.is_empty() {
            sink.push(
                Finding::new("nonfinite_localization", "nonfinite_derivative_unlocalized", sev,
                    Basis::Numerical, Confidence::High,
                    format!("At {}, the value of {label} is finite but its derivative is not, and no operator with an infinite derivative at this point was found.", pt.label))
                .why(if cleared { CLAMP_CLEARS } else { "The Jacobian or gradient the solver factors contains NaN/Inf." })
                .next("Look for overflow in derivative terms (high powers or exponentials of large arguments).")
                .cons(vec![label.clone()]),
            );
        }
        for (expr, why) in poles {
            sink.push(
                Finding::new("nonfinite_localization", "infinite_derivative_at_point", sev,
                    Basis::Numerical, Confidence::Certain,
                    format!("At {}, {label} has a finite value but an infinite derivative: `{expr}` ({why}).", pt.label))
                .why(if cleared { CLAMP_CLEARS } else { "The Jacobian or gradient the solver factors contains Inf, so the solve aborts with Invalid_Number_Detected." })
                .next("Start, or bound the variable, slightly away from the point where the derivative blows up.")
                .cons(vec![label.clone()])
                .evidence(json!({"point": pt.label, "expression": expr, "reason": why})),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::NlProblemParts;

    fn prob(
        cons: Vec<Expr>,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
        x_l: Vec<Number>,
        x_u: Vec<Number>,
    ) -> NlProblem {
        let n = x_l.len();
        NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Const(0.0),
            obj_constant: 0.0,
            constraints: cons,
            x_l,
            x_u,
            x0: vec![0.0; n],
            g_l,
            g_u,
            var_names: Vec::new(),
            con_names: Vec::new(),
        })
        .unwrap()
    }
    fn v(j: usize) -> Expr {
        Expr::Var(j)
    }
    fn c(x: Number) -> Expr {
        Expr::Const(x)
    }
    fn un(op: UnaryOp, a: Expr) -> Expr {
        Expr::Unary(op, Box::new(a))
    }
    fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        Expr::Binary(op, Box::new(a), Box::new(b))
    }
    fn codes(r: &Report) -> Vec<&'static str> {
        r.findings.iter().map(|f| f.code).collect()
    }
    fn run(p: &NlProblem) -> Report {
        let mut r = Report::new("t".into(), p.n, p.m, 50);
        check_domains(p, &DiagnoseOptions::default(), &mut r);
        r
    }

    #[test]
    fn log_of_negative_box_is_certain() {
        // log(x) with x in [-3, -1].
        let p = prob(
            vec![un(UnaryOp::Log, v(0))],
            vec![-1e19],
            vec![1e19],
            vec![-3.0],
            vec![-1.0],
        );
        assert_eq!(codes(&run(&p)), vec!["expression_domain_violation"]);
    }

    #[test]
    fn log_reaching_outside_is_possible() {
        let p = prob(
            vec![un(UnaryOp::Log, v(0))],
            vec![-1e19],
            vec![1e19],
            vec![-1.0],
            vec![2.0],
        );
        assert_eq!(
            codes(&run(&p)),
            vec!["possible_expression_domain_violation"]
        );
    }

    #[test]
    fn log_touching_zero_is_boundary_and_inside_is_clean() {
        let p = prob(
            vec![un(UnaryOp::Log, v(0))],
            vec![-1e19],
            vec![1e19],
            vec![0.0],
            vec![2.0],
        );
        assert_eq!(codes(&run(&p)), vec!["domain_boundary_reachable"]);
        assert_eq!(run(&p).findings[0].severity, Severity::Warning);
        let ok = prob(
            vec![un(UnaryOp::Log, v(0))],
            vec![-1e19],
            vec![1e19],
            vec![1.0],
            vec![2.0],
        );
        assert!(run(&ok).findings.is_empty());
    }

    #[test]
    fn sqrt_of_shifted_variable_is_not_flagged_by_rounding() {
        // sqrt(x - 1) with x in [2, 3]: the outward-rounded subtraction must
        // not invent a possible violation.
        let p = prob(
            vec![un(UnaryOp::Sqrt, bin(BinOp::Sub, v(0), c(1.0)))],
            vec![-1e19],
            vec![1e19],
            vec![2.0],
            vec![3.0],
        );
        assert!(run(&p).findings.is_empty(), "{:?}", codes(&run(&p)));
    }

    #[test]
    fn division_by_interval_containing_zero() {
        let p = prob(
            vec![bin(BinOp::Div, c(1.0), v(0))],
            vec![-1e19],
            vec![1e19],
            vec![-1.0],
            vec![1.0],
        );
        assert_eq!(
            codes(&run(&p)),
            vec!["possible_expression_domain_violation"]
        );
    }

    #[test]
    fn fractional_power_of_negative_base() {
        let p = prob(
            vec![bin(BinOp::Pow, v(0), c(0.5))],
            vec![-1e19],
            vec![1e19],
            vec![-2.0],
            vec![-1.0],
        );
        assert_eq!(codes(&run(&p)), vec!["expression_domain_violation"]);
        let ok = prob(
            vec![bin(BinOp::Pow, v(0), c(2.0))],
            vec![-1e19],
            vec![1e19],
            vec![-2.0],
            vec![-1.0],
        );
        assert!(
            run(&ok).findings.is_empty(),
            "an integer power of a negative base is fine"
        );
    }

    #[test]
    fn row_that_can_never_hold() {
        // x^2 == -1.
        let p = prob(
            vec![bin(BinOp::Pow, v(0), c(2.0))],
            vec![-1.0],
            vec![-1.0],
            vec![-1e19],
            vec![1e19],
        );
        assert_eq!(codes(&run(&p)), vec!["row_cannot_be_satisfied"]);
        // exp(x) <= -1.
        let p = prob(
            vec![un(UnaryOp::Exp, v(0))],
            vec![-1e19],
            vec![-1.0],
            vec![-1e19],
            vec![1e19],
        );
        assert_eq!(codes(&run(&p)), vec!["row_cannot_be_satisfied"]);
        // exp(x) <= 0 is not flagged: exp(-16) ≈ 1e-7 satisfies it to within
        // the feasibility tolerance, so it is not a proof of infeasibility.
        let p = prob(
            vec![un(UnaryOp::Exp, v(0))],
            vec![-1e19],
            vec![0.0],
            vec![-1e19],
            vec![1e19],
        );
        assert!(run(&p).findings.is_empty());
        // x^2 == 1 is fine.
        let ok = prob(
            vec![bin(BinOp::Pow, v(0), c(2.0))],
            vec![1.0],
            vec![1.0],
            vec![-1e19],
            vec![1e19],
        );
        assert!(run(&ok).findings.is_empty());
    }

    #[test]
    fn atan_ratio_is_flagged() {
        let p = prob(
            vec![un(UnaryOp::Atan, bin(BinOp::Div, v(0), v(1)))],
            vec![-1e19],
            vec![1e19],
            vec![-1.0, -1.0],
            vec![1.0, 1.0],
        );
        let cs = codes(&run(&p));
        assert!(cs.contains(&"atan_ratio_may_need_atan2"), "{cs:?}");
    }

    fn point(p: &NlProblem, x: Vec<Number>) -> PointEval {
        PointEval::evaluate_nl(p, x, "test point".into())
    }

    #[test]
    fn localizes_log_of_negative_at_point() {
        let p = prob(
            vec![bin(BinOp::Add, un(UnaryOp::Log, v(0)), v(1))],
            vec![0.0],
            vec![0.0],
            vec![-1e19; 2],
            vec![1e19; 2],
        );
        let pt = point(&p, vec![-1.0, 0.0]);
        let mut r = Report::new("t".into(), 2, 1, 50);
        check_nonfinite(&p, &pt, None, &mut r);
        assert_eq!(codes(&r), vec!["nonfinite_from_domain_violation"]);
        // A point inside the domain reports nothing.
        let pt = point(&p, vec![1.0, 0.0]);
        let mut r = Report::new("t".into(), 2, 1, 50);
        check_nonfinite(&p, &pt, None, &mut r);
        assert!(r.findings.is_empty());
    }

    #[test]
    fn localizes_overflow_and_derivative_pole() {
        // exp(1000) overflows; acos at 1 has an infinite derivative. (The
        // evaluator returns 0 for sqrt'(0) and for x^p at 0 by convention, so
        // neither ever reaches the Jacobian as Inf.)
        let p = prob(
            vec![un(UnaryOp::Exp, v(0)), un(UnaryOp::Acos, v(1))],
            vec![0.0, 0.0],
            vec![1e19, 1e19],
            vec![-1e19; 2],
            vec![1e19; 2],
        );
        let pt = point(&p, vec![1000.0, 1.0]);
        let mut r = Report::new("t".into(), 2, 2, 50);
        check_nonfinite(&p, &pt, None, &mut r);
        let cs = codes(&r);
        assert!(cs.contains(&"nonfinite_from_overflow"), "{cs:?}");
        assert!(cs.contains(&"infinite_derivative_at_point"), "{cs:?}");
    }
}
