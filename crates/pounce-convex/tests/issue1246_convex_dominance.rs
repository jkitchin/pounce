//! gh#1246 finding 3: convex-dominated columns are fixed at their lower bound.
//!
//! A column with the same support as two others, equal to them in every row
//! but one, lies on the segment between them in that row; if it costs
//! strictly more than the matching mix of the two, its weight can always move
//! onto them, so it is zero at every optimum. One fixture per branch of
//! `convex_dominance_once`:
//!
//! * strictly above the hull: removed;
//! * exactly on a hull edge: kept (a tie could be circular across groups);
//! * an anchor bounded above: nothing removed (it could not take the weight);
//! * a column in `P`: never grouped;
//! * a nonzero lower bound: removed at it, with the rows and objective shifted;
//! * identical columns: the cheapest survives, an exact tie keeps one;
//! * a randomized KKT roundtrip over families that share rows, so the same
//!   column is in several groups at once.
//!
//! Every case solves with and without presolve and checks the *original*
//! problem's KKT conditions at the postsolved point, so a removal that loses
//! the optimum, or a recovered multiplier with the wrong sign, fails here.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const INF: f64 = f64::INFINITY;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn removed(prob: &QpProblem) -> usize {
    match presolve(prob) {
        PresolveOutcome::Reduced(ps) => ps.stats().convex_dominated_cols,
        _ => panic!("expected a reduction"),
    }
}

/// The original problem's KKT conditions at `s`, to `tol`.
fn assert_kkt(p: &QpProblem, s: &QpSolution, tol: f64) {
    let mut g = p.c.clone();
    p.p_mul(&s.x, &mut g);
    p.at_mul(&s.y, &mut g);
    p.gt_mul(&s.z, &mut g);
    for j in 0..p.n {
        let stat = g[j] - s.z_lb[j] + s.z_ub[j];
        assert!(stat.abs() < tol, "stationarity[{j}] = {stat:e}");
        assert!(
            s.z_lb[j] > -tol && s.z_ub[j] > -tol,
            "bound multiplier sign [{j}]"
        );
        assert!(
            s.x[j] >= p.lb_of(j) - tol && s.x[j] <= p.ub_of(j) + tol,
            "box [{j}]"
        );
        if p.lb_of(j).is_finite() {
            assert!(
                (s.z_lb[j] * (s.x[j] - p.lb_of(j))).abs() < 1e-5,
                "lb comp [{j}]"
            );
        }
    }
    let mut ax = vec![0.0; p.m_eq()];
    p.a_mul(&s.x, &mut ax);
    for (i, (a, b)) in ax.iter().zip(&p.b).enumerate() {
        assert!((a - b).abs() < tol, "row {i}: {a} vs {b}");
    }
    let mut gx = vec![0.0; p.m_ineq()];
    p.g_mul(&s.x, &mut gx);
    for i in 0..p.m_ineq() {
        let slack = p.h[i] - gx[i];
        assert!(slack > -tol, "G row {i}: slack {slack:e}");
        assert!(s.z[i] > -tol, "z[{i}] = {}", s.z[i]);
        assert!((s.z[i] * slack).abs() < 1e-5, "ineq comp row {i}");
    }
}

/// Presolved and plain solves agree on the objective, and the presolved
/// point is a KKT point of the original problem.
fn same_answer(prob: &QpProblem) {
    let o = QpOptions::default();
    let plain = solve_qp_ipm(prob, &o, backend);
    let pre = solve_with_presolve(prob, |q| solve_qp_ipm(q, &o, backend));
    assert_eq!(plain.status, QpStatus::Optimal);
    assert_eq!(pre.status, QpStatus::Optimal);
    assert!(
        (plain.obj - pre.obj).abs() <= 1e-6 * (1.0 + plain.obj.abs()),
        "plain {} vs presolved {}",
        plain.obj,
        pre.obj
    );
    assert_kkt(prob, &pre, 1e-6);
}

/// Three columns sharing rows 0 and 1, equal in row 0 and differing in row 1
/// (`a = 1, 2, 3`), so the middle one is the average of the outer two. It is
/// covered by the `≥` row 0 and capped by the `≤` row 1, so the optimum has to
/// use some column. `costs` decides which survive; `ub` boxes them.
fn family(costs: [f64; 3], lb: [f64; 3], ub: [f64; 3], p: Vec<Triplet>) -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: p,
        c: costs.to_vec(),
        a: vec![],
        b: vec![],
        g: vec![
            // −(x₀ + x₁ + x₂) ≤ −4   (cover four units)
            Triplet::new(0, 0, -1.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(0, 2, -1.0),
            // x₀ + 2x₁ + 3x₂ ≤ 10
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(1, 2, 3.0),
        ],
        h: vec![-4.0, 10.0],
        lb: lb.to_vec(),
        ub: ub.to_vec(),
    }
}

#[test]
fn a_column_strictly_above_the_hull_is_removed() {
    // mix of the outer two at a = 2 costs (1 + 2)/2 = 1.5 < 3.
    let prob = family([1.0, 3.0, 2.0], [0.0; 3], [INF; 3], vec![]);
    assert_eq!(removed(&prob), 1);
    same_answer(&prob);
}

#[test]
fn a_column_on_a_hull_edge_is_kept() {
    // Exactly the mix: removing it is not wrong here, but a tie between two
    // groups could have each column excuse the other, so ties stay.
    let prob = family([1.0, 1.5, 2.0], [0.0; 3], [INF; 3], vec![]);
    assert_eq!(removed(&prob), 0);
    same_answer(&prob);
}

#[test]
fn an_anchor_bounded_above_removes_nothing() {
    // x₂ ≤ 0.5: the move onto the outer columns could exceed it.
    let prob = family([1.0, 3.0, 2.0], [0.0; 3], [INF, INF, 0.5], vec![]);
    assert_eq!(removed(&prob), 0);
    same_answer(&prob);
}

#[test]
fn a_column_in_the_hessian_is_never_grouped() {
    let prob = family(
        [1.0, 3.0, 2.0],
        [0.0; 3],
        [INF; 3],
        vec![Triplet::new(1, 1, 1.0)],
    );
    assert_eq!(removed(&prob), 0);
    same_answer(&prob);
}

#[test]
fn a_nonzero_lower_bound_shifts_the_rows_and_objective() {
    // x₁ ≥ 0.25: fixed there, so the rows and the objective absorb it.
    let prob = family([1.0, 3.0, 2.0], [0.0, 0.25, 0.0], [INF; 3], vec![]);
    assert_eq!(removed(&prob), 1);
    same_answer(&prob);
}

/// Identical columns: the cheaper of the pair stays. With equal costs one of
/// them goes, and which one is the same in every group the pair shares. A
/// third, different column keeps the rows from being parallel and the
/// equality from being a doubleton, which earlier rules would fold first.
#[test]
fn identical_columns_keep_one() {
    for (costs, expect) in [([1.0, 2.0], 1), ([2.0, 1.0], 1), ([1.0, 1.0], 1)] {
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![costs[0], costs[1], 5.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
            ],
            b: vec![3.0],
            g: vec![
                Triplet::new(0, 0, 2.0),
                Triplet::new(0, 1, 2.0),
                Triplet::new(0, 2, 1.0),
            ],
            h: vec![10.0],
            lb: vec![0.0; 3],
            ub: vec![INF; 3],
        };
        assert_eq!(removed(&prob), expect, "costs {costs:?}");
        same_answer(&prob);
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random families over shared rows. Each column covers one `≥` row and sits
/// in a few `≤` capacity rows, and siblings differ from a base column in one
/// row's coefficient, so groups overlap: a column can be an anchor in one
/// group and removable in another, which is the case the strict margin
/// exists for. Costs are drawn from a small set, so exact ties and
/// collinear points are common.
#[test]
fn randomized_families_roundtrip_to_a_kkt_point() {
    let mut rng = Lcg(0x1246);
    let mut total_removed = 0;
    for trial in 0..60 {
        let n_cover = 2 + rng.below(2) as usize;
        let n_cap = 3 + rng.below(3) as usize;
        let m = n_cover + n_cap;
        let mut cols: Vec<(Vec<(usize, f64)>, f64)> = Vec::new();
        for _ in 0..(3 + rng.below(4)) {
            let cover = rng.below(n_cover as u64) as usize;
            let mut base: Vec<(usize, f64)> = vec![(cover, -1.0)];
            for r in n_cover..m {
                if rng.below(2) == 0 {
                    base.push((r, [1.0, 2.0, 3.0][rng.below(3) as usize]));
                }
            }
            let cost = [1.0, 2.0, 3.0, 4.0][rng.below(4) as usize];
            cols.push((base.clone(), cost));
            // Siblings: vary one entry of the base.
            for _ in 0..(2 + rng.below(4)) {
                let mut sib = base.clone();
                let k = rng.below(sib.len() as u64) as usize;
                sib[k].1 *= [0.5, 1.5, 2.0, 3.0][rng.below(4) as usize];
                let c = [1.0, 2.0, 3.0, 4.0, 6.0][rng.below(5) as usize];
                cols.push((sib, c));
            }
        }
        let n = cols.len();
        let mut g = Vec::new();
        for (j, (e, _)) in cols.iter().enumerate() {
            for &(r, v) in e {
                g.push(Triplet::new(r, j, v));
            }
        }
        let mut h = vec![-1.0; n_cover];
        h.extend(std::iter::repeat_n(20.0, n_cap));
        let prob = QpProblem {
            n,
            p_lower: vec![],
            c: cols.iter().map(|c| c.1).collect(),
            a: vec![],
            b: vec![],
            g,
            h,
            lb: vec![0.0; n],
            ub: vec![INF; n],
        };
        // Every cover row needs a column; skip draws that left one empty.
        let covered = (0..n_cover).all(|r| cols.iter().any(|(e, _)| e[0].0 == r));
        if !covered {
            continue;
        }
        total_removed += removed(&prob);
        let o = QpOptions::default();
        let plain = solve_qp_ipm(&prob, &o, backend);
        let pre = solve_with_presolve(&prob, |q| solve_qp_ipm(q, &o, backend));
        assert_eq!(plain.status, QpStatus::Optimal, "trial {trial}");
        assert_eq!(pre.status, QpStatus::Optimal, "trial {trial}");
        assert!(
            (plain.obj - pre.obj).abs() <= 1e-6 * (1.0 + plain.obj.abs()),
            "trial {trial}: plain {} vs presolved {}",
            plain.obj,
            pre.obj
        );
        assert_kkt(&prob, &pre, 1e-6);
    }
    // The generator must actually reach the rule, or this proves nothing.
    assert!(
        total_removed > 20,
        "only {total_removed} removals across trials"
    );
}
