//! gh#974 — `pin_working_set` must keep pruning while the pinned set shrinks.
//!
//! Pinning a working set whose constraint block is rank-deficient trips the
//! masked-rank guard in `factor_pinned_primal`; the caller prunes to an
//! independent subset and factors again. That retry was single shot, but the
//! rank test's verdict depends on the shift the factorization settles at, so
//! the retry's *own* guard can reject a subset the first prune called
//! independent — and its `?` then surfaced a hard `LinearSolverFailure`.
//! `pounce-convex` reports that as `numerical_failure` at iteration 0 with a
//! zero-filled point, which is how the issue first showed it (seed 388, the
//! homotopy's `t = 1` handoff pinning 14 rows, pruned to 13, rejected at 12).
//!
//! The issue's own route no longer reaches here once dependent equality rows
//! are pruned on cold entry, so this drives `solve_with_working_set` directly.
//! Data: the issue's seed 388 (`numpy.random.default_rng`, printed with
//! `repr`), with its `G` rows and its `|x| ≤ 3` box written as rows exactly as
//! `pounce-convex` hands them to this crate. The working set pins the four
//! equality rows (one of them 2.5 × row 0) and eight general rows. Over 4000
//! random 12- to 19-row working sets on this data the single-shot retry
//! returned a hard error on 159; the loop on none.

use pounce_feral::FeralSolverInterface;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};

// seed 388: n=15, 3 independent equality rows + row 0 scaled by 2.5, 6 general inequality rows
const L388: [f64; 15] = [
    0.4623527139807304,
    -0.5030101628589015,
    0.37671621409393663,
    -0.8930019426425657,
    -0.7241105154865255,
    0.9050445856282395,
    0.2992657644113015,
    -0.007946543080506605,
    -0.24810921487061502,
    0.6492455148657802,
    -0.6609533197362643,
    -0.5016386661282591,
    0.6353693786594669,
    -0.14832909980344866,
    2.449247562873487,
];
const C388: [f64; 15] = [
    -0.7315369333782729,
    -0.5154864962599535,
    -1.022568604899134,
    0.12920179568047446,
    0.5812613269508529,
    0.20198912945545783,
    -0.9619218734274643,
    -0.5550746888553302,
    1.316682008308259,
    -0.43804449823855435,
    1.1863301509405448,
    0.0025943946974067863,
    0.10299258384768865,
    -1.2586329130278147,
    1.303953581434348,
];
const A388: [[f64; 15]; 3] = [
    [
        1.7805803767604464,
        0.01244296650657871,
        0.8278133331152813,
        -3.1518930977477653,
        -0.40371317324272155,
        -1.1855010535049542,
        1.7951768487088668,
        -1.2766535595343427,
        0.5034278893624906,
        -0.007987798585907264,
        0.36219188220034637,
        0.12792266097222219,
        -1.5557325501538117,
        -0.1465210101986788,
        -0.37799188412889817,
    ],
    [
        0.14585606421837125,
        0.7466109811119199,
        1.8299950255874333,
        0.28596606553658527,
        0.9358555375954101,
        -0.9221236064986403,
        1.1010794654190716,
        -0.332450475757209,
        1.389975579944678,
        2.069513400910282,
        0.7012472962080782,
        0.7529361088978461,
        -0.13675326490522388,
        -0.0019325354498979362,
        1.3450137411174752,
    ],
    [
        -1.4952198527549942,
        -0.2624445961748636,
        -0.8130708480828471,
        -1.3753455475639225,
        0.10634937252353081,
        -0.09719402305971764,
        -0.6866328226760606,
        1.7415587062630433,
        0.21776747740965632,
        -0.15448190156496822,
        -0.42242263596799956,
        -0.8932577008070234,
        -2.139332117463004,
        -1.630419781499398,
        0.07281808499691247,
    ],
];
const B388: [f64; 4] = [
    -0.45880213696087324,
    1.2606993883439173,
    -4.7600192544291895,
    -1.1470053424021802,
];
const G388: [[f64; 15]; 6] = [
    [
        -0.13896918427061533,
        -0.4724325344205648,
        1.9513406022538062,
        0.8705957377603024,
        0.3226486687293457,
        -1.0147622986609752,
        -1.8802278913791728,
        -1.2432567792589069,
        0.12623950523964478,
        0.3567355862224305,
        0.2644225523851181,
        0.16547202329977806,
        -0.1103124448597283,
        -0.6226148359377608,
        -1.8582210490926068,
    ],
    [
        1.654646819117971,
        0.2132986824673605,
        1.075192433990684,
        -0.4616362763269629,
        0.5617103922468348,
        1.5404382823410538,
        -0.18270049537583866,
        0.6130719706402181,
        2.812312161425623,
        0.7572096814442457,
        0.3776932877660025,
        0.9350018819346718,
        -1.6819091481607844,
        -0.637085689252854,
        -0.2571480898763022,
    ],
    [
        -0.575485305689605,
        0.3955906120157775,
        -1.1100383005523782,
        -0.2978849822970921,
        -0.819523132013627,
        0.5545710742207112,
        -0.3552346635473443,
        -1.4433022689353074,
        0.4619047973495733,
        -0.1688755482677482,
        -1.1780261772343286,
        1.0588028234710054,
        -0.7154101081951838,
        0.038399873339425336,
        0.8112451831250279,
    ],
    [
        -0.886413382115819,
        -0.11315382774142427,
        0.8570217624410303,
        -0.26815183396361897,
        -1.5641567188753835,
        1.7790484795145813,
        0.7784030179334629,
        2.506653402012084,
        0.5484582210112151,
        -1.5305870139093698,
        0.06880114521243971,
        -2.0951010683177826,
        -0.8098241681911733,
        -1.1438618785853845,
        -1.9848820071047104,
    ],
    [
        -1.0339149754744206,
        -0.6031502155000419,
        1.810025923660438,
        -0.5279011414705386,
        0.3389372056152109,
        1.6553926492777633,
        1.3779818601259683,
        0.5838520987125694,
        0.8812939298227396,
        -0.43025197151341327,
        0.6541872063303299,
        -0.8906634020623493,
        -0.7532643164697365,
        -1.455455392737726,
        -0.0360642239414809,
    ],
    [
        -1.3362119592132073,
        0.42094672247162995,
        -0.5737357773987221,
        -0.8666260272669091,
        -1.3234041340509715,
        -1.6442971995071682,
        -1.7389324203591179,
        1.7917719850603426,
        -0.6357842369713884,
        0.45160687032456637,
        0.71624173392003,
        -0.037088143345637266,
        0.347831427610499,
        -0.8217210840437537,
        0.10606739383067883,
    ],
];
const H388: [f64; 6] = [
    2.1803048077886276,
    -1.0084841155392297,
    -0.6347795343302625,
    -3.5844505786947667,
    -1.2450232943015207,
    -5.251596079772503,
];

#[test]
fn a_twice_rejected_prune_is_pruned_again_not_a_hard_error() {
    let n = 15usize;
    let (mut hi, mut hj, mut hv) = (vec![], vec![], vec![]);
    for i in 0..n {
        for j in 0..=i {
            hi.push(i as i32 + 1);
            hj.push(j as i32 + 1);
            hv.push(L388[i] * L388[j]);
        }
    }
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(n as i32, hi, hj));
    h.set_values(&hv);

    // Rows in `pounce-convex`'s order: the equalities (the last one 2.5 × row
    // 0, with numpy's own right-hand side), the general rows, then the box.
    let mut rows: Vec<Vec<f64>> = A388.iter().map(|r| r.to_vec()).collect();
    rows.push(A388[0].iter().map(|v| 2.5 * v).collect());
    let mut bl: Vec<f64> = B388.to_vec();
    let mut bu: Vec<f64> = B388.to_vec();
    for (i, r) in G388.iter().enumerate() {
        rows.push(r.to_vec());
        bl.push(-1e20);
        bu.push(H388[i]);
    }
    for sign in [1.0, -1.0] {
        for j in 0..n {
            let mut r = vec![0.0; n];
            r[j] = sign;
            rows.push(r);
            bl.push(-1e20);
            bu.push(3.0);
        }
    }
    let m = rows.len();
    let (mut ai, mut aj, mut av) = (vec![], vec![], vec![]);
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            if v != 0.0 {
                ai.push(i as i32 + 1);
                aj.push(j as i32 + 1);
                av.push(v);
            }
        }
    }
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(m as i32, n as i32, ai, aj));
    a.set_values(&av);
    let (xl, xu) = (vec![-1e20; n], vec![1e20; n]);
    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: &C388,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut constraints = vec![ConsStatus::Inactive; m];
    for (i, c) in constraints.iter_mut().enumerate() {
        if i < 4 {
            *c = ConsStatus::Equality;
        } else if [7, 9, 22, 28, 29, 34, 39].contains(&i) || i == 4 {
            *c = ConsStatus::AtUpper;
        }
    }
    let working = WorkingSet {
        constraints,
        bounds: vec![BoundStatus::Inactive; n],
    };
    let backend: Box<dyn SparseSymLinearSolverInterface> = Box::new(FeralSolverInterface::new());
    let sol = ParametricActiveSetSolver::new(backend)
        .solve_with_working_set(&qp, &working, &QpOptions::default())
        .expect("a rank-deficient hint is pruned, not a hard error");
    assert_eq!(sol.status, QpStatus::Optimal);
    // Oracle: Clarabel 0.11.1 on the same data, as in the issue.
    assert!(
        (sol.obj - -19.67063989646251).abs() < 1e-6 * 19.67,
        "obj {}",
        sol.obj
    );
}
