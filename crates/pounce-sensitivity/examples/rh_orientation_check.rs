//! Which way round is `compute_reduced_hessian`? Decided by measurement.
//!
//! `reduced_hessian.rs`'s unit test feeds a synthetic dense `K` and selects
//! two of its rows, so `B K⁻¹ Bᵀ` there is a submatrix of an inverse. But the
//! production API takes **pin constraint** indices, and
//! `map_pin_g_to_kkt_rows` puts those at `n_x + n_s + c_block(g)` — the `y_c`
//! multiplier block, not the `x` block. For `K = [[H, Aᵀ], [A, 0]]` the
//! `(y_c, y_c)` block of `K⁻¹` is `−(A H⁻¹ Aᵀ)⁻¹`, which inverts once more
//! than the `x`-block case, so the two paths land on opposite sides.
//!
//! Model: `f = ½(2x₀² + 2x₁² + 2x₀x₁)`, i.e. `H = [[2,1],[1,2]]`, with both
//! variables pinned by equalities. Every variable pinned ⇒ the reduced
//! Hessian is `H` itself.
//!
//! * `[[2,1],[1,2]]`          ⇒ the genuine reduced Hessian.
//! * `[[−2,−1],[−1,−2]]`      ⇒ its negation — the documented convention
//!   (`crossover_sigma_downstream.rs`: "returns `−H_R` under its sign
//!   convention"; `crossover_sigma_frame.rs`: "carries the augmented system's
//!   leading minus on a multiplier row").
//! * `[[2/3,−1/3],[−1/3,2/3]]` ⇒ its inverse.
//!
//! MEASURED: `[[−2,−1],[−1,−2]]`. The magnitudes are `H`'s, not `H⁻¹`'s, so
//! the pin path returns `−H_R` and **not** an inverse. gh#936's step 1 is
//! still wrong about which columns are soft, but because of the sign: on
//! `−H_R` the ascending order puts the most negative — the *stiffest* mode —
//! first.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint,
};
use pounce_sensitivity::Solver;

struct PinnedQuad;

impl TNLP for PinnedQuad {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_scaling_parameters(&mut self, _r: ScalingRequest<'_>) -> bool {
        false
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -1.0e19;
        b.x_u[0] = 1.0e19;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        // x0 = 0.3, x1 = -0.2
        b.g_l[0] = 0.3;
        b.g_u[0] = 0.3;
        b.g_l[1] = -0.2;
        b.g_u[1] = -0.2;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = -0.2;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[0] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0] + x[1];
        g[1] = 2.0 * x[1] + x[0];
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
                irow[1] = 1;
                jcol[1] = 1;
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = 1.0;
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1, 1]);
                jcol.copy_from_slice(&[0 as Index, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj * 2.0;
                values[1] = obj * 1.0;
                values[2] = obj * 2.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn main() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-10, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(PinnedQuad));
    let mut s = Solver::new(app, tnlp);
    let st = s.solve();
    assert!(
        matches!(
            st,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{st:?}"
    );

    let (hr, vals, _vecs) = s
        .compute_reduced_hessian_eigen(&[0, 1], 1.0)
        .expect("reduced hessian");

    println!("H            = [[2, 1], [1, 2]]        eigenvalues 1, 3");
    println!("H^-1         = [[2/3, -1/3], [-1/3, 2/3]]  eigenvalues 1/3, 1");
    println!();
    println!(
        "compute_reduced_hessian(pins=[0,1]) = [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
        hr[0], hr[2], hr[1], hr[3]
    );
    println!(
        "  eigenvalues (ascending) = [{:.6}, {:.6}]",
        vals[0], vals[1]
    );
    println!();
    if (hr[0] - 2.0).abs() < 1e-6 {
        println!("=> the GENUINE reduced Hessian, positive sign.");
        println!("   Ascending order would put the SOFT modes first.");
    } else if (hr[0] + 2.0).abs() < 1e-6 {
        println!("=> -H_R: the genuine reduced Hessian NEGATED (the documented convention).");
        println!("   Magnitudes are H's (2, 1), not H^-1's (2/3, 1/3), so this is NOT an inverse.");
        println!("   Ascending order over negatives puts the most negative -- the STIFFEST");
        println!("   mode -- first, so gh#936 step 1 is wrong about which columns are soft,");
        println!("   but by the SIGN convention, not by an inversion.");
    } else if (hr[0] - 2.0 / 3.0).abs() < 1e-6 {
        println!("=> the INVERSE reduced Hessian.");
    } else {
        println!("=> neither; investigate.");
    }

    // For contrast, the operator the scaling study iterates on is the x block
    // of K^-1, which IS an inverse -- verified there against analytic
    // eigenpairs to 1e-10 at n = 100 000. The two paths differ because pins
    // land in the y_c multiplier block, where the Schur complement inverts
    // once more.
    let dims = s.block_dims().unwrap();
    println!();
    println!("kkt blocks (x,s,y_c,y_d,z_l,z_u,v_l,v_u) = {dims:?}");
}
