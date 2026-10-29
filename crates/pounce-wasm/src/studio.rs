//! The interactive debugger in the browser (the `studio` feature).
//!
//! [`pounce_debug_solve`] solves the loaded model with the same JSON
//! debugger `pounce --debug-json` runs: the newline-delimited protocol
//! (`hello`, `pause`, `progress`, `result`, `terminated`) is written to
//! stdout, and commands are read from stdin, one per line. Under WASI those
//! are `fd_write` on fd 1 and `fd_read` on fd 0, so the host decides how a
//! command arrives. The studio build runs the module through Binaryen's
//! Asyncify (`build.sh --studio`), which lets `fd_read` be an async
//! JavaScript import that resolves when the reader clicks Step: the solve
//! stays suspended at the pause in between, with no threads and no shared
//! memory.
//!
//! As with the CLI's JSON mode, stdout is a pure protocol channel:
//! `print_level` is forced to 0 and the per-iteration `progress` events
//! replace the iteration table.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_cli::cli::DebugMode;
use pounce_cli::debug_repl::{EquationBook, SolverDebugger, StructureBook};
use pounce_common::types::Index;
use pounce_nl::nl_reader::{
    NlProblem, constraint_jacobian_sparsity, render_all_constraint_equations,
};
use pounce_presolve::incidence::{EqualityIncidence, ProbeView};

use crate::{
    LAST_SOLVE, LOADED, error_json, guarded, solve_loaded_with, str_from_parts, to_payload,
};

/// What the debugger's `print equation` and `diagnose` need from the source
/// model, captured at load time because [`NlTnlp`](pounce_nl::nl_reader::NlTnlp)
/// takes ownership of the parsed problem.
struct ModelBooks {
    var_names: Vec<String>,
    con_names: Vec<String>,
    equations: Vec<String>,
    n: usize,
    m: usize,
    jac_irow: Vec<Index>,
    jac_jcol: Vec<Index>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
}

thread_local! {
    static BOOKS: RefCell<Option<ModelBooks>> = const { RefCell::new(None) };
}

/// Record the model's names, rendered equations and Jacobian structure for
/// a later [`pounce_debug_solve`]. Called by `pounce_load`.
pub(crate) fn remember_model(prob: &NlProblem) {
    let (jac_irow, jac_jcol) = constraint_jacobian_sparsity(prob);
    let books = ModelBooks {
        var_names: prob.var_names.clone(),
        con_names: prob.con_names.clone(),
        equations: render_all_constraint_equations(prob),
        n: prob.n,
        m: prob.m,
        jac_irow,
        jac_jcol,
        g_l: prob.g_l.clone(),
        g_u: prob.g_u.clone(),
    };
    BOOKS.with(|slot| *slot.borrow_mut() = Some(books));
}

/// Hand the debugger the equation and structure books, built the same way
/// the CLI builds them for `--debug-json`.
fn attach_books(dbg: &mut SolverDebugger) {
    BOOKS.with(|slot| {
        let slot = slot.borrow();
        let Some(b) = slot.as_ref() else { return };
        dbg.set_equation_book(EquationBook::new(b.con_names.clone(), b.equations.clone()));
        let probe = ProbeView {
            n_vars: b.n,
            m_rows: b.m,
            jac_irow: &b.jac_irow,
            jac_jcol: &b.jac_jcol,
            jac_values: None,
            g_l: &b.g_l,
            g_u: &b.g_u,
            linearity: None,
            one_based: false,
            eq_tol: 1e-12,
            excluded_vars: None,
            excluded_rows: None,
        };
        dbg.set_structure_book(StructureBook::new(
            EqualityIncidence::from_probe(&probe),
            b.con_names.clone(),
            b.var_names.clone(),
        ));
    });
}

/// Solve the loaded model under the JSON debugger. `opts` is an
/// `ipopt.opt`-style options string, as for `pounce_solve`. The protocol
/// runs on stdin/stdout for the duration of the call; when the solve ends
/// a `terminated` event is written and the call returns the same JSON
/// result `pounce_solve` does (so `.sol` / CSV export work afterwards).
///
/// Closing stdin (EOF) at a pause aborts the solve, as it does for the CLI.
///
/// # Safety
/// `opts_ptr`/`opts_len` must describe a readable region valid for the
/// call, or be `(null, 0)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_debug_solve(opts_ptr: *const u8, opts_len: usize) -> *mut u8 {
    guarded("debug_solve", || {
        // SAFETY: forwarded from this function's own safety contract.
        let opts = match unsafe { str_from_parts(opts_ptr, opts_len) } {
            Ok(s) => s,
            Err(e) => return error_json(e),
        };
        let Some(tnlp) = LOADED.with(|slot| slot.borrow().clone()) else {
            return error_json("no model loaded — call pounce_load first");
        };
        let result = solve_loaded_with(tnlp, opts, |app| {
            let _ = app.options_mut().read_from_str("print_level 0\n", true);
            let reg = Some(Rc::clone(app.registered_options()));
            let mut dbg = SolverDebugger::new(DebugMode::Json, reg);
            attach_books(&mut dbg);
            app.set_debug_hook(Rc::new(RefCell::new(dbg)));
        });
        println!("{}", terminated_event(&result));
        to_payload(&result.to_string())
    })
}

/// The CLI's `terminated` lifecycle event, from the result payload and the
/// status [`solve_loaded_with`] recorded.
fn terminated_event(result: &serde_json::Value) -> serde_json::Value {
    if let Some(err) = result.get("error") {
        return serde_json::json!({ "event": "terminated", "error": err });
    }
    let message = LAST_SOLVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|o| pounce_solve_report::console::status_message(o.status))
    });
    let evals = &result["evals"];
    serde_json::json!({
        "event": "terminated",
        "status": result["status"],
        "status_message": message,
        "iterations": result["iterations"],
        "objective": result["objective"],
        "evals": {
            "obj": evals["objective"],
            "obj_grad": evals["objective_grad"],
            "constr": evals["constraints"],
            "constr_jac": evals["constraint_jac"],
            "hess": evals["hessian"],
        },
    })
}

// ---------------------------------------------------------------------------
// Analysis (pounce-studio-core)
// ---------------------------------------------------------------------------

/// `pounce-studio-core`'s analysis of the last solve's report: `summary`,
/// `findings` (each with its glossary entry: meaning, what to try, citation
/// keys), `stalls`, `restoration_windows`, and the column-oriented `trace`.
/// `null` when no solve has completed since the model was loaded.
#[unsafe(no_mangle)]
pub extern "C" fn pounce_analyze() -> *mut u8 {
    guarded("analyze", || {
        let report = LAST_SOLVE.with(|slot| slot.borrow().as_ref().map(|o| o.report.to_string()));
        let Some(report) = report else {
            return to_payload("null");
        };
        match pounce_studio_core::SolveReport::from_json_str(&report) {
            Ok(r) => to_payload(&analyze(&r).to_string()),
            Err(e) => error_json(format!("solve report did not load: {e}")),
        }
    })
}

fn analyze(r: &pounce_studio_core::SolveReport) -> serde_json::Value {
    use pounce_studio_core::{analysis, glossary};
    let findings: Vec<serde_json::Value> = analysis::diagnose(r)
        .into_iter()
        .map(|f| {
            serde_json::json!({
                "severity": f.severity,
                "code": f.code,
                "message": f.message,
                "glossary": glossary::finding(f.code),
            })
        })
        .collect();
    serde_json::json!({
        "summary": analysis::summarize(r),
        "findings": findings,
        "stalls": analysis::find_stalls(r),
        "restoration_windows": analysis::restoration_windows(r),
        "trace": analysis::convergence_trace(r),
    })
}

/// The studio-core glossary, for tooltips: `columns` (iteration-log column →
/// definition, typical range, what an abnormal value means), `findings`
/// (finding code → meaning, what to try), and `citations`.
#[unsafe(no_mangle)]
pub extern "C" fn pounce_glossary() -> *mut u8 {
    use pounce_studio_core::glossary;
    guarded("glossary", || {
        let columns: serde_json::Map<String, serde_json::Value> = glossary::all_columns()
            .into_iter()
            .map(|k| (k.to_string(), serde_json::json!(glossary::column(k))))
            .collect();
        let findings: serde_json::Map<String, serde_json::Value> = glossary::all_findings()
            .into_iter()
            .map(|k| (k.to_string(), serde_json::json!(glossary::finding(k))))
            .collect();
        let v = serde_json::json!({
            "columns": columns,
            "findings": findings,
            "citations": glossary::all_citations(),
        });
        to_payload(&v.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_NL: &str = include_str!("../tests/simple.nl");

    fn take(ptr: *mut u8) -> serde_json::Value {
        assert!(!ptr.is_null());
        // SAFETY: a payload this module just returned.
        let text = unsafe {
            let len = std::ptr::read_unaligned(ptr.cast::<u32>()) as usize;
            String::from_utf8_lossy(std::slice::from_raw_parts(ptr.add(4), len)).into_owned()
        };
        unsafe { crate::pounce_free_payload(ptr) };
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn analysis_reads_the_report_the_solve_wrote() {
        let loaded = unsafe {
            crate::pounce_load(
                SIMPLE_NL.as_ptr(),
                SIMPLE_NL.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
            )
        };
        take(loaded);
        assert!(take(pounce_analyze()).is_null(), "no solve yet");
        let result = take(unsafe { crate::pounce_solve(std::ptr::null(), 0) });
        let a = take(pounce_analyze());
        assert_eq!(a["summary"]["status"], result["status"]);
        assert_eq!(a["summary"]["iteration_count"], result["iterations"]);
        let converged = a["findings"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["code"] == "converged")
            .expect("a clean solve reports `converged`");
        assert!(converged["glossary"]["meaning"].is_string(), "{converged}");
        assert_eq!(
            a["trace"]["iter"].as_array().unwrap().len() as u64,
            result["iterations"].as_u64().unwrap() + 1
        );
    }

    #[test]
    fn the_glossary_explains_every_column_and_finding() {
        let g = take(pounce_glossary());
        let cols = g["columns"].as_object().unwrap();
        assert!(
            cols.contains_key("inf_pr") && cols.contains_key("mu"),
            "{:?}",
            cols.keys()
        );
        assert!(cols.values().all(|c| c["definition"].is_string()));
        let findings = g["findings"].as_object().unwrap();
        assert!(findings.contains_key("max_iter_exceeded"));
        assert!(findings.values().all(|f| f["what_to_try"].is_string()));
        assert!(!g["citations"].as_array().unwrap().is_empty());
    }
}
