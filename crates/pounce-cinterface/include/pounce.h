/* pounce.h — C API for the POUNCE nonlinear interior-point solver.
 *
 * Drop-in replacement for Ipopt 3.14's `IpStdCInterface.h`. Every
 * function name, argument list, and return code matches upstream so a
 * caller linking against libipopt can swap to libpounce_cinterface
 * without source changes.
 *
 * Quick-start (C):
 *
 *   #include "pounce.h"
 *
 *   IpoptProblem nlp = CreateIpoptProblem(
 *       n, x_L, x_U, m, g_L, g_U,
 *       nele_jac, nele_hess, 0,
 *       eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h);
 *   AddIpoptNumOption(nlp, "tol", 1e-8);
 *   AddIpoptIntOption(nlp, "max_iter", 500);
 *   enum ApplicationReturnStatus status = IpoptSolve(
 *       nlp, x, NULL, &obj, NULL, NULL, NULL, user_data);
 *   FreeIpoptProblem(nlp);
 *
 * Build & link example (macOS):
 *
 *   cargo build --release -p pounce-cinterface
 *   cc app.c -I crates/pounce-cinterface/include \
 *       -L target/release -lpounce_cinterface \
 *       -Wl,-rpath,target/release -o app
 */

#ifndef POUNCE_H
#define POUNCE_H

#include <stdbool.h>

#include "IpoptReturnCodes.h"

/* -----------------------------------------------------------------
 * Version
 * ----------------------------------------------------------------- */
#define POUNCE_VERSION_MAJOR 0
#define POUNCE_VERSION_MINOR 1
#define POUNCE_VERSION_PATCH 0
#define POUNCE_VERSION "0.1.0"

#ifdef __cplusplus
extern "C" {
#endif

/* Scalar typedefs — match upstream Ipopt for binary compatibility. */
typedef double ipnumber;
typedef int    ipindex;

/* Deprecated upstream aliases, kept for source-level compatibility. */
typedef ipnumber Number;
typedef ipindex  Index;
typedef bool     Bool;
#ifndef TRUE
#  define TRUE  1
#endif
#ifndef FALSE
#  define FALSE 0
#endif

/** Opaque handle to a pounce problem. */
struct IpoptProblemInfo;
typedef struct IpoptProblemInfo* IpoptProblem;

/** Pointer for arbitrary caller state passed to every callback. */
typedef void* UserDataPtr;

/* -----------------------------------------------------------------
 * Callback signatures (identical to Ipopt C API).
 * All callbacks return true on success, false on error.
 * `new_x` / `new_lambda` indicate whether x / lambda changed since
 * the last call.
 *
 * Jacobian / Hessian callbacks are dispatched in two modes:
 *   values == NULL  → fill iRow/jCol with the sparsity pattern
 *   values != NULL  → fill values in the same element order
 * ----------------------------------------------------------------- */

typedef bool (*Eval_F_CB)(
    ipindex     n,
    ipnumber*   x,
    bool        new_x,
    ipnumber*   obj_value,
    UserDataPtr user_data);

typedef bool (*Eval_Grad_F_CB)(
    ipindex     n,
    ipnumber*   x,
    bool        new_x,
    ipnumber*   grad_f,
    UserDataPtr user_data);

typedef bool (*Eval_G_CB)(
    ipindex     n,
    ipnumber*   x,
    bool        new_x,
    ipindex     m,
    ipnumber*   g,
    UserDataPtr user_data);

typedef bool (*Eval_Jac_G_CB)(
    ipindex     n,
    ipnumber*   x,
    bool        new_x,
    ipindex     m,
    ipindex     nele_jac,
    ipindex*    iRow,
    ipindex*    jCol,
    ipnumber*   values,
    UserDataPtr user_data);

typedef bool (*Eval_H_CB)(
    ipindex     n,
    ipnumber*   x,
    bool        new_x,
    ipnumber    obj_factor,
    ipindex     m,
    ipnumber*   lambda,
    bool        new_lambda,
    ipindex     nele_hess,
    ipindex*    iRow,
    ipindex*    jCol,
    ipnumber*   values,
    UserDataPtr user_data);

typedef bool (*Intermediate_CB)(
    ipindex     alg_mod,
    ipindex     iter_count,
    ipnumber    obj_value,
    ipnumber    inf_pr,
    ipnumber    inf_du,
    ipnumber    mu,
    ipnumber    d_norm,
    ipnumber    regularization_size,
    ipnumber    alpha_du,
    ipnumber    alpha_pr,
    ipindex     ls_trials,
    UserDataPtr user_data);

/* -----------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------- */

/** Allocate a new problem handle.
 *
 *   n           number of primal variables
 *   x_L/x_U     variable lower/upper bounds (length n; ±1e19 for ±∞)
 *   m           number of constraints
 *   g_L/g_U     constraint lower/upper bounds (length m)
 *   nele_jac    number of nonzeros in the Jacobian
 *   nele_hess   number of nonzeros in the lower-triangular Hessian
 *   index_style 0 = C (0-based indices), 1 = Fortran (1-based)
 *   eval_*      callback function pointers
 *
 * Returns NULL on invalid arguments (negative dims, missing required
 * callbacks, NULL bound pointers when the corresponding dim > 0). */
IpoptProblem CreateIpoptProblem(
    ipindex        n,
    ipnumber*      x_L,
    ipnumber*      x_U,
    ipindex        m,
    ipnumber*      g_L,
    ipnumber*      g_U,
    ipindex        nele_jac,
    ipindex        nele_hess,
    ipindex        index_style,
    Eval_F_CB      eval_f,
    Eval_G_CB      eval_g,
    Eval_Grad_F_CB eval_grad_f,
    Eval_Jac_G_CB  eval_jac_g,
    Eval_H_CB      eval_h);

/** Free a problem handle. After this call the pointer is invalid. */
void FreeIpoptProblem(IpoptProblem ipopt_problem);

/* -----------------------------------------------------------------
 * Options
 * Each Add*Option function returns true on success, false if the
 * keyword is unknown or the value violates registered bounds.
 * ----------------------------------------------------------------- */

bool AddIpoptStrOption(IpoptProblem ipopt_problem, char* keyword, char* val);
bool AddIpoptNumOption(IpoptProblem ipopt_problem, char* keyword, ipnumber val);
bool AddIpoptIntOption(IpoptProblem ipopt_problem, char* keyword, ipindex val);

/** Open a file to receive solver output at `print_level`. Equivalent
 *  to setting the `output_file` and `file_print_level` options and
 *  attaching a journalist FileJournal. */
bool OpenIpoptOutputFile(
    IpoptProblem ipopt_problem,
    char*        file_name,
    int          print_level);

/** Install user-provided NLP scaling. Pass NULL for `x_scaling` or
 *  `g_scaling` to leave that axis unscaled. Set option
 *  `nlp_scaling_method = user-scaling` for the scaling to take
 *  effect. */
bool SetIpoptProblemScaling(
    IpoptProblem ipopt_problem,
    ipnumber     obj_scaling,
    ipnumber*    x_scaling,
    ipnumber*    g_scaling);

/* -----------------------------------------------------------------
 * Intermediate callback
 * ----------------------------------------------------------------- */

/** Install (or remove, with cb == NULL) a per-iteration callback.
 *  Returning false from the callback signals
 *  ApplicationReturnStatus::User_Requested_Stop. */
bool SetIntermediateCallback(
    IpoptProblem    ipopt_problem,
    Intermediate_CB intermediate_cb);

/* -----------------------------------------------------------------
 * Solve
 *
 *   problem   handle from CreateIpoptProblem()
 *   x         [in/out] initial point (length n) → primal solution
 *   g         [out]    constraint values g(x*), or NULL to skip
 *   obj_val   [out]    objective f(x*), or NULL to skip
 *   mult_g    [out]    constraint multipliers λ (length m), or NULL
 *   mult_x_L  [out]    lower-bound multipliers z_L (length n), or NULL
 *   mult_x_U  [out]    upper-bound multipliers z_U (length n), or NULL
 *   user_data forwarded unmodified to every callback
 *
 * Returns an ApplicationReturnStatus (see IpoptReturnCodes.h).
 * ----------------------------------------------------------------- */
enum ApplicationReturnStatus IpoptSolve(
    IpoptProblem ipopt_problem,
    ipnumber*    x,
    ipnumber*    g,
    ipnumber*    obj_val,
    ipnumber*    mult_g,
    ipnumber*    mult_x_L,
    ipnumber*    mult_x_U,
    UserDataPtr  user_data);

/* -----------------------------------------------------------------
 * Inspection (valid only during the intermediate callback)
 *
 * These mirror Ipopt 3.14's GetIpoptCurrent* functions. Pass NULL for
 * any output buffer to skip retrieving it. They return false until
 * pounce's algorithm core invokes the intermediate callback per
 * iteration (currently a follow-up — the signature is here so callers
 * can link against it today).
 * ----------------------------------------------------------------- */

bool GetIpoptCurrentIterate(
    IpoptProblem ipopt_problem,
    bool         scaled,
    ipindex      n,
    ipnumber*    x,
    ipnumber*    z_L,
    ipnumber*    z_U,
    ipindex      m,
    ipnumber*    g,
    ipnumber*    lambda);

bool GetIpoptCurrentViolations(
    IpoptProblem ipopt_problem,
    bool         scaled,
    ipindex      n,
    ipnumber*    x_L_violation,
    ipnumber*    x_U_violation,
    ipnumber*    compl_x_L,
    ipnumber*    compl_x_U,
    ipnumber*    grad_lag_x,
    ipindex      m,
    ipnumber*    nlp_constraint_violation,
    ipnumber*    compl_g);

/* -----------------------------------------------------------------
 * Library info
 * ----------------------------------------------------------------- */

/** Get the pounce version as `major.minor.release`. Any pointer may
 *  be NULL to skip that component. */
void GetIpoptVersion(int* major, int* minor, int* release);

/* -----------------------------------------------------------------
 * Pounce extensions — post-solve statistics
 *
 * Convenience accessors not present in upstream Ipopt's C API. All are
 * valid only after IpoptSolve() has returned; they yield zero before
 * the first solve.
 * ----------------------------------------------------------------- */

/** Number of IPM iterations in the most recent solve. */
ipindex  GetIpoptIterCount(IpoptProblem ipopt_problem);

/** Wall-clock solve time in seconds from the most recent solve. */
ipnumber GetIpoptSolveTime(IpoptProblem ipopt_problem);

/** Final primal infeasibility from the most recent solve. */
ipnumber GetIpoptPrimalInf(IpoptProblem ipopt_problem);

/** Final dual infeasibility from the most recent solve. */
ipnumber GetIpoptDualInf(IpoptProblem ipopt_problem);

/** Final complementarity error from the most recent solve. */
ipnumber GetIpoptComplInf(IpoptProblem ipopt_problem);

/* -----------------------------------------------------------------
 * Pounce extensions — active-set SQP working-set warm start
 *
 * Phase 5c (§7.2 of docs/research/active-set-sqp-warm-start.md).
 * These functions are only meaningful when the `algorithm` option
 * has been set to "active-set-sqp" via AddIpoptStrOption.
 *
 * Status enum values are stable across versions:
 *   0 = Inactive,       1 = AtLower (active at lower bound),
 *   2 = AtUpper,        3 = Fixed (variables) or Equality (constraints).
 * ----------------------------------------------------------------- */

typedef int IpoptBoundStatus;
typedef int IpoptConsStatus;

#define POUNCE_WS_INACTIVE     0
#define POUNCE_WS_AT_LOWER     1
#define POUNCE_WS_AT_UPPER     2
#define POUNCE_WS_FIXED_OR_EQ  3

/**
 * Retrieve the QP working set produced by the most recent SQP solve.
 * Pass NULL for either output buffer to skip that side; otherwise
 * `bound_status_out` must hold at least `n` ints and
 * `cons_status_out` at least `m` ints.
 *
 * Returns 1 (TRUE) on success, 0 (FALSE) when no working set is
 * available — e.g. no SQP solve has been run, the IPM path was
 * used, or the SQP solve converged at iter 0 (no QP solved).
 */
Bool IpoptGetWorkingSet(
    IpoptProblem      ipopt_problem,
    IpoptBoundStatus *bound_status_out,
    IpoptConsStatus  *cons_status_out
);

/**
 * Supply a warm-start working set consumed by the next IpoptSolve.
 * `bound_status_in` must hold `n` valid status codes (or NULL to
 * cold-start bounds); `cons_status_in` must hold `m` valid status
 * codes (or NULL to cold-start constraints). Returns 1 on success,
 * 0 on a NULL problem handle, an out-of-range status code, or
 * both inputs NULL.
 */
Bool IpoptSetWarmStartWorkingSet(
    IpoptProblem            ipopt_problem,
    const IpoptBoundStatus *bound_status_in,
    const IpoptConsStatus  *cons_status_in
);

/** Drop any pending warm-start working set without solving. */
Bool IpoptClearWarmStartWorkingSet(IpoptProblem ipopt_problem);

/**
 * Convenience one-shot solve combining IpoptSetWarmStartWorkingSet,
 * IpoptSolve, and IpoptGetWorkingSet. Any of the in/out buffers
 * may be NULL to skip that side. Returns the
 * ApplicationReturnStatus integer (same contract as IpoptSolve).
 *
 * NOTE: an invalid warm-start input (out-of-range status code,
 * dimension mismatch) is silently discarded — the solve proceeds
 * with cold-start instead. Callers who need to detect this fall-
 * back must invoke IpoptSetWarmStartWorkingSet directly first
 * and check its Bool return value before calling IpoptSolve.
 */
enum ApplicationReturnStatus IpoptSolveWarmStart(
    IpoptProblem            ipopt_problem,
    ipnumber               *x,
    ipnumber               *g,
    ipnumber               *obj_val,
    ipnumber               *mult_g,
    ipnumber               *mult_x_L,
    ipnumber               *mult_x_U,
    const IpoptBoundStatus *bound_status_in,
    const IpoptConsStatus  *cons_status_in,
    IpoptBoundStatus       *bound_status_out,
    IpoptConsStatus        *cons_status_out,
    UserDataPtr             user_data
);

/* -----------------------------------------------------------------
 * Pounce extensions — JSON solve-report writing
 *
 * The CLI's `--json-output` path writes a `pounce.solve-report/v1`
 * file. These two functions expose the same payload to embedders
 * (GAMS solver link, custom drivers) so that downstream tools — the
 * studio MCP server's `diagnose`, `find_stalls`, `convergence_trace`,
 * etc. — work against any cinterface-driven solve.
 * ----------------------------------------------------------------- */

/**
 * Enable per-iteration trajectory capture on the next IpoptSolve.
 * Must be called BEFORE IpoptSolve for the per-iter trace to land in
 * the report; off by default to avoid the (small) per-iter cost.
 *
 * Returns 1 (TRUE) on success, 0 (FALSE) when ipopt_problem is NULL.
 */
Bool IpoptEnableIterHistory(IpoptProblem ipopt_problem);

/**
 * Write the most recent IpoptSolve result to `path` as a
 * `pounce.solve-report/v1` JSON file. `detail` is `"summary"` or
 * `"full"`; pass NULL for the default ("summary"). At
 * `"full"`, the per-iteration trajectory is included when
 * IpoptEnableIterHistory was called before the solve.
 *
 * The `kind` of the input descriptor is recorded as `"tnlp-direct"`
 * because the C API receives callbacks rather than an .nl file or
 * builtin name.
 *
 * Returns 1 (TRUE) on a successful write, 0 (FALSE) for: NULL handle,
 * no prior solve, an invalid `detail`, a bad path, or an I/O error.
 */
Bool IpoptWriteSolveReport(
    IpoptProblem  ipopt_problem,
    const char   *path,
    const char   *detail
);

/* ===========================================================
 * Factor-once / solve-many session API
 *
 * The Solver session keeps the converged KKT factor alive
 * between calls so several follow-up operations (parametric
 * sensitivity sweep, reduced Hessian over different pin
 * sets, raw KKT back-solve) reuse the same factorization.
 *
 *   IpoptProblem prob = CreateIpoptProblem(...);
 *   // ... AddIpoptStrOption / SetIntermediateCallback as usual
 *   IpoptSolver sol = IpoptCreateSolver(&prob);       // consumes prob
 *   IpoptSolverSolve(sol, x, g, &obj, ...);            // run the IPM
 *   IpoptSolverParametricStep(sol, n_pins, pins, deltas, dx);
 *   IpoptSolverReducedHessian(sol, n_pins, pins, 1.0, hr);
 *   IpoptSolverKktSolve(sol, rhs, lhs);
 *   IpoptFreeSolver(sol);
 *
 * The classic `IpoptSolve` API is unchanged and unaffected.
 * =========================================================== */

/** Opaque solver-session handle. */
typedef struct IpoptSolverInfo* IpoptSolver;

/**
 * Construct a session from a prepared IpoptProblem. Consumes
 * `*prob_handle` (the inner pointer is nulled out so the caller
 * cannot accidentally double-free). Returns NULL on a NULL/empty
 * input handle.
 */
IpoptSolver IpoptCreateSolver(IpoptProblem* prob_handle);

/** Release a session handle. After this call the pointer is invalid. */
void IpoptFreeSolver(IpoptSolver solver);

/**
 * Run the IPM. Same output buffer contract as IpoptSolve: `x` is
 * in/out (initial guess in, solution out); `g`, `obj_val`,
 * `mult_g`, `mult_x_L`, `mult_x_U` are out-only and may be NULL.
 * `user_data` is threaded into the C callbacks unchanged.
 */
enum ApplicationReturnStatus IpoptSolverSolve(
    IpoptSolver  solver,
    Number      *x,
    Number      *g,
    Number      *obj_val,
    Number      *mult_g,
    Number      *mult_x_L,
    Number      *mult_x_U,
    UserDataPtr  user_data
);

/**
 * Dimension of the augmented KKT system held by the session.
 * Returns -1 on a NULL handle or before a successful Solve.
 */
Index IpoptSolverGetKktDim(IpoptSolver solver);

/**
 * Apply the converged KKT factor to `rhs` (length = KKT dim).
 * Writes the result into `lhs`. Returns 1 (TRUE) on success,
 * 0 (FALSE) on NULL inputs or absent factor.
 */
Bool IpoptSolverKktSolve(
    IpoptSolver    solver,
    const Number  *rhs,
    Number        *lhs
);

/**
 * Parametric step: given perturbations `deltas` on the constraints
 * named by `pin_indices` (length `n_pins`), write the predicted
 * primal step `dx` into `dx_out` (length n).
 */
Bool IpoptSolverParametricStep(
    IpoptSolver    solver,
    Index          n_pins,
    const Index   *pin_indices,
    const Number  *deltas,
    Number        *dx_out
);

/**
 * Reduced Hessian `H_R = obj_scal * B K^-1 B^T` over the pinned
 * equality-constraint rows in `pin_indices` (0-based indices into
 * g(x)). Writes a dense `n_pins x n_pins` matrix in column-major
 * order to `hr_out`.
 *
 * `H_R` is in natural (unscaled) units: any NLP scaling the IPM
 * applied (`nlp_scaling_method`) is undone before the value is
 * reported, so `-inv(H_R)` is directly the parameter covariance of
 * an estimation problem (pounce#128). `obj_scal` is a plain extra
 * multiplier (pass 1.0).
 */
Bool IpoptSolverReducedHessian(
    IpoptSolver    solver,
    Index          n_pins,
    const Index   *pin_indices,
    Number         obj_scal,
    Number        *hr_out
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POUNCE_H */
