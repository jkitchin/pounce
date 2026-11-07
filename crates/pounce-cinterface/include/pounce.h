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
#define POUNCE_VERSION_MINOR 16
#define POUNCE_VERSION_PATCH 0
#define POUNCE_VERSION "0.16.0"

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
 *  effect.
 *
 *  `x_scaling` is applied as a change of variables, so the solution
 *  and bound multipliers IpoptSolve writes back are in your own units.
 *  A factor that is not finite and positive is refused by IpoptSolve,
 *  which returns Invalid_Option and prints why. */
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
 *  ApplicationReturnStatus::User_Requested_Stop.
 *
 *  Fires from the outer loop with `alg_mod = 0` (RegularMode) and from
 *  the feasibility-restoration inner solver with `alg_mod = 1`
 *  (RestorationPhaseMode). Two things about the restoration fires:
 *
 *   - The scalars beside `alg_mod` (`obj_value`, `inf_pr`, `inf_du`,
 *     `mu`, `d_norm`, `regularization_size`, `alpha_*`, `ls_trials`)
 *     describe the min-||c||_1 feasibility subproblem, not your NLP.
 *     `alg_mod` is what tells them apart; do not plot them on one axis
 *     without checking it.
 *   - The GetIpoptCurrent* inspectors report no data for the duration.
 *     The restoration iterate is not a point of your problem, and does
 *     not even have its dimensions.
 *
 *  Returning false from a restoration fire ends the solve at the last
 *  iterate accepted for your NLP, not at the subproblem's iterate — so
 *  a caller aborting on a deadline gets back a point it can use. */
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
 *   mult_g    [in/out] constraint multipliers λ (length m), or NULL
 *   mult_x_L  [in/out] lower-bound multipliers z_L (length n), or NULL
 *   mult_x_U  [in/out] upper-bound multipliers z_U (length n), or NULL
 *   user_data forwarded unmodified to every callback
 *
 * The three multiplier buffers follow upstream Ipopt's IpoptSolve: they
 * always receive the final multipliers, and under
 * `warm_start_init_point=yes` they are also READ as the starting duals
 * (Ipopt convention, L = f + λ'g, z_L, z_U >= 0 — exactly what a previous
 * IpoptSolve wrote back, so feed its output in unchanged). With the
 * option off (the default) they are never read and may be uninitialized.
 * Under a warm start, initialize them or pass NULL: a NULL buffer means
 * "not supplied" and that block starts unseeded (bound multipliers fall
 * back to `bound_mult_init_val`, λ to zero) — upstream Ipopt instead
 * rejects a NULL here.
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
 * any output buffer to skip retrieving it.
 *
 * Both are live: pounce's algorithm core installs the context these
 * read from around every intermediate-callback invocation. Outside
 * one — before the first iteration, after IpoptSolve returns, or from
 * another thread — they return false and write nothing, which is the
 * upstream contract.
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

/**
 * Restoration-phase activity in the most recent solve. Any pointer may
 * be NULL to skip that component.
 *
 * `calls` is how many times restoration was entered, `inner_iters` the
 * total iterations its inner solver ran, `outer_iters` the outer
 * iterations driving a restoration trial step, and `wall_secs` the
 * cumulative seconds spent there — enough to answer "did this solve
 * struggle, and how much of it was restoration?".
 *
 * This is solve-level. Individual iterations can be labelled too — the
 * intermediate callback fires from the restoration inner solver with
 * `alg_mod = 1` (`RestorationPhaseMode`); see `SetIntermediateCallback`
 * for what those fires do and do not carry. These counters remain the
 * only source for the inner iteration count and the wall time, and the
 * only way to ask the question without installing a callback.
 */
void GetPounceRestorationStats(
    IpoptProblem ipopt_problem,
    ipindex*     calls,
    ipindex*     inner_iters,
    ipindex*     outer_iters,
    ipnumber*    wall_secs);

/**
 * Finite-difference Hessian census for the most recent solve. Any
 * pointer may be NULL to skip that component.
 *
 * Only `hessian_approximation=finite-difference` populates these. Every
 * other mode leaves `pattern_used` at -1 and the counts at 0, so -1 is
 * how you tell "the mode did not run" from "it ran and the pattern was
 * empty".
 *
 * `pattern_used` is 0 for the TNLP's declared Hessian structure and 1
 * for the pattern derived from the Jacobian. It reports what the solve
 * ENDED UP with, not what was requested: `fd_hessian_pattern=declared`
 * falls back to the Jacobian derivation whenever the TNLP declares no
 * Hessian structure, and that fallback is the difference between 17 and
 * 341 probe groups on benchmarks/large_scale laptime -- which is the
 * question this call exists to answer.
 *
 * `nnz` is the coloured pattern's lower-triangle nonzero count, `n` the
 * columns the colouring ran over (so `groups / n` is the fraction of a
 * dense scheme's probes this pattern costs), `groups`
 * the probe groups per Hessian (each one extra gradient and Jacobian
 * evaluation per rebuild), and `rho_max` the pattern's widest row.
 * `coloring_fell_back` is 1 when a requested star colouring failed
 * validation and Curtis-Powell-Reid was substituted.
 *
 * `objective_clique_widened` is 1 when the objective clique fell back to
 * a conservative structural set because the model stated no objective
 * linearity. It is the field that explains a surprising `groups`: the
 * clique is then the nonlinear-variable set, or every variable, and the
 * probe count reflects that rather than the objective's true support. A
 * model implementing get_objective_variables_linearity pays none of it.
 */
void GetPounceFdHessianStats(
    IpoptProblem ipopt_problem,
    ipindex*     pattern_used,
    ipindex*     nnz,
    ipindex*     n,
    ipindex*     groups,
    ipindex*     rho_max,
    ipindex*     coloring_fell_back,
    ipindex*     objective_clique_widened);

/* -----------------------------------------------------------------
 * Pounce extensions — linear-solver post-mortem
 *
 * What the KKT linear solver did during the most recent solve. The
 * `linear_solver` option selects the backend (`feral`, pounce's own
 * sparse LDL^T, is the default); `solver_name` reports the one that
 * actually ran, which is the reliable way to confirm it.
 *
 * The struct is versioned with the library, not independently: a
 * caller must compile against the pounce.h that ships with the
 * libpounce_cinterface it loads. Fields pounce collects unconditionally
 * are always set; the optional ones carry the sentinels named below
 * when the backend did not report them.
 * ----------------------------------------------------------------- */

typedef struct {
    /** Backend that ran, e.g. "feral". NUL-terminated, truncated to
     *  fit; empty when no backend reported a summary. */
    char     solver_name[32];
    /** factor() calls completed over the solve. */
    ipindex  n_factors;
    /** Of `n_factors`, how many reused the previous symbolic
     *  factorization (sparsity pattern unchanged). */
    ipindex  n_pattern_reuse;
    /** Of `n_factors`, how many needed a fresh symbolic factorization. */
    ipindex  n_pattern_changes;
    /** Maximum nnz(L)/nnz(A) over the solve; NaN when unreported. */
    ipnumber max_fill_ratio;
    /** Smallest |pivot| seen over the solve; NaN when unreported. */
    ipnumber min_abs_pivot;
    /** Largest |pivot| seen over the solve; NaN when unreported. */
    ipnumber max_abs_pivot;
    /** Inertia of the final factorization; each -1 when unreported. */
    ipindex  last_inertia_positive;
    ipindex  last_inertia_negative;
    ipindex  last_inertia_zero;
    /** nnz of the final factorization's matrix / factor; -1 when
     *  unreported. */
    ipindex  last_nnz_a;
    ipindex  last_nnz_l;
} PounceLinearSolverStats;

/**
 * Fill `stats` with the linear-solver post-mortem of the most recent
 * solve. Returns false — leaving `stats` untouched — when the problem
 * has not been solved yet or the backend reported no summary.
 *
 * Timings (symbolic analysis, numeric factorization, back-solve) are
 * deliberately absent: pounce does not instrument them today, and this
 * struct reports only what it already collects.
 */
Bool GetPounceLinearSolverStats(
    IpoptProblem             ipopt_problem,
    PounceLinearSolverStats* stats);

/* -----------------------------------------------------------------
 * Pounce extensions — option introspection
 * ----------------------------------------------------------------- */

/** Value type of a registered option, as reported by
 *  GetPounceOptionType. */
enum PounceOptionType {
    POUNCE_OPTION_UNKNOWN = 0, /**< not a registered option */
    POUNCE_OPTION_NUMBER  = 1, /**< AddIpoptNumOption */
    POUNCE_OPTION_INTEGER = 2, /**< AddIpoptIntOption */
    POUNCE_OPTION_STRING  = 3  /**< AddIpoptStrOption */
};

/**
 * Which AddIpopt*Option a keyword expects, so a caller forwarding
 * options from a differently-typed source (a scripting language's
 * dictionary, say) can pick the right setter instead of guessing from
 * the value it happens to hold. Returns POUNCE_OPTION_UNKNOWN for a
 * keyword this build does not register — which also answers "is this
 * option available here?".
 *
 * `ipopt_problem` may be NULL: an option's type is a property of the
 * build, not of a problem, and a caller deciding how to forward options
 * need not have created one yet.
 */
int GetPounceOptionType(IpoptProblem ipopt_problem, const char* keyword);

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
 * Statuses are indexed by YOUR row and variable numbering. (The SQP
 * works on a reordered constraint vector internally — equalities
 * first — and both entry points used to expose that ordering, so a
 * problem whose first row was an inequality and whose second was an
 * equality reported the two statuses swapped.) A fixed variable
 * (x_L == x_U) is absent from the internal problem and is reported as
 * POUNCE_WS_FIXED_OR_EQ.
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
 *
 * Status codes are validated against the problem, not merely against
 * the enum's range. POUNCE_WS_FIXED_OR_EQ asserts x_L == x_U for a
 * variable, or g_L == g_U for a row; POUNCE_WS_AT_LOWER / _AT_UPPER
 * assert that the bound being sat on is finite. Those are claims about
 * the model rather than guesses about the active set, so a false one
 * returns 0 (FALSE) instead of being accepted and acted on. (Accepting
 * them silently over-constrained the solve and returned a wrong optimum
 * on a convex program, with this function having returned TRUE.)
 *
 * Buffers are indexed by YOUR row and variable numbering, matching what
 * IpoptGetWorkingSet hands back, so the documented round-trip through
 * the two is order-preserving.
 *
 * Only the working set is supplied here. The starting iterate stays
 * the caller's: the next IpoptSolve warm-starts from the `x` buffer
 * it is passed, exactly as it would without this call. Under
 * `warm_start_init_point=yes` that solve also seeds the SQP's
 * multipliers from its `mult_g` / `mult_x_L` / `mult_x_U` buffers
 * (upstream Ipopt's in/out contract for those arguments); with the
 * option off — the default — they stay strictly outputs and the
 * multipliers start at zero.
 *
 * The staged working set is consumed by one solve. A following
 * IpoptSolve without a fresh call here cold-starts its working set.
 */
Bool IpoptSetWarmStartWorkingSet(
    IpoptProblem            ipopt_problem,
    const IpoptBoundStatus *bound_status_in,
    const IpoptConsStatus  *cons_status_in
);

/** Drop any pending warm-start working set without solving. */
Bool IpoptClearWarmStartWorkingSet(IpoptProblem ipopt_problem);

/**
 * Declare which variables enter the problem NONLINEARLY.
 *
 * The C-API face of Ipopt's TNLP::get_number_of_nonlinear_variables /
 * get_list_of_nonlinear_variables pair (which upstream exposes only to
 * C++ callers), for frontends that already know their model's
 * structure — CasADi's `pass_nonlinear_variables`, an algebraic
 * modeling language, a hand-written driver.
 *
 * The effect is confined to the LIMITED-MEMORY Hessian
 * (`hessian_approximation=limited-memory`): curvature is approximated
 * over the declared subset only, and the Hessian is exactly zero for
 * every other variable — the win on a model whose variables mostly
 * enter linearly. Exact-Hessian solves ignore the declaration, and so
 * does any solve that never calls this: the default is "all variables
 * are nonlinear", identical to previous behavior. The subset takes
 * precedence over the `num_linear_variables` option, matching Ipopt.
 *
 * `pos_nonlin_vars` holds `num_nonlin_vars` variable indices in the
 * problem's own index style (the `index_style` given to
 * CreateIpoptProblem). The subset may be arbitrary and noncontiguous;
 * order does not matter. Declaring all `n` variables is the same as
 * not calling this at all.
 *
 * Returns 1 on success, 0 — leaving any previous declaration in place
 * — on a NULL handle, a negative or oversized count, a NULL array with
 * a positive count, or an out-of-range index.
 */
Bool IpoptSetNonlinearVariables(
    IpoptProblem   ipopt_problem,
    ipindex        num_nonlin_vars,
    const ipindex *pos_nonlin_vars
);

/** Drop a declared nonlinear-variable subset (back to "all nonlinear"). */
Bool IpoptClearNonlinearVariables(IpoptProblem ipopt_problem);

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
 * Run the IPM. Same buffer contract as IpoptSolve: `x` is in/out
 * (initial guess in, solution out); `g`, `obj_val` are out-only;
 * `mult_g`, `mult_x_L`, `mult_x_U` are out, and also in (starting
 * duals) under `warm_start_init_point=yes`. All but `x` may be NULL.
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
 *
 * The solve is in natural (unscaled) units: any NLP scaling the IPM
 * applied internally (`nlp_scaling_method`) is undone, so RHS and
 * result are in the user's own units (pounce#128). The same applies
 * to IpoptSolverParametricStep and IpoptSolverReducedHessian. Use
 * IpoptSolverKktSolveScaled for the raw solver-space back-solve
 * (the pre-#128 behavior).
 */
Bool IpoptSolverKktSolve(
    IpoptSolver    solver,
    const Number  *rhs,
    Number        *lhs
);

/**
 * IpoptSolverKktSolve without the natural-units correction: the
 * back-solve runs in the solver's internal scaled space. Identical
 * to IpoptSolverKktSolve when no NLP scaling is active.
 */
Bool IpoptSolverKktSolveScaled(
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
 * Reduced Hessian `obj_scal * B K^-1 B^T` over the pinned
 * equality-constraint rows in `pin_indices` (0-based indices into
 * g(x)). Writes a dense `n_pins x n_pins` matrix in column-major
 * order to `hr_out`.
 *
 * The value is in natural (unscaled) units: any NLP scaling the IPM
 * applied (`nlp_scaling_method`) is undone before it is reported, so
 * `-inv(...)` of it is directly the parameter covariance of an
 * estimation problem (pounce#128). `obj_scal` is a plain extra
 * multiplier (pass 1.0).
 *
 * SIGN CONVENTION: this writes `-H_R`, not `H_R` (gh#937). Pin rows
 * select the `y_c` multiplier block, over which `B K^-1 B^T` is the
 * multiplier sensitivity `dlambda/dp = -d2(f*)/dp2` — which is why the
 * covariance recipe above negates. A well-posed minimum therefore
 * reports an all-negative spectrum; negate `hr_out` to read
 * curvature.
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
