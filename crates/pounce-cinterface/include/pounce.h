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
#define POUNCE_VERSION_MINOR 17
#define POUNCE_VERSION_PATCH 0
#define POUNCE_VERSION "0.17.0"

/* -----------------------------------------------------------------
 * Error codes: one namespace for every POUNCE extension
 *
 * Each value has exactly one meaning wherever it is returned, so a
 * wrapper can keep one code-to-message table (PounceErrorString is
 * that table). Generic codes are -1..-19; each function family has its
 * own block of 20. A new code is appended inside its block and a
 * number is never reused. Non-negative returns stay per function: 0 is
 * POUNCE_OK, and the QP statuses and AddPounceComplementsConstraint
 * outcomes are results, not errors.
 *
 * Check order, the same in every function, so a call with several
 * faults reports the same one whichever function it is: (1) the handle;
 * (2) flags and enum arguments; (3) counts, then the arrays they size;
 * (4) per entry over the whole batch, in batch order: range, then
 * duplicate / already paired, then NaN; (5) per entry over the whole
 * batch: what the entry would mean (shape, row not free, variable
 * bounds, Complements class, pair changed); (6) state beyond the batch
 * (working set; the QP solve's checks). An argument fault anywhere in a
 * batch is reported before a semantic fault anywhere in it, and
 * bad_index names the first entry with the earliest-stage fault.
 * ----------------------------------------------------------------- */
#define POUNCE_OK                        0
/* generic, -1..-19 */
#define POUNCE_ERR_NULL                 -1   /* NULL handle, or a NULL array the call needed */
#define POUNCE_ERR_COUNT                -2   /* count < 0, > n|m, or inconsistent with a NULL index array */
#define POUNCE_ERR_OUT_OF_RANGE         -3   /* index outside [0,n) / [0,m) (in index style) */
#define POUNCE_ERR_DUPLICATE            -4   /* index given twice where each may appear once */
#define POUNCE_ERR_NAN                  -5   /* a NaN value */
#define POUNCE_ERR_FLAGS                -6   /* flags has a bit this library does not define */
#define POUNCE_ERR_ARGUMENT             -7   /* unknown view or cone kind; bad cone dim/alpha; P entries whose sum overflows */
#define POUNCE_ERR_INTERNAL             -8   /* inconsistent handle or internal error; nothing changed */
/* complementarity pairs, -20..-39 */
#define POUNCE_ERR_VAR_ALREADY_PAIRED  -20   /* var in a pair, earlier in this batch, or in an earlier accepted call */
#define POUNCE_ERR_ROW_ALREADY_PAIRED  -21   /* row in a pair, earlier in this batch, or in an earlier accepted call */
#define POUNCE_ERR_SHAPE               -22   /* neither one-sided, one-sided+bound, nor mixed */
#define POUNCE_ERR_ROW_NOT_FREE        -23   /* a Complements row has / would get a finite bound */
#define POUNCE_ERR_VAR_BOUNDS          -24   /* var bounds NaN or crossed (l > u) */
/* bound updates, -40..-59 */
#define POUNCE_ERR_PAIR_CHANGED        -40   /* a declared pair would pair different sides */
#define POUNCE_ERR_COMPLEMENTS_CLASS   -41   /* a Complements variable would change table row */
#define POUNCE_ERR_WORKING_SET         -42   /* the staged working set would be contradicted */
/* QP handle, -60..-79 */
#define POUNCE_ERR_INCOMPLETE          -60   /* no objective, or declared rows never set */
#define POUNCE_ERR_CONES               -61   /* cones do not partition the m_ineq rows; PSD mixed with exp/pow */
#define POUNCE_ERR_OPTIONS             -62   /* a switch the engine would drop; with cones: active-set, tau, tau_max, qp_*, hessian_inertia other than auto */
#define POUNCE_ERR_INDEFINITE          -63   /* method=ipm and P not certified PSD */

#ifdef __cplusplus
extern "C" {
#endif

/** The message for a POUNCE_OK / POUNCE_ERR_* code: a static string the
 *  caller must not free, "unknown error code" for any other value, never
 *  NULL. Function-specific non-negative results read as unknown. */
const char* PounceErrorString(int code);

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
 *  which returns Invalid_Option and prints why.
 *
 *  With complementarity pairs declared, every lowered solve uses these
 *  factors for your rows and variables; the rows the lowering adds (one
 *  product row per pair) and any lifted variables are unscaled. */
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
 *  With complementarity pairs declared, the callback fires from every
 *  lowered solve the driver runs (each starts again at iter_count 0), and
 *  its scalars describe the lowered model. The GetIpoptCurrent*
 *  inspectors report no data during such a solve, whatever is requested.
 *  Returning false ends the whole run: the driver starts no further solve
 *  and IpoptSolve returns User_Requested_Stop.
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
 *
 * Complementarity pairs (AddPounceComplementarityPair and friends): with
 * pairs declared, IpoptSolve hands the model to POUNCE's complementarity
 * driver, as the CLI and Python do. The driver runs a sequence of solves of
 * a lowered model (a product row per pair, possibly lifted variables),
 * certifies the answer, and reports one point; that point, its g, objective
 * and multipliers are what come back in the buffers above, and the status
 * is that solve's. GetIpoptIterCount and the other post-solve accessors
 * describe the same reported solve, not the whole run. The
 * complementarity_strategy, complementarity_continuation and
 * complementarity_lift options steer the driver. The second-opinion ladder
 * does not run after it. With no pairs declared nothing here changes.
 * A staged SQP working set (IpoptSetWarmStartWorkingSet) cannot seed a
 * solve with pairs: IpoptSolve then returns Invalid_Problem_Definition
 * before any callback runs, and leaves both the working set and the pairs
 * in place (clear one of them).
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
 *
 * During a solve with complementarity pairs declared they also return
 * false and write nothing, for every request shape: the iterate is the
 * lowered model's, not a point of your problem's shape. This holds per
 * solve, not per thread: a pair-free IpoptSolve of another problem
 * started from inside that solve's callback has working inspectors for
 * its own duration.
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
 * used, the SQP solve converged at iter 0 (no QP solved), the
 * last solve had complementarity pairs (its solves were of a lowered
 * model with more rows), pairs are declared now, or the most recent
 * IpoptSolve did not complete (refused, or failed before it finished).
 * Nothing is written when it returns 0.
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
 *
 * Not with complementarity pairs: an IpoptSolve on a problem with both a
 * staged working set and declared pairs returns Invalid_Problem_Definition
 * without solving and consumes neither (the working set is over your rows;
 * the driver solves a lowered model with more). The same holds for
 * IpoptSolveWarmStart with an input working set.
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

/* ---------------------------------------------------------------------
 * Declared complementarity pairs (gh#1147)
 *
 * A pair (row, var) declares   0 <= x_var - x_L[var]  _|_  g_row(x) - g_L[row] >= 0
 * (or the upper-bound orientation of either side), in POUNCE's native
 * shapes, exactly as Python's Problem(complementarity=) and the .nl
 * reader accept them:
 *   one-sided           the variable and the row each have exactly one
 *                       finite bound;
 *   one-sided + bound   the row has one finite side, the variable is boxed
 *                       (the row's side picks the bound in the pair);
 *   mixed               the variable is boxed with x_L < x_U, the row free.
 * Anything else (a boxed or equality row, both sides free, a fixed
 * variable with a free row, ...) is refused with POUNCE_ERR_SHAPE.
 * Bounds at or beyond +-1e19 are infinite. See docs/src/complementarity.md.
 *
 * Indices are in the problem's index style (the index_style given to
 * CreateIpoptProblem), like IpoptSetNonlinearVariables. Each declaration
 * is validated at once, against the handle's bounds (those given to
 * CreateIpoptProblem, as AddPounceComplementsConstraint may have edited
 * them) and the pairs already declared, by the same classifier the
 * solve's lowering runs; a refused declaration leaves the declared set
 * unchanged. Pairs persist on the handle across solves.
 *
 * IpoptSolve solves a problem with pairs through the complementarity
 * driver (see IpoptSolve). Sessions (IpoptCreateSolver) and SQP
 * working-set warm starts refuse pairs; the live inspectors report no
 * data during such a solve. The Fortran shim declares no pairs.
 *
 * The driver's path through its lowered solves is sensitive to how the
 * model is written: a row's constant in g (g = 2y - 1 - x, g_L = 0) or in
 * its bound (g = 2y - x, g_L = 1) is the same feasible set but can take a
 * different number of iterations. Under hessian_approximation=limited-memory
 * declare the nonlinear variables (IpoptSetNonlinearVariables) as an .nl
 * file's header does, or the run differs from the CLI's on the same model.
 * See docs/src/c-api.md, "Matching the CLI".
 * examples/complementarity.c uses every function below.
 * ------------------------------------------------------------------- */

/** Append one pair. Returns POUNCE_OK or a POUNCE_ERR_* code. */
int AddPounceComplementarityPair(
    IpoptProblem ipopt_problem,
    ipindex      row,
    ipindex      var
);

/**
 * Replace the declared pairs with (rows[k], vars[k]), k < n_pairs. All or
 * nothing: the batch is validated on its own (not against the set it
 * replaces) and stored only if every pair passes. n_pairs == 0 clears the
 * set (rows/vars may then be NULL). `bad_index` may be NULL; otherwise it
 * receives the batch position of the first refused pair (the second
 * occurrence, for a duplicate), or -1 on success, POUNCE_ERR_NULL,
 * POUNCE_ERR_FLAGS and POUNCE_ERR_COUNT. n_pairs < 0, or above min(n, m)
 * (which must repeat a variable or a row), is POUNCE_ERR_COUNT, refused
 * before rows/vars are read. Range and already-paired checks
 * run over the whole batch before any shape check (the shared check order
 * above), so an argument fault is reported before a shape fault anywhere
 * in the batch. `flags` must be 0: no bit is
 * defined yet, and an unknown bit is POUNCE_ERR_FLAGS (checked after the
 * handle, before the count), so a later relaxation is opt-in and an older
 * library never ignores one.
 */
int SetPounceComplementarityPairs(
    IpoptProblem   ipopt_problem,
    ipindex        n_pairs,
    const ipindex *rows,
    const ipindex *vars,
    unsigned int   flags,
    ipindex       *bad_index
);

/** Drop every declared pair. FALSE only on a NULL handle. Bounds written
 *  by AddPounceComplementsConstraint stay written, and the outcome record of
 *  the most recent solve (GetPounceComplementarityStats) stays readable
 *  until the next solve. */
Bool ClearPounceComplementarityPairs(IpoptProblem ipopt_problem);

/** Number of declared pairs; -1 on a NULL handle. */
ipindex GetPounceComplementarityPairCount(IpoptProblem ipopt_problem);

/**
 * Copy the declared pairs, in declaration order and in the problem's index
 * style, into `rows` and `vars` (each sized by
 * GetPounceComplementarityPairCount). FALSE, writing nothing, on a NULL
 * handle or a NULL array while pairs are declared; with no pairs, TRUE and
 * the arrays may be NULL.
 */
Bool GetPounceComplementarityPairs(
    IpoptProblem ipopt_problem,
    ipindex     *rows,
    ipindex     *vars
);

/* ---------------------------------------------------------------------
 * MOI / JuMP [F, x] in Complements (gh#1147)
 *
 * AddPounceComplementsConstraint(problem, row, var) takes MOI's semantics
 * -- F = g_row is a free function, and the sign it must have is read off
 * x_var's bounds -- and transcribes it onto the native pairs above by
 * EDITING THE HANDLE'S COPY OF THE ROW BOUNDS, then declaring the native
 * pair. The row must be free (g_L <= -1e19 and g_U >= 1e19). The return
 * code says what was done (bounds at or beyond +-1e19 are infinite):
 *
 *   x bounds        row edit               pair          returns
 *   l < u, finite   none                   native mixed  POUNCE_COMPLEMENTS_PAIR_MIXED
 *   l only          g_L = 0     (F >= 0)   one-sided     POUNCE_COMPLEMENTS_PAIR_LOWER
 *   u only          g_U = 0     (F <= 0)   one-sided     POUNCE_COMPLEMENTS_PAIR_UPPER
 *   free            g_L = g_U = 0 (F = 0)  none          POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR
 *   l == u          none        (F free)   none          POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR
 *
 * Checks, in order: NULL handle; row/var in range (index style); the
 * variable, then the row, not already used -- by a declared pair or by an
 * earlier accepted call of this function, pair or not, so one F or one x
 * cannot be in two Complements constraints; row free; x's bounds neither
 * NaN nor crossed. A refused call changes nothing. The edits are permanent:
 * ClearPounceComplementarityPairs / SetPounceComplementarityPairs drop
 * pairs, not edits, and do not free a row or variable for another call.
 * So after a clear the handle solves a plain NLP with the edited row
 * bounds -- neither the original model nor the complementarity one. The
 * only way back to the original bounds is FreeIpoptProblem and
 * CreateIpoptProblem again. GetPounceBounds reads the bounds before the
 * edit (VIEW_USER) and after it (VIEW_EFFECTIVE); SetPounceBounds can move
 * x's bounds within its table row and keeps the edit in step.
 * A row that already has a side is a native declaration: use
 * AddPounceComplementarityPair. GetPounceComplementarityPairs lists the
 * pairs declared here like any other.
 * ------------------------------------------------------------------- */
#define POUNCE_COMPLEMENTS_PAIR_MIXED              0
#define POUNCE_COMPLEMENTS_PAIR_LOWER              1
#define POUNCE_COMPLEMENTS_PAIR_UPPER              2
#define POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR       3
#define POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR       4

/** Declare [F, x] in Complements for (row, var); see the table above. A
 *  non-negative return is success, a negative one a refusal. */
int AddPounceComplementsConstraint(
    IpoptProblem ipopt_problem,
    ipindex      row,
    ipindex      var
);

/* ---------------------------------------------------------------------
 * In-place bound updates (gh#1147 Stage 2)
 *
 * SetPounceBounds changes variable and constraint bounds on an existing
 * handle, keeping everything else on it: options, scaling, the
 * nonlinear-variables hint, declared pairs and Complements declarations.
 * It saves no solver work -- every IpoptSolve builds its solver afresh --
 * only the Free + Create + replay of that state. Warm starts stay explicit:
 * pass the previous x and multipliers to IpoptSolve, with
 * warm_start_init_point=yes.
 *
 * Sparse: (x_idx[k], x_L[k], x_U[k]) for k < nx and (g_idx[k], g_L[k],
 * g_U[k]) for k < ng, indices in the problem's index style. A NULL index
 * array with the full count (n or m) is dense; with any other nonzero
 * count it is POUNCE_ERR_COUNT, as is a count above n (or m), which
 * must repeat an index. A NULL value array leaves that side
 * of the named entries unchanged. Cost: O(nx + ng + pairs touched),
 * independent of n, m and the number of pairs.
 *
 * flags must be 0 (both functions). No bit is defined yet; it is there so
 * that a later relaxation of a refusal below can be asked for, rather than
 * changing what an existing call means. A bit this library does not define
 * is POUNCE_ERR_FLAGS, so an older library never ignores one.
 *
 * All or nothing. The batch is checked against the bounds it would
 * install, in this order, and a refused call changes nothing:
 *   1. the handle, then flags (and GetPounceBounds' view), then counts; then per entry, variables first, in batch order: index in
 *      range, not already in the batch (POUNCE_ERR_DUPLICATE), no NaN;
 *   2. per entry in batch order, for a variable or row in a declared pair
 *      or a Complements declaration:
 *      - a Complements variable keeps its row of the MOI table (lower-only,
 *        upper-only, boxed l < u, fixed, free), else
 *        POUNCE_ERR_COMPLEMENTS_CLASS;
 *      - a Complements row stays free in the caller's bounds, else
 *        POUNCE_ERR_ROW_NOT_FREE;
 *      - a declared pair keeps a native shape that pairs the same
 *        variable bound (lower/upper) with the same row side (lower/upper),
 *        or stays mixed, else POUNCE_ERR_PAIR_CHANGED. Values may
 *        move, and the variable may gain or lose the bound that is not in
 *        the pair. (A one-sided pair x >= l _|_ g <= u whose x gains an
 *        upper bound would re-classify as pairing g <= u with x <= x_U: a
 *        different condition, so it is refused.)
 *   3. a working set staged by IpoptSetWarmStartWorkingSet stays true at
 *      the entries touched, else POUNCE_ERR_WORKING_SET.
 * Crossed bounds (l > u) are not refused here; the solve reports them, as
 * for CreateIpoptProblem. To change what a pair or a Complements
 * declaration means, build a new problem.
 *
 * bad_index (may be NULL) receives the position of the refused entry in
 * the batch -- 0..nx-1 for variables, nx..nx+ng-1 for rows -- or -1.
 * A pair is checked once, against everything the batch does to it, and
 * PAIR_CHANGED is charged to the batch's first entry naming its variable
 * or row (variables come first), not necessarily the one that changed it.
 *
 * Two views. The USER bounds are what CreateIpoptProblem and
 * SetPounceBounds wrote. The EFFECTIVE bounds are what the solve (and
 * every other function) sees: the user bounds with the row edits
 * AddPounceComplementsConstraint made. They differ only on those rows, and
 * SetPounceBounds writes the user view and re-derives the edit.
 * GetPounceBounds reads either; it takes the same index arguments
 * (duplicates allowed), writes nothing on a refusal, and skips a NULL
 * output array.
 *
 * The outcome of the most recent solve stays readable after an update; it
 * describes that solve. Sessions (IpoptCreateSolver) own their problem and
 * take no updates.
 * ------------------------------------------------------------------- */
#define POUNCE_BOUNDS_VIEW_USER                0
#define POUNCE_BOUNDS_VIEW_EFFECTIVE           1

/** Update bounds in place; see above. Returns POUNCE_OK or a
 *  POUNCE_ERR_* code. */
int SetPounceBounds(
    IpoptProblem    ipopt_problem,
    ipindex         nx,
    const ipindex  *x_idx,
    const ipnumber *x_L,
    const ipnumber *x_U,
    ipindex         ng,
    const ipindex  *g_idx,
    const ipnumber *g_L,
    const ipnumber *g_U,
    unsigned int    flags,
    ipindex        *bad_index
);

/** Read bounds in `view` (POUNCE_BOUNDS_VIEW_USER or _EFFECTIVE); see
 *  above. Returns POUNCE_OK or a POUNCE_ERR_* code. */
int GetPounceBounds(
    IpoptProblem    ipopt_problem,
    ipindex         nx,
    const ipindex  *x_idx,
    ipnumber       *x_L,
    ipnumber       *x_U,
    ipindex         ng,
    const ipindex  *g_idx,
    ipnumber       *g_L,
    ipnumber       *g_U,
    int             view,
    unsigned int    flags,
    ipindex        *bad_index
);

/* ---------------------------------------------------------------------
 * Extensible output structs (gh#1147)
 *
 * PounceComplementarityStats, PounceComplementaritySolve,
 * PounceQpResiduals and PounceQpCertificate begin with
 *
 *     ipindex struct_size;
 *
 * which the caller sets to sizeof(the struct) before the call:
 *
 *     PounceQpResiduals r;
 *     r.struct_size = sizeof r;
 *     if (PounceQpGetResiduals(qp, &r)) ...
 *
 * A struct_size smaller than this release's sizeof (0 from "= {0}", or an
 * uninitialized value) is refused: the getter returns FALSE and writes
 * nothing. Otherwise the getter writes this release's fields only: it
 * never writes struct_size, nor any byte past this release's sizeof.
 *
 * Growth rule: a later release only appends fields at the end (nothing is
 * removed, reordered or retyped), and writes min(struct_size, its sizeof)
 * bytes. A program built against this header keeps working with a newer
 * library, which honours its smaller struct_size; a program built
 * against a newer header gets this release's fields from this library,
 * the rest of its buffer untouched.
 *
 * PounceLinearSolverStats has no struct_size: it shipped before this
 * convention, and adding one would change its layout under existing
 * callers. Its layout is frozen.
 * ------------------------------------------------------------------- */

/* ---------------------------------------------------------------------
 * Complementarity outcome (gh#1147)
 *
 * What the complementarity driver did on the most recent IpoptSolve: the
 * same fields, through the same encodings, as the solve report's
 * "complementarity" block and Python's info["complementarity"]. The
 * getters answer "nothing" (FALSE, or -1 for the text getter) when the
 * most recent solve did not run the driver: no solve yet, a solve without
 * declared pairs, or a solve refused or failed before the driver returned.
 * The record describes the most recent solve: clearing the pairs does not
 * erase it, the next solve replaces it.
 *
 * Sentinels instead of discriminants: a real that was not measured, or is
 * not finite (the report's null), reads NaN; a count or 0/1 flag whose
 * record is absent (no certificate, the B-check did not run) reads -1.
 * Stage labels are "exact product", "Scholtes continuation",
 * "l1 penalty", "branch NLP", "branch polish". Both structs are
 * extensible (struct_size first; see above).
 * ------------------------------------------------------------------- */
#define POUNCE_CC_CLASS_S             0   /* strongly stationary */
#define POUNCE_CC_CLASS_W             1   /* weakly stationary only */
#define POUNCE_CC_CLASS_NONE          2   /* not a stationary point of the model */
#define POUNCE_CC_CLASS_NOT_SEARCHED  3   /* not determined: multiplier set too large to search */
#define POUNCE_CC_CLASS_ABSENT       -1   /* no certificate: model not evaluable there */

#define POUNCE_CC_REPORTED_ANSWER               0   /* a formulation answered */
#define POUNCE_CC_REPORTED_WEAK_FALLBACK        1   /* a W point, when none answered */
#define POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED 2   /* converged, products in tolerance, uncertified */
#define POUNCE_CC_REPORTED_VERDICT              3   /* no feasible point: the exact-product verdict */

#define POUNCE_CC_B_VERDICT_B            0   /* B-stationary */
#define POUNCE_CC_B_VERDICT_NOT_B        1   /* a branch descends */
#define POUNCE_CC_B_VERDICT_NOT_CHECKED  2   /* not decided: see POUNCE_CC_TEXT_B_REASON */
#define POUNCE_CC_B_VERDICT_ABSENT      -1   /* the check did not run */

typedef struct {
    /** Set by the caller to sizeof(PounceComplementarityStats). */
    ipindex  struct_size;
    /** complementarity_strategy: "both", "exact" or "continuation". */
    char     strategy[16];
    /** POUNCE_CC_REPORTED_*. */
    ipindex  reported;
    /** Stage label of the formulation that answered; "" when none did. */
    char     accepted_by[32];
    /** Stage label of the weakly stationary fallback; "" when not one. */
    char     fallback[32];
    /** Lowered solves run (see GetPounceComplementaritySolve). */
    ipindex  n_solves;
    /** Iterations summed over every lowered solve. GetIpoptIterCount is
     *  the reported solve's alone. */
    ipindex  total_iterations;
    /** Reported point's complementarity violation; NaN when unmeasured. */
    ipnumber max_abs_product;
    /** POUNCE_CC_CLASS_*. The fields from n_pairs to primal_violation are
     *  the certificate's: -1 / NaN when certificate_class is ABSENT. */
    ipindex  certificate_class;
    ipindex  n_pairs;
    ipindex  n_biactive;
    /** Pairs with neither side at zero. */
    ipindex  n_neither;
    /** 0/1: the stationarity multipliers are unique (MPCC-LICQ). */
    ipindex  multipliers_unique;
    /** Stationarity residual and the band it is judged against; NaN when
     *  unmeasured. */
    ipnumber residual;
    ipnumber resid_tol;
    /** 0/1: residual within the tight floor. */
    ipindex  tight;
    /** 0/1: primal_violation within the feasibility tolerance. */
    ipindex  feasible;
    /** Largest row or bound violation; NaN when unmeasured. */
    ipnumber primal_violation;
    /** POUNCE_CC_B_VERDICT_*. The three fields after it are -1 when
     *  b_verdict is ABSENT. */
    ipindex  b_verdict;
    ipindex  b_n_biactive;
    /** Branch LPs solved. */
    ipindex  b_branches_solved;
    /** 0/1: every branch was enumerated. */
    ipindex  b_enumerated;
    /** Branch NLPs that moved the point off a non-B-stationary one. */
    ipindex  escapes;
    /** Pairs whose row was lifted into a variable (complementarity_lift). */
    ipindex  lifted;
    /** Largest min(|G|, |H|) over the pairs; NaN when unmeasured. */
    ipnumber natural_residual;
    /** 0/1: the branch polish ran; polish_taken (0/1) is meaningful only
     *  when it did, and reads 0 otherwise. */
    ipindex  polish_ran;
    ipindex  polish_taken;
    /** The polish's natural_residual before and after; NaN when it did not
     *  run or the value was not measured. */
    ipnumber polish_residual_before;
    ipnumber polish_residual_after;
    /** natural_residual above which the polish runs (10 tol). */
    ipnumber polish_threshold;
} PounceComplementarityStats;

/**
 * Fill `stats` with what the driver did on the most recent solve. FALSE,
 * leaving `stats` untouched, on a NULL argument, a stats->struct_size
 * smaller than sizeof(PounceComplementarityStats), before any solve, and
 * when the most recent solve did not run the driver.
 */
Bool GetPounceComplementarityStats(
    IpoptProblem                ipopt_problem,
    PounceComplementarityStats *stats
);

typedef struct {
    /** Set by the caller to sizeof(PounceComplementaritySolve). */
    ipindex  struct_size;
    /** Stage label (see above). */
    char     stage[32];
    /** 0/1: a relaxed solve (G.H <= tau); tau is NaN when 0. */
    ipindex  relaxed;
    ipnumber tau;
    /** ApplicationReturnStatus of this lowered solve. */
    ipindex  status;
    /** Its iterations, every attempt inside it included. */
    ipindex  iterations;
    /** Complementarity violation at its point; NaN when unmeasured. */
    ipnumber max_abs_product;
    /** Wall-clock seconds; NaN when unmeasured. */
    ipnumber seconds;
} PounceComplementaritySolve;

/**
 * Fill `solve` with lowered solve k (0-based, in the order run, k <
 * n_solves). FALSE, leaving `solve` untouched, on a NULL argument, a short
 * solve->struct_size, k out of range, and whenever
 * GetPounceComplementarityStats would be FALSE.
 */
Bool GetPounceComplementaritySolve(
    IpoptProblem                ipopt_problem,
    ipindex                     k,
    PounceComplementaritySolve *solve
);

#define POUNCE_CC_TEXT_SUMMARY        0   /* the driver's one-line account */
#define POUNCE_CC_TEXT_POLISH_REASON  1   /* why the polish's point was not taken */
#define POUNCE_CC_TEXT_B_REASON       2   /* why the B-check was not decided */

/**
 * One of the driver's texts (POUNCE_CC_TEXT_*), with snprintf's contract:
 * writes at most buflen-1 bytes plus a NUL and returns the text's full
 * length in bytes without the NUL, so buf = NULL, buflen = 0 sizes the
 * buffer and a return >= buflen means it was truncated. A NULL buf or
 * buflen <= 0 writes nothing. -1, writing nothing, when there is no such
 * text: a NULL handle, an unknown `which`, whenever
 * GetPounceComplementarityStats would be FALSE, a POLISH_REASON unless the
 * polish ran and its point was not taken, a B_REASON unless b_verdict is
 * NOT_CHECKED.
 */
ipindex GetPounceComplementarityText(
    IpoptProblem ipopt_problem,
    int          which,
    char        *buf,
    ipindex      buflen
);

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
 *
 * The output working-set buffers are written only when the solve
 * completed and IpoptGetWorkingSet has a set to give; after a refused
 * solve (complementarity pairs with a working set) they are untouched.
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
 * When that solve ran the complementarity driver (pairs were declared),
 * the report carries the `complementarity` block at either detail level:
 * the same block, from the same conversion, as the CLI's --json-output
 * and Python's report_path write for the same driver outcome. Its fields
 * are the ones GetPounceComplementarityStats / GetPounceComplementaritySolve
 * / GetPounceComplementarityText return, with `null` (or an absent key)
 * where those read NaN, -1 or "". The block is absent after a solve that
 * did not run the driver, and a refused solve leaves no report at all.
 *
 * When that solve ran no driver and the second-opinion ladder ran at
 * least one rung, the report carries the `second_opinion` block at either
 * detail level, the same block the CLI and Python write: the rungs tried,
 * the promoted one, and the base solve's status and iteration count (on a
 * promotion, the report's status and iteration count are the promoted
 * rung's). Absent when no rung ran and after a solve with pairs.
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
 *
 * Also returns NULL when the problem has declared complementarity pairs,
 * and then does NOT consume it: `*prob_handle` stays valid, the caller
 * still owns it (solve it with IpoptSolve, free it with FreeIpoptProblem).
 * A session keeps the KKT factor of the model it solved, which with pairs
 * would be the lowered model's, so the parametric step and reduced Hessian
 * over your rows would answer a different system. At print_level >= 1 one
 * line on stderr says why.
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

/* -----------------------------------------------------------------
 * Matrix-form convex QP / conic entry (gh#1147 Stage 3). Not an Ipopt
 * API: every name carries the Pounce prefix.
 *
 *   minimize    1/2 x'P x + c'x
 *   subject to  A x = b,   G x <= h  (G x <=_K h with cones),   lb <= x <= ub
 *
 * P is given as its LOWER triangle (row >= col); A and G as full
 * triplets. Triplet indices are 0-based for index_style 0 and 1-based
 * for index_style 1. Duplicate triplets are summed. Every builder
 * validates all of its input first and changes nothing when it
 * returns FALSE (NULL handle or array, index out of range, an
 * upper-triangle entry of P, a non-finite value). A builder that
 * returns TRUE discards the previous solve's result.
 *
 * Cones (PounceQpAddCone) partition the G rows in order; each block's
 * slack s = h - G x must lie in its cone (table at POUNCE_CONE_*). A PSD
 * block of order k holds svec(S): the lower triangle of the k x k slack
 * column by column, (0,0), (1,0), ..., (k-1,0), (1,1), (2,1), ..., with
 * every off-diagonal entry scaled by sqrt(2), so <X,Y> = svec(X).svec(Y).
 * Variable bounds are allowed with cones (a trailing nonnegative block).
 *
 * Without cones the solve is Python's solve_qp pipeline (presolve when
 * qp_presolve=yes, then the IPM or the active-set engine, postsolve,
 * the unverified-Optimal demotion); with any cone it is solve_socp's
 * conic IPM. Both run pounce_convex::pipeline, the code Python runs.
 *
 * PSD screen (gh#112). Under method=ipm (with or without cones) a P
 * that is not certified
 * positive semidefinite is refused (POUNCE_ERR_INDEFINITE). With
 * hessian_inertia=auto (the default) the verdict comes from
 * pounce_convex::certify_psd_lower_triangle, as in the CLI's dispatch:
 * P passes when the inertia of P + t*I has no negative eigenvalue, at
 * the CLI's band t = 1e-9 * min(max|P_ij|, 1). A factorization that
 * cannot decide counts as not certified. The verdict is kept on the
 * handle until PounceQpSetObjective or a hessian_inertia setting. Python decides the same
 * question in numpy (relative tolerance 1e-8*max|P_ij|, eigenvalues
 * for n <= 1500), so on a borderline P (lambda_min within a few
 * tolerances of zero, or a badly scaled rank-deficient P) the two can
 * disagree; this entry errs toward refusing. Without cones,
 * hessian_inertia=psd, indefinite or unknown replaces the verdict with
 * the caller's claim; with cones any value but auto is refused
 * (POUNCE_ERR_OPTIONS), since solve_socp always screens.
 * Under method=active-set the verdict is handed to the engine
 * (indefinite P is solved to a local point, not refused).
 *
 * Getters describe the most recent PounceQpSolve that returned a
 * status (>= 0): before one, and after a solve that returned an
 * error (< 0), they return FALSE and write nothing. A builder call
 * that returns TRUE erases the result; an option call does not.
 * ----------------------------------------------------------------- */
struct PounceQpInfo;
typedef struct PounceQpInfo* PounceQp;

/* PounceQpSolve statuses (>= 0), one per pounce_convex::QpStatus. */
#define POUNCE_QP_STATUS_OPTIMAL             0
#define POUNCE_QP_STATUS_OPTIMAL_INACCURATE  1
#define POUNCE_QP_STATUS_PRIMAL_INFEASIBLE   2
#define POUNCE_QP_STATUS_DUAL_INFEASIBLE     3
#define POUNCE_QP_STATUS_ITERATION_LIMIT     4
#define POUNCE_QP_STATUS_TIME_LIMIT          5
#define POUNCE_QP_STATUS_NUMERICAL_FAILURE   6


/* PounceQpAddCone kinds. */
#define POUNCE_CONE_NONNEG  0   /* dim rows, dim >= 0: s >= 0 */
#define POUNCE_CONE_SOC     1   /* dim rows, dim >= 1: s0 >= ||s[1:]|| */
#define POUNCE_CONE_EXP     2   /* dim must be 3: s1*exp(s0/s1) <= s2, s1 > 0 (closure) */
#define POUNCE_CONE_POW     3   /* dim must be 3; alpha in (0, 1): |s0| <= s1^alpha s2^(1-alpha) */
#define POUNCE_CONE_PSD     4   /* dim = matrix order k >= 1; k(k+1)/2 svec rows: smat(s) PSD */

/* PounceQpCertificate.kind. */
#define POUNCE_QP_CERT_NONE               0
#define POUNCE_QP_CERT_PRIMAL_INFEASIBLE  1   /* Farkas ray in y, z, z_lb, z_ub */
#define POUNCE_QP_CERT_DUAL_INFEASIBLE    2   /* recession ray in x */

/** Final KKT residuals (Python's QpResult.residuals). Without cones the
 *  first four are read above their finite-precision floor in the
 *  objective's units (what the Optimal verdict is judged on); the
 *  _raw four are the plain absolute values. With cones both sets are
 *  the plain conic residuals. Extensible (struct_size first; see
 *  "Extensible output structs" above). */
typedef struct {
    /** Set by the caller to sizeof(PounceQpResiduals). */
    ipindex  struct_size;
    ipnumber primal_infeasibility;
    ipnumber dual_infeasibility;
    ipnumber complementarity;
    ipnumber kkt_error;
    ipnumber primal_infeasibility_raw;
    ipnumber dual_infeasibility_raw;
    ipnumber complementarity_raw;
    ipnumber kkt_error_raw;
} PounceQpResiduals;

/** Infeasibility certificate and HSDE scalars (Python's
 *  certificate_scale, tau, kappa). `scale` is the ray's inf-norm, NaN
 *  when kind is POUNCE_QP_CERT_NONE. The ray carries the embedding's
 *  1/tau scale; divide by `scale` for a unit ray. tau and kappa are NaN
 *  when the driver records none: the direct IPM (qp_hsde=no), the
 *  active-set engine, and the exponential/power-cone driver.
 *  Extensible (struct_size first). */
typedef struct {
    /** Set by the caller to sizeof(PounceQpCertificate). */
    ipindex  struct_size;
    ipindex  kind;
    ipnumber scale;
    ipnumber tau;
    ipnumber kappa;
} PounceQpCertificate;

/** n >= 1 variables, m_eq >= 0 and m_ineq >= 0 rows, index_style 0 or
 *  1. NULL on any other argument. */
PounceQp PounceCreateQp(ipindex n, ipindex m_eq, ipindex m_ineq, ipindex index_style);

/** Free a handle. NULL is a no-op. The handle must not be used after. */
void PounceFreeQp(PounceQp qp);

/** P (lower triangle, nnz_p triplets) and c (length n, required). */
Bool PounceQpSetObjective(PounceQp qp, ipindex nnz_p, const ipindex* p_row,
                          const ipindex* p_col, const ipnumber* p_val,
                          const ipnumber* c);

/** A (m_eq x n) and b (length m_eq). Required when m_eq > 0. */
Bool PounceQpSetEqualities(PounceQp qp, ipindex nnz_a, const ipindex* a_row,
                           const ipindex* a_col, const ipnumber* a_val,
                           const ipnumber* b);

/** G (m_ineq x n) and h (length m_ineq). Required when m_ineq > 0. */
Bool PounceQpSetInequalities(PounceQp qp, ipindex nnz_g, const ipindex* g_row,
                             const ipindex* g_col, const ipnumber* g_val,
                             const ipnumber* h);

/** Variable box, each length n or NULL (no bound on that side). An
 *  entry <= -1e19 in lb or >= 1e19 in ub is no bound. FALSE, storing
 *  nothing, for NaN, lb[i] >= 1e19, ub[i] <= -1e19 or lb[i] > ub[i];
 *  lb[i] == ub[i] (fixed) is accepted. */
Bool PounceQpSetBounds(PounceQp qp, const ipnumber* lb, const ipnumber* ub);

/** Append a cone (POUNCE_CONE_*) to the partition of the G rows, in
 *  order. Returns its 0-based position, or POUNCE_ERR_NULL /
 *  POUNCE_ERR_ARGUMENT. alpha is read for POUNCE_CONE_POW only. */
int PounceQpAddCone(PounceQp qp, int kind, ipindex dim, ipnumber alpha);

/** tol (0,1); max_iter (whole, >= 1); time_limit (seconds, >= 0);
 *  tau, tau_max (0,1); qp_reg (>= 0). FALSE on an unknown key or a
 *  value out of range. */
Bool PounceQpSetNumOption(PounceQp qp, const char* key, ipnumber value);

/** method: ipm | active-set; hessian_inertia: auto | psd | indefinite |
 *  unknown; qp_presolve, qp_hsde, qp_equilibrate, qp_crossover: yes |
 *  no. FALSE on an unknown key or value. */
Bool PounceQpSetStrOption(PounceQp qp, const char* key, const char* value);

/** POUNCE_QP_STATUS_* (>= 0), or POUNCE_ERR_* (< 0) before any
 *  engine runs. Discards the previous result either way. */
int PounceQpSolve(PounceQp qp);

/** Copy the solution out; any output may be NULL. x, z_lb, z_ub: n;
 *  y: m_eq; z: m_ineq. Signs: P x + c + A'y + G'z - z_lb + z_ub = 0,
 *  z, z_lb, z_ub >= 0 (z cone-dual with cones). FALSE (nothing written)
 *  without a result, or if a result vector is not of its stated length. */
Bool PounceQpGetSolution(PounceQp qp, ipnumber* x, ipnumber* y, ipnumber* z,
                         ipnumber* z_lb, ipnumber* z_ub, ipnumber* obj,
                         ipindex* iters);

/** FALSE (nothing written) for a NULL handle or out, an
 *  out->struct_size smaller than sizeof(PounceQpResiduals), or no
 *  result. */
Bool PounceQpGetResiduals(PounceQp qp, PounceQpResiduals* out);

/** TRUE after every solve that returned a status; kind says whether a
 *  certificate is present. FALSE (nothing written) for a NULL handle or
 *  out, a short out->struct_size, or no result. */
Bool PounceQpGetCertificate(PounceQp qp, PounceQpCertificate* out);

/** pounce.solve-report/v1 JSON at detail "summary" (or NULL) / "full".
 *  solution.engine is cvx-qp, qp-active-set, or cvx-qcqp (any cone);
 *  solution.lambda is y then z. FALSE for a NULL handle or path, no
 *  result, an unknown detail, or an I/O error. */
Bool PounceQpWriteSolveReport(PounceQp qp, const char* path, const char* detail);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POUNCE_H */
