/**
 * gams_pounce.c — GAMS solver link for POUNCE.
 *
 * Translates between the GAMS Modeling Object (GMO) API and POUNCE's C API
 * (`pounce.h`, a drop-in port of Ipopt 3.14's `IpStdCInterface.h`). Produces
 * a shared library that drops into a GAMS installation so that models can
 * use `option nlp = pounce;`.
 *
 * Entry points (prefix "pou" registered in gmscmpun.txt):
 *   pouCreate      — allocate solver data
 *   pouFree        — free solver data
 *   pouReadyAPI    — initialize GAMS API libraries
 *   pouCallSolver  — extract problem, solve, report solution
 *
 * Build:  make -C gams  (see Makefile)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "pounce.h"
#include "gmomcc.h"
#include "gevmcc.h"

/* ---------------------------------------------------------------------------
 * Solver data
 * --------------------------------------------------------------------------- */

typedef struct {
    gmoHandle_t gmo;
    gevHandle_t gev;

    /* Problem dimensions */
    int n;          /* number of variables */
    int m;          /* number of constraints */
    int nnz_jac;    /* Jacobian nonzeros */
    int nnz_hess;   /* Hessian nonzeros (lower triangle) */

    /* 1.0 for minimization, -1.0 for maximization */
    double obj_sign;

    /* Jacobian CSR structure (from gmoGetMatrixRow) */
    int *jac_rowstart;  /* length m + 1 */
    int *jac_colidx;    /* length nnz_jac */
    double *jac_values_init; /* length nnz_jac; linear coefs from gmoGetMatrixRow,
                                used directly for entries with nlflag == 0 */
    int *jac_nlflag;    /* length nnz_jac; 0 = linear entry, !=0 = nonlinear */
    char *row_has_nl;   /* length m; nonzero if row has any nonlinear entries */

    /* Dense gradient scratch buffer (length n).
     * Only positions referenced by the current row are written/read each call,
     * so the buffer is cleared sparsely via jac_colidx for that row. */
    double *grad_buf;

    /* Has analytical Hessian? */
    int have_hessian;
} PounceGamsData;

/* ---------------------------------------------------------------------------
 * Callback: objective f(x)
 * --------------------------------------------------------------------------- */

static bool gams_eval_f(ipindex n, ipnumber *x, bool new_x,
                        ipnumber *obj, UserDataPtr user_data)
{
    PounceGamsData *d = (PounceGamsData *)user_data;
    double fval;
    int numerr;

    (void)n; (void)new_x;

    if (gmoEvalFuncObj(d->gmo, x, &fval, &numerr) != 0 || numerr > 0)
        return false;

    *obj = d->obj_sign * fval;
    return true;
}

/* ---------------------------------------------------------------------------
 * Callback: gradient of f(x)
 * --------------------------------------------------------------------------- */

static bool gams_eval_grad_f(ipindex n, ipnumber *x, bool new_x,
                             ipnumber *grad, UserDataPtr user_data)
{
    PounceGamsData *d = (PounceGamsData *)user_data;
    double fval, gx;
    int numerr;

    (void)new_x;

    if (gmoEvalGradObj(d->gmo, x, &fval, grad, &gx, &numerr) != 0 || numerr > 0)
        return false;

    if (d->obj_sign < 0.0) {
        for (int j = 0; j < n; j++)
            grad[j] = -grad[j];
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * Callback: constraints g(x)
 * --------------------------------------------------------------------------- */

static bool gams_eval_g(ipindex n, ipnumber *x, bool new_x,
                        ipindex m, ipnumber *g, UserDataPtr user_data)
{
    PounceGamsData *d = (PounceGamsData *)user_data;
    double fval;
    int numerr;

    (void)n; (void)new_x;

    for (int i = 0; i < m; i++) {
        if (gmoEvalFunc(d->gmo, i, x, &fval, &numerr) != 0 || numerr > 0)
            return false;
        g[i] = fval;
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * Callback: Jacobian of constraints
 *
 * Structure mode (values == NULL): expand CSR to COO.
 *
 * Values mode: for each row,
 *   - if the row has no nonlinear entries, copy the cached linear coefficients
 *     directly (gmoGetMatrixRow already gave us those at setup);
 *   - otherwise, sparse-clear grad_buf at the row's structural columns, call
 *     gmoEvalGrad once, and pull out the sparse entries.
 *
 * The sparse clear (length = row_nnz) replaces a per-row memset of the full
 * n-vector, which was O(m*n) per Jacobian evaluation. That O(m*n) was the
 * dominant cost on large qcqp instances.
 * --------------------------------------------------------------------------- */

static bool gams_eval_jac_g(ipindex n, ipnumber *x, bool new_x,
                            ipindex m, ipindex nele_jac,
                            ipindex *iRow, ipindex *jCol,
                            ipnumber *values, UserDataPtr user_data)
{
    PounceGamsData *d = (PounceGamsData *)user_data;

    (void)n; (void)new_x; (void)nele_jac;

    if (values == NULL) {
        /* Sparsity pattern: expand CSR to COO (0-based) */
        int k = 0;
        for (int i = 0; i < m; i++) {
            for (int j = d->jac_rowstart[i]; j < d->jac_rowstart[i + 1]; j++) {
                iRow[k] = i;
                jCol[k] = d->jac_colidx[j];
                k++;
            }
        }
    } else {
        for (int i = 0; i < m; i++) {
            int rs = d->jac_rowstart[i];
            int re = d->jac_rowstart[i + 1];

            if (!d->row_has_nl[i]) {
                /* Pure-linear row: constant gradient, just copy cached coefs. */
                for (int j = rs; j < re; j++)
                    values[j] = d->jac_values_init[j];
                continue;
            }

            /* Sparse clear of only this row's structural columns. */
            for (int j = rs; j < re; j++)
                d->grad_buf[d->jac_colidx[j]] = 0.0;

            double fval, gx;
            int numerr;
            if (gmoEvalGrad(d->gmo, i, x, &fval, d->grad_buf, &gx, &numerr) != 0
                || numerr > 0)
                return false;

            for (int j = rs; j < re; j++)
                values[j] = d->grad_buf[d->jac_colidx[j]];
        }
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * Callback: Hessian of the Lagrangian (lower triangle)
 *
 * POUNCE convention:  H = obj_factor * nabla^2 f + sum_i lambda_i * nabla^2 c_i
 *
 * gmoHessLagValue(gmo, x, pi, w, objweight, conweight, numerr) computes:
 *   w = objweight * nabla^2 f + conweight * sum_i pi_i * nabla^2 c_i
 *
 * GAMS multiplier convention: pi_gams = -lambda_ipopt.
 * The Ipopt solver link negates lambda before passing to gmoHessLagValue.
 * Equivalently, we pass lambda directly with conweight = -1.0:
 *   w = objweight * nabla^2 f + (-1) * sum_i lambda_i * nabla^2 c_i
 *     = objweight * nabla^2 f - sum_i lambda_i * nabla^2 c_i
 * which matches negating lambda and using conweight = 1.0.
 *
 * For maximization, obj_sign = -1, so objweight = -obj_factor negates the
 * objective Hessian (POUNCE minimizes -f).
 * --------------------------------------------------------------------------- */

static bool gams_eval_h(ipindex n, ipnumber *x, bool new_x,
                        ipnumber obj_factor,
                        ipindex m, ipnumber *lambda, bool new_lambda,
                        ipindex nele_hess,
                        ipindex *iRow, ipindex *jCol,
                        ipnumber *values, UserDataPtr user_data)
{
    PounceGamsData *d = (PounceGamsData *)user_data;

    (void)n; (void)new_x; (void)m; (void)new_lambda; (void)nele_hess;

    if (values == NULL) {
        /* Sparsity pattern: COO (lower triangle) from gmoHessLagStruct */
        gmoHessLagStruct(d->gmo, iRow, jCol);
    } else {
        int numerr;
        double objweight = d->obj_sign * obj_factor;

        /* conweight = -1.0: equivalent to negating lambda (GAMS sign convention) */
        if (gmoHessLagValue(d->gmo, x, lambda, values,
                            objweight, -1.0, &numerr) != 0
            || numerr > 0)
            return false;
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * Option file parsing
 *
 * Reads lines of the form "key value" from pounce.opt (or .op2 etc.)
 * Lines starting with '*' are comments. Blank lines are skipped.
 * --------------------------------------------------------------------------- */

static void parse_option_file(IpoptProblem nlp, const char *filename,
                              gevHandle_t gev)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) return;

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        /* Skip comments and blank lines */
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '*' || *p == '#') continue;

        /* Remove trailing newline */
        char *nl = strchr(p, '\n');
        if (nl) *nl = '\0';
        char *cr = strchr(p, '\r');
        if (cr) *cr = '\0';

        /* Split "key value" */
        char key[256], val[256];
        if (sscanf(p, "%255s %255s", key, val) < 2)
            continue;

        /* Try integer first, then double, then string */
        char *endptr;
        long ival = strtol(val, &endptr, 10);
        if (*endptr == '\0') {
            if (AddIpoptIntOption(nlp, key, (int)ival)) {
                gevLogStat(gev, line);
                continue;
            }
        }

        double dval = strtod(val, &endptr);
        if (*endptr == '\0') {
            if (AddIpoptNumOption(nlp, key, dval)) {
                gevLogStat(gev, line);
                continue;
            }
        }

        if (AddIpoptStrOption(nlp, key, val)) {
            gevLogStat(gev, line);
            continue;
        }

        /* Unknown option */
        char msgbuf[512];
        snprintf(msgbuf, sizeof(msgbuf), "*** Warning: unknown option '%s'", key);
        gevLogStat(gev, msgbuf);
    }
    fclose(fp);
}

/* ---------------------------------------------------------------------------
 * GAMS solver link entry points
 * --------------------------------------------------------------------------- */

#if defined(_WIN32)
# define DllExport __declspec(dllexport)
# define STDCALL   __stdcall
#else
# define DllExport __attribute__((visibility("default")))
# define STDCALL
#endif

/** Allocate solver data and initialize GAMS API wrappers.
 *
 * Returns 0 on success, 1 on failure (with error in msgBuf).
 * GAMS calls gmoGetReady/gevGetReady here (before ReadyAPI),
 * matching the pattern used by IPOPT and other solver links.
 */
DllExport int STDCALL pouCreate(void **Cptr, char *msgBuf, int msgBufLen)
{
    *Cptr = NULL;

    /* Initialize GAMS API wrappers (function pointers) */
    if (!gmoGetReady(msgBuf, msgBufLen))
        return 1;
    if (!gevGetReady(msgBuf, msgBufLen))
        return 1;

    PounceGamsData *data = (PounceGamsData *)calloc(1, sizeof(PounceGamsData));
    if (!data) {
        snprintf(msgBuf, msgBufLen, "pounce: memory allocation failed");
        return 1;
    }
    *Cptr = data;
    msgBuf[0] = '\0';
    return 0;
}

/** Free solver data. */
DllExport void STDCALL pouFree(void **Cptr)
{
    if (Cptr && *Cptr) {
        PounceGamsData *data = (PounceGamsData *)*Cptr;
        free(data->jac_rowstart);
        free(data->jac_colidx);
        free(data->jac_values_init);
        free(data->jac_nlflag);
        free(data->row_has_nl);
        free(data->grad_buf);
        free(data);
        *Cptr = NULL;
    }
}

/** Initialize GAMS API — receive GMO handle and extract GEV. */
DllExport int STDCALL pouReadyAPI(void *Cptr, gmoHandle_t gmo)
{
    PounceGamsData *data = (PounceGamsData *)Cptr;

    data->gmo = gmo;
    data->gev = (gevHandle_t)gmoEnvironment(gmo);
    if (!data->gev) {
        gmoSolveStatSet(gmo, gmoSolveStat_SetupErr);
        gmoModelStatSet(gmo, gmoModelStat_ErrorNoSolution);
        return 1;
    }

    return 0;
}

/** Map POUNCE return status to GAMS model/solve status.
 *
 * Integer values here are POUNCE's `ApplicationReturnStatus` (see
 * IpoptReturnCodes.h) — identical to Ipopt 3.14's numbering. Non-optimal
 * terminal codes that still carry a usable primal iterate map to
 * gmoModelStat_Feasible (7) so GAMS sees the point rather than treating
 * it as an internal failure.
 */
static void map_status_to_gams(int status, int *model_stat, int *solve_stat)
{
    switch (status) {
    case Solve_Succeeded:
        *model_stat = gmoModelStat_OptimalLocal;
        *solve_stat = gmoSolveStat_Normal;
        break;
    case Solved_To_Acceptable_Level:
        *model_stat = gmoModelStat_Feasible;
        *solve_stat = gmoSolveStat_Normal;
        break;
    case Feasible_Point_Found:
        *model_stat = gmoModelStat_Feasible;
        *solve_stat = gmoSolveStat_Normal;
        break;
    case Infeasible_Problem_Detected:
        *model_stat = gmoModelStat_InfeasibleLocal;
        *solve_stat = gmoSolveStat_Solver;
        break;
    case Search_Direction_Becomes_Too_Small:
        *model_stat = gmoModelStat_Feasible;
        *solve_stat = gmoSolveStat_Solver;
        break;
    case Diverging_Iterates:
        *model_stat = gmoModelStat_Unbounded;
        *solve_stat = gmoSolveStat_Solver;
        break;
    case User_Requested_Stop:
        *model_stat = gmoModelStat_Feasible;
        *solve_stat = gmoSolveStat_User;
        break;
    case Maximum_Iterations_Exceeded:
        *model_stat = gmoModelStat_Feasible;
        *solve_stat = gmoSolveStat_Iteration;
        break;
    case Restoration_Failed:
        *model_stat = gmoModelStat_InfeasibleIntermed;
        *solve_stat = gmoSolveStat_Solver;
        break;
    case Error_In_Step_Computation:
        *model_stat = gmoModelStat_Feasible;
        *solve_stat = gmoSolveStat_SolverErr;
        break;
    case Maximum_CpuTime_Exceeded:
    case Maximum_WallTime_Exceeded:
        *model_stat = gmoModelStat_Feasible;
        *solve_stat = gmoSolveStat_Resource;
        break;
    case Not_Enough_Degrees_Of_Freedom:
    case Invalid_Problem_Definition:
    case Invalid_Option:
        *model_stat = gmoModelStat_ErrorNoSolution;
        *solve_stat = gmoSolveStat_SetupErr;
        break;
    case Invalid_Number_Detected:
        *model_stat = gmoModelStat_InfeasibleIntermed;
        *solve_stat = gmoSolveStat_EvalError;
        break;
    case Internal_Error:
    default:
        *model_stat = gmoModelStat_ErrorNoSolution;
        *solve_stat = gmoSolveStat_InternalErr;
        break;
    }
}

/** True when POUNCE's return status leaves a usable primal point in x. */
static int pounce_status_has_solution(int status)
{
    switch (status) {
    case Solve_Succeeded:
    case Solved_To_Acceptable_Level:
    case Feasible_Point_Found:
    case Infeasible_Problem_Detected:       /* best-so-far iterate */
    case Search_Direction_Becomes_Too_Small:
    case User_Requested_Stop:
    case Maximum_Iterations_Exceeded:
    case Error_In_Step_Computation:
    case Maximum_CpuTime_Exceeded:
    case Maximum_WallTime_Exceeded:
        return 1;
    default:
        return 0;
    }
}

/** Solve the NLP problem. */
DllExport int STDCALL pouCallSolver(void *Cptr)
{
    PounceGamsData *data = (PounceGamsData *)Cptr;
    gmoHandle_t gmo = data->gmo;
    gevHandle_t gev = data->gev;
    char msg[512];
    int n, m, rc;

    /* All heap pointers initialized here so that goto cleanup is always safe */
    double *x_l     = NULL;
    double *x_u     = NULL;
    double *g_l     = NULL;
    double *g_u     = NULL;
    double *x       = NULL;
    double *g_vals  = NULL;
    double *mult_g  = NULL;
    double *mult_xl = NULL;
    double *mult_xu = NULL;
    IpoptProblem nlp = NULL;

    gevLogStat(gev, "");
    gevLogStat(gev, "--- POUNCE: A Rust Interior-Point Optimizer");
    gevLogStat(gev, "");

    /* ---------------------------------------------------------------
     * Configure GMO
     * --------------------------------------------------------------- */
    gmoObjStyleSet(gmo, gmoObjType_Fun);
    gmoObjReformSet(gmo, 1);
    gmoIndexBaseSet(gmo, 0);

    /* Objective sense */
    data->obj_sign = (gmoSense(gmo) == gmoObj_Max) ? -1.0 : 1.0;

    /* ---------------------------------------------------------------
     * Extract problem dimensions
     * --------------------------------------------------------------- */
    data->n = gmoN(gmo);
    data->m = gmoM(gmo);
    data->nnz_jac = gmoNZ(gmo);
    n = data->n;
    m = data->m;

    if (n == 0) {
        gevLogStat(gev, "*** Error: problem has no variables");
        gmoSolveStatSet(gmo, gmoSolveStat_SetupErr);
        gmoModelStatSet(gmo, gmoModelStat_ErrorNoSolution);
        return 1;
    }

    snprintf(msg, sizeof(msg), "  Variables: %d, Constraints: %d, Jacobian NZ: %d",
             n, m, data->nnz_jac);
    gevLogStat(gev, msg);

    /* ---------------------------------------------------------------
     * Load Hessian
     * --------------------------------------------------------------- */
    {
        int do2dir, doHess;
        rc = gmoHessLoad(gmo, 0.0, &do2dir, &doHess);
        if (rc != 0 || !doHess) {
            data->have_hessian = 0;
            data->nnz_hess = 0;
            gevLogStat(gev, "  Analytical Hessian not available, using L-BFGS");
        } else {
            data->have_hessian = 1;
            data->nnz_hess = gmoHessLagNz(gmo);
            snprintf(msg, sizeof(msg), "  Hessian NZ: %d", data->nnz_hess);
            gevLogStat(gev, msg);
        }
    }

    /* ---------------------------------------------------------------
     * Variable bounds
     * --------------------------------------------------------------- */
    x_l = (double *)malloc(n * sizeof(double));
    x_u = (double *)malloc(n * sizeof(double));
    if (!x_l || !x_u) goto oom;

    gmoGetVarLower(gmo, x_l);
    gmoGetVarUpper(gmo, x_u);

    /* Map GAMS infinity to POUNCE infinity (1e19) */
    {
        double gams_pinf = gmoPinf(gmo);
        double gams_minf = gmoMinf(gmo);
        for (int j = 0; j < n; j++) {
            if (x_l[j] <= gams_minf) x_l[j] = -1e19;
            if (x_u[j] >= gams_pinf) x_u[j] =  1e19;
        }
    }

    /* ---------------------------------------------------------------
     * Constraint bounds from equation types and RHS
     * --------------------------------------------------------------- */
    if (m > 0) {
        g_l = (double *)malloc(m * sizeof(double));
        g_u = (double *)malloc(m * sizeof(double));
        if (!g_l || !g_u) goto oom;

        for (int i = 0; i < m; i++) {
            int etyp = gmoGetEquTypeOne(gmo, i);
            double rhs = gmoGetRhsOne(gmo, i);

            switch (etyp) {
            case gmoequ_E:  /* =E= equality */
                g_l[i] = rhs;
                g_u[i] = rhs;
                break;
            case gmoequ_G:  /* =G= greater-or-equal */
                g_l[i] = rhs;
                g_u[i] = 1e19;
                break;
            case gmoequ_L:  /* =L= less-or-equal */
                g_l[i] = -1e19;
                g_u[i] = rhs;
                break;
            case gmoequ_N:  /* =N= free / nonbinding */
                g_l[i] = -1e19;
                g_u[i] = 1e19;
                break;
            default:
                snprintf(msg, sizeof(msg),
                         "*** Warning: unsupported equation type %d for row %d",
                         etyp, i);
                gevLogStat(gev, msg);
                g_l[i] = -1e19;
                g_u[i] = 1e19;
                break;
            }
        }
    }

    /* ---------------------------------------------------------------
     * Jacobian structure (CSR from GMO, stored for value callbacks).
     *
     * We keep jacval (linear coefficients for entries with nlflag == 0)
     * and nlflag itself so that gams_eval_jac_g can (a) copy linear-row
     * values directly without calling the GMO evaluator, and (b) sparse-
     * clear the dense gradient buffer at only the structural columns.
     * --------------------------------------------------------------- */
    if (m > 0 && data->nnz_jac > 0) {
        data->jac_rowstart    = (int *)malloc((m + 1) * sizeof(int));
        data->jac_colidx      = (int *)malloc(data->nnz_jac * sizeof(int));
        data->jac_values_init = (double *)malloc(data->nnz_jac * sizeof(double));
        data->jac_nlflag      = (int *)malloc(data->nnz_jac * sizeof(int));
        data->row_has_nl      = (char *)calloc(m, sizeof(char));
        data->grad_buf        = (double *)calloc(n, sizeof(double));
        if (!data->jac_rowstart || !data->jac_colidx || !data->jac_values_init
            || !data->jac_nlflag || !data->row_has_nl || !data->grad_buf)
            goto oom;

        gmoGetMatrixRow(gmo, data->jac_rowstart, data->jac_colidx,
                        data->jac_values_init, data->jac_nlflag);

        for (int i = 0; i < m; i++) {
            for (int j = data->jac_rowstart[i]; j < data->jac_rowstart[i + 1]; j++) {
                if (data->jac_nlflag[j]) {
                    data->row_has_nl[i] = 1;
                    break;
                }
            }
        }
    }

    /* ---------------------------------------------------------------
     * Create POUNCE problem
     *
     * CreateIpoptProblem callback order is (eval_f, eval_g, eval_grad_f,
     * eval_jac_g, eval_h) — matches Ipopt's IpStdCInterface.
     * --------------------------------------------------------------- */
    {
        Eval_H_CB hess_cb = data->have_hessian ? gams_eval_h : NULL;
        int nnz_hess_arg  = data->have_hessian ? data->nnz_hess : 0;

        nlp = CreateIpoptProblem(
            n, x_l, x_u,
            m, g_l, g_u,
            data->nnz_jac, nnz_hess_arg,
            0,  /* index_style: 0 = C (0-based) indexing */
            gams_eval_f,
            gams_eval_g,
            gams_eval_grad_f,
            gams_eval_jac_g,
            hess_cb);
    }

    if (!nlp) {
        gevLogStat(gev, "*** Error: CreateIpoptProblem failed");
        gmoSolveStatSet(gmo, gmoSolveStat_SetupErr);
        gmoModelStatSet(gmo, gmoModelStat_ErrorNoSolution);
        rc = 1;
        goto cleanup;
    }

    /* ---------------------------------------------------------------
     * Default options from GAMS environment
     * --------------------------------------------------------------- */
    {
        int iterlim = gevGetIntOpt(gev, gevIterLim);
        if (iterlim < ITERLIM_INFINITY)
            AddIpoptIntOption(nlp, "max_iter", iterlim);

        double reslim = gevGetDblOpt(gev, gevResLim);
        if (reslim < RESLIM_INFINITY)
            AddIpoptNumOption(nlp, "max_wall_time", reslim);
    }

    /* Default print level */
    AddIpoptIntOption(nlp, "print_level", 5);

    /* Use L-BFGS if no analytical Hessian */
    if (!data->have_hessian)
        AddIpoptStrOption(nlp, "hessian_approximation", "limited-memory");

    /* ---------------------------------------------------------------
     * Read option file (pounce.opt, pounce.op2, ...)
     * --------------------------------------------------------------- */
    if (gmoOptFile(gmo) > 0) {
        char optfilename[512];
        gmoNameOptFile(gmo, optfilename);
        snprintf(msg, sizeof(msg), "  Reading option file %s", optfilename);
        gevLogStat(gev, msg);
        parse_option_file(nlp, optfilename, gev);
    }

    /* ---------------------------------------------------------------
     * Allocate solution arrays and set initial point
     * --------------------------------------------------------------- */
    x       = (double *)malloc(n * sizeof(double));
    g_vals  = m > 0 ? (double *)malloc(m * sizeof(double)) : NULL;
    mult_g  = m > 0 ? (double *)calloc(m, sizeof(double)) : NULL;
    mult_xl = (double *)calloc(n, sizeof(double));
    mult_xu = (double *)calloc(n, sizeof(double));

    if (!x || (m > 0 && (!g_vals || !mult_g)) || !mult_xl || !mult_xu)
        goto oom;

    gmoGetVarL(gmo, x);

    /* ---------------------------------------------------------------
     * Solve
     * --------------------------------------------------------------- */
    {
        double obj_val = 0.0;
        int status = (int)IpoptSolve(nlp, x, g_vals, &obj_val,
                                     mult_g, mult_xl, mult_xu,
                                     (void *)data);

        /* Always log the raw pounce return code so setup-time failures
         * (which return immediately without any solver-side message) can be
         * distinguished. Maps via map_status_to_gams below; this is the
         * pre-mapping integer per IpoptReturnCodes.h. */
        {
            char rcmsg[128];
            snprintf(rcmsg, sizeof(rcmsg),
                     "POUNCE return code: %d", status);
            gevLogStat(gev, rcmsg);
        }

        /* -----------------------------------------------------------
         * Report solution back to GAMS
         * ----------------------------------------------------------- */
        int model_stat, solve_stat;
        map_status_to_gams(status, &model_stat, &solve_stat);

        /* Objective in GAMS convention (undo our sign flip for max).
         * Report for any status that carries a usable primal iterate, so
         * GAMS trace rows for MaxIter / timeout / numerical-error returns
         * show the best-so-far objective instead of zero. */
        if (pounce_status_has_solution(status)) {
            double gams_obj = (data->obj_sign < 0.0) ? -obj_val : obj_val;
            gmoSetHeadnTail(gmo, gmoHobjval, gams_obj);
        }

        gmoModelStatSet(gmo, model_stat);
        gmoSolveStatSet(gmo, solve_stat);

        /* Report solver time and iteration count to GAMS trace */
        {
            int    iters_now = GetIpoptIterCount(nlp);
            double wall_now  = GetIpoptSolveTime(nlp);
            gmoSetHeadnTail(gmo, gmoHresused,  wall_now);
            gmoSetHeadnTail(gmo, gmoHiterused, (double)iters_now);
        }

        /* Negate constraint multipliers: POUNCE lambda -> GAMS pi */
        if (mult_g) {
            for (int i = 0; i < m; i++)
                mult_g[i] = -mult_g[i];
        }

        /* Set primal + dual solution */
        gmoSetSolution2(gmo, x, mult_g);

        /* Variable marginals: z_L - z_U, negated for max problems */
        {
            double *var_marg = (double *)calloc(n, sizeof(double));
            if (var_marg) {
                for (int j = 0; j < n; j++)
                    var_marg[j] = mult_xl[j] - mult_xu[j];
                if (data->obj_sign < 0.0) {
                    for (int j = 0; j < n; j++)
                        var_marg[j] = -var_marg[j];
                }
                gmoSetVarM(gmo, var_marg);
                free(var_marg);
            }
        }

        /* Print post-solve summary (mirrors Ipopt's EXIT block) */
        {
            int    iters      = GetIpoptIterCount(nlp);
            double wall_time  = GetIpoptSolveTime(nlp);
            double primal_inf = GetIpoptPrimalInf(nlp);
            double dual_inf   = GetIpoptDualInf(nlp);
            double compl_inf  = GetIpoptComplInf(nlp);
            double gams_obj   = (data->obj_sign < 0.0) ? -obj_val : obj_val;

            gevLogStat(gev, "");
            snprintf(msg, sizeof(msg),
                     "Number of Iterations....: %d", iters);
            gevLogStat(gev, msg);
            gevLogStat(gev, "");
            snprintf(msg, sizeof(msg),
                     "                                   (unscaled)");
            gevLogStat(gev, msg);
            snprintf(msg, sizeof(msg),
                     "Objective..............: %24.16e", gams_obj);
            gevLogStat(gev, msg);
            snprintf(msg, sizeof(msg),
                     "Dual infeasibility.....: %24.16e", dual_inf);
            gevLogStat(gev, msg);
            snprintf(msg, sizeof(msg),
                     "Constraint violation...: %24.16e", primal_inf);
            gevLogStat(gev, msg);
            snprintf(msg, sizeof(msg),
                     "Complementarity........: %24.16e", compl_inf);
            gevLogStat(gev, msg);
            gevLogStat(gev, "");
            snprintf(msg, sizeof(msg),
                     "Total seconds in POUNCE: %.3f", wall_time);
            gevLogStat(gev, msg);
            gevLogStat(gev, "");
        }

        snprintf(msg, sizeof(msg),
                 "  Solve status: %d (%s), Model status: %d (%s)",
                 solve_stat,
                 solve_stat <= 13 ? solveStatusTxt[solve_stat] : "?",
                 model_stat,
                 model_stat <= 19 ? modelStatusTxt[model_stat] : "?");
        gevLogStat(gev, msg);
    }

    rc = 0;

cleanup:
    if (nlp) FreeIpoptProblem(nlp);
    free(x_l);
    free(x_u);
    free(g_l);
    free(g_u);
    free(x);
    free(g_vals);
    free(mult_g);
    free(mult_xl);
    free(mult_xu);
    if (data->have_hessian)
        gmoHessUnload(gmo);
    return rc;

oom:
    gevLogStat(gev, "*** Error: out of memory");
    gmoSolveStatSet(gmo, gmoSolveStat_InternalErr);
    gmoModelStatSet(gmo, gmoModelStat_ErrorNoSolution);
    rc = 1;
    goto cleanup;
}
