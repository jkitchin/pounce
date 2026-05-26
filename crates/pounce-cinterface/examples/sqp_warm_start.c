/*
 * sqp_warm_start.c — Phase 5c §7.2 working-set warm-start demo in C.
 *
 * Solves a sequence of perturbed convex QPs through pounce's
 * active-set SQP path, carrying the working set across solves via
 * `IpoptSetWarmStartWorkingSet` / `IpoptGetWorkingSet` (or the
 * one-shot `IpoptSolveWarmStart`).
 *
 * Problem (matches the tutorial / Python notebook):
 *
 *     min ½‖x − p‖²   s.t.  sum(x) = 1,  x ≥ 0,    n = 3
 *
 * Build (against the workspace's pounce-cinterface library):
 *
 *     cargo build --release -p pounce-cinterface
 *     cc -I crates/pounce-cinterface/include \
 *        -L target/release \
 *        crates/pounce-cinterface/examples/sqp_warm_start.c \
 *        -o sqp_warm_start \
 *        -lpounce_cinterface -Wl,-rpath,$PWD/target/release
 *     ./sqp_warm_start
 *
 * Expected output: 4 solves, each succeeds with the working set
 * carried forward. The first solve uses the SQP cold-start path
 * (no working set supplied); subsequent solves use the previous
 * solve's working set as warm-start input.
 */

#include "pounce.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---- problem state shared with callbacks --------------------------------- */

typedef struct {
    double p[3];   /* the moving parameter */
} ProblemData;

/* ---- TNLP callbacks ------------------------------------------------------ */

static bool eval_f(ipindex n, ipnumber *x, bool new_x,
                   ipnumber *obj, UserDataPtr user_data)
{
    (void)new_x;
    const ProblemData *d = (const ProblemData *)user_data;
    double s = 0.0;
    for (int j = 0; j < n; j++) {
        double e = x[j] - d->p[j];
        s += e * e;
    }
    *obj = 0.5 * s;
    return true;
}

static bool eval_grad_f(ipindex n, ipnumber *x, bool new_x,
                        ipnumber *grad, UserDataPtr user_data)
{
    (void)new_x;
    const ProblemData *d = (const ProblemData *)user_data;
    for (int j = 0; j < n; j++) grad[j] = x[j] - d->p[j];
    return true;
}

static bool eval_g(ipindex n, ipnumber *x, bool new_x,
                   ipindex m, ipnumber *g, UserDataPtr user_data)
{
    (void)new_x; (void)m; (void)user_data;
    double s = 0.0;
    for (int j = 0; j < n; j++) s += x[j];
    g[0] = s;
    return true;
}

static bool eval_jac_g(ipindex n, ipnumber *x, bool new_x,
                       ipindex m, ipindex nele_jac,
                       ipindex *irow, ipindex *jcol, ipnumber *values,
                       UserDataPtr user_data)
{
    (void)x; (void)new_x; (void)m; (void)user_data;
    if (irow != NULL && jcol != NULL) {
        for (int j = 0; j < nele_jac; j++) {
            irow[j] = 0;
            jcol[j] = j;
        }
    }
    if (values != NULL) {
        for (int j = 0; j < nele_jac; j++) values[j] = 1.0;
    }
    return true;
}

static bool eval_h(ipindex n, ipnumber *x, bool new_x,
                   ipnumber obj_factor,
                   ipindex m, ipnumber *lambda, bool new_lambda,
                   ipindex nele_hess,
                   ipindex *irow, ipindex *jcol, ipnumber *values,
                   UserDataPtr user_data)
{
    (void)x; (void)new_x; (void)m; (void)lambda; (void)new_lambda; (void)user_data;
    if (irow != NULL && jcol != NULL) {
        for (int j = 0; j < nele_hess; j++) {
            irow[j] = j;
            jcol[j] = j;
        }
    }
    if (values != NULL) {
        for (int j = 0; j < nele_hess; j++) values[j] = obj_factor;
    }
    return true;
}

/* ---- driver -------------------------------------------------------------- */

static void print_step(int k, const double *p, const double *x,
                       int status, const IpoptBoundStatus *bounds,
                       const IpoptConsStatus *cons)
{
    printf("step %d:  p = (%6.3f, %6.3f, %6.3f)   x = (%6.4f, %6.4f, %6.4f)   "
           "status = %d   bounds = [%d,%d,%d]   cons = [%d]\n",
           k, p[0], p[1], p[2], x[0], x[1], x[2], status,
           bounds[0], bounds[1], bounds[2], cons[0]);
}

int main(void)
{
    const int n = 3, m = 1;

    double x_l[3] = {0.0, 0.0, 0.0};
    double x_u[3] = {1e20, 1e20, 1e20};
    double g_l[1] = {1.0};
    double g_u[1] = {1.0};

    IpoptProblem prob = CreateIpoptProblem(
        n, x_l, x_u,
        m, g_l, g_u,
        /* nele_jac */ n,
        /* nele_hess */ n,
        /* index_style */ 0,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h
    );
    if (!prob) {
        fprintf(stderr, "CreateIpoptProblem failed\n");
        return 1;
    }

    /* Select the active-set SQP driver. */
    AddIpoptStrOption(prob, "algorithm", "active-set-sqp");
    AddIpoptIntOption(prob, "print_level", 0);

    /* Parameter sequence: small perturbations of a starting point
     * that keeps x[2] active at its lower bound. */
    const double parameters[4][3] = {
        {0.50, 0.40, -0.10},
        {0.52, 0.39, -0.05},
        {0.55, 0.37,  0.08},  /* x[2] active set may flip here */
        {0.54, 0.38,  0.08},
    };

    ProblemData data;
    double x[3] = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    double obj;

    IpoptBoundStatus bounds[3];
    IpoptConsStatus  cons[1];

    for (int k = 0; k < 4; k++) {
        /* Update the parameter inside the callback's user_data. */
        for (int j = 0; j < n; j++) data.p[j] = parameters[k][j];

        int status;
        if (k == 0) {
            /* First solve: cold start, no working set supplied.
             * IpoptSolve handles this. */
            status = IpoptSolve(prob, x, NULL, &obj,
                                NULL, NULL, NULL, &data);
            /* Read the WS the solver produced for the next step. */
            IpoptGetWorkingSet(prob, bounds, cons);
        } else {
            /* Subsequent solves: warm-start with the previous
             * iteration's working set. `IpoptSolveWarmStart` is the
             * one-shot convenience entry combining
             * Set + Solve + Get. The input/output WS buffers may
             * alias (we reuse `bounds` and `cons`). */
            status = IpoptSolveWarmStart(
                prob, x, NULL, &obj,
                NULL, NULL, NULL,
                bounds, cons,        /* in  */
                bounds, cons,        /* out */
                &data
            );
        }

        print_step(k, data.p, x, status, bounds, cons);

        if (status != 0) {
            fprintf(stderr, "solve failed at step %d (status %d)\n", k, status);
            FreeIpoptProblem(prob);
            return 1;
        }
    }

    /* Final sanity check: sum(x) == 1 to solver tolerance. */
    double s = 0.0;
    for (int j = 0; j < n; j++) s += x[j];
    if (fabs(s - 1.0) > 1e-6) {
        fprintf(stderr, "sum(x) = %.9f != 1\n", s);
        FreeIpoptProblem(prob);
        return 1;
    }
    printf("all 4 solves succeeded; final sum(x) = %.9f\n", s);

    FreeIpoptProblem(prob);
    return 0;
}
