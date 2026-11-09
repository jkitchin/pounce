/*
 * bounds_update.c — changing bounds on an existing problem (gh#1147 Stage 2).
 *
 * QpecSmall (see complementarity.c) with both pairs written the MOI way: the
 * rows are created FREE and each declared with
 * AddPounceComplementsConstraint, which writes F >= 0 into the handle's
 * bounds because y1 and y2 have only lower bounds:
 *
 *     min (x-1)^2 + (y1-1)^2 + y2^2,   0 <= x <= 2,
 *         [2 y1 - 1 - x, y1] in Complements,   y1 >= 0,
 *         [x + 2 y2 - 1, y2] in Complements,   y2 >= 0.
 *
 * Answer (1, 1, 0). Then, on the same handle, x's box shrinks to
 * [0.25, 0.75] with a sparse SetPounceBounds call. Both pairs, the options
 * and the Complements edits stay on the handle. The new answer is
 * (3/4, 7/8, 1/8), objective 3/32: at x = 3/4 both rows must be tight.
 *
 * Along the way: the two bound views (the caller's free rows, and the
 * F >= 0 the solve sees), and two refused updates that change nothing:
 * giving y1 an upper bound (it would move to another row of the MOI table,
 * which can change the pair) and giving a Complements row a finite bound.
 *
 * Build and run every example in this directory with
 *
 *     make c-api-examples
 *
 * `tests/c_examples_build_and_run.rs` compiles and runs this file on every
 * `cargo test -p pounce-cinterface`, and checks the last line it prints.
 */

#include "pounce.h"

#include <math.h>
#include <stdio.h>

#define N 3
#define M 2
#define INF 2e19 /* at or beyond 1e19 a bound is infinite */

/* ---- QpecSmall callbacks ------------------------------------------------- */

static bool eval_f(ipindex n, ipnumber *x, bool new_x,
                   ipnumber *obj, UserDataPtr user_data)
{
    (void)n; (void)new_x; (void)user_data;
    *obj = (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 1.0) * (x[1] - 1.0)
         + x[2] * x[2];
    return true;
}

static bool eval_grad_f(ipindex n, ipnumber *x, bool new_x,
                        ipnumber *grad, UserDataPtr user_data)
{
    (void)n; (void)new_x; (void)user_data;
    grad[0] = 2.0 * (x[0] - 1.0);
    grad[1] = 2.0 * (x[1] - 1.0);
    grad[2] = 2.0 * x[2];
    return true;
}

static bool eval_g(ipindex n, ipnumber *x, bool new_x,
                   ipindex m, ipnumber *g, UserDataPtr user_data)
{
    (void)n; (void)new_x; (void)m; (void)user_data;
    g[0] = 2.0 * x[1] - 1.0 - x[0];
    g[1] = x[0] + 2.0 * x[2] - 1.0;
    return true;
}

static bool eval_jac_g(ipindex n, ipnumber *x, bool new_x,
                       ipindex m, ipindex nele_jac,
                       ipindex *irow, ipindex *jcol, ipnumber *values,
                       UserDataPtr user_data)
{
    static const ipindex r[4] = {0, 0, 1, 1};
    static const ipindex c[4] = {0, 1, 0, 2};
    static const ipnumber v[4] = {-1.0, 2.0, 1.0, 2.0};
    (void)n; (void)x; (void)new_x; (void)m; (void)user_data;
    for (int k = 0; k < nele_jac; k++) {
        if (values == NULL) {
            irow[k] = r[k];
            jcol[k] = c[k];
        } else {
            values[k] = v[k];
        }
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
    (void)n; (void)x; (void)new_x; (void)m; (void)lambda; (void)new_lambda;
    (void)user_data;
    for (int k = 0; k < nele_hess; k++) {
        if (values == NULL) {
            irow[k] = k;
            jcol[k] = k;
        } else {
            values[k] = 2.0 * obj_factor;
        }
    }
    return true;
}

/* ---- helpers ------------------------------------------------------------- */

static int fail(IpoptProblem prob, const char *what)
{
    fprintf(stderr, "bounds_update.c: %s\n", what);
    FreeIpoptProblem(prob);
    return 1;
}

/* Solve from (0, 1/2, 1/2); 1 if it succeeded at `want` (to 1e-6). */
static int solve_to(IpoptProblem prob, const ipnumber want[N], ipnumber *obj)
{
    ipnumber x[N] = {0.0, 0.5, 0.5};
    int status = IpoptSolve(prob, x, NULL, obj, NULL, NULL, NULL, NULL);
    printf("  status %d, x = (%.6f, %.6f, %.6f), objective %.6f\n",
           status, x[0], x[1], x[2], *obj);
    if (status != Solve_Succeeded) return 0;
    for (int i = 0; i < N; i++) {
        if (fabs(x[i] - want[i]) > 1e-6) return 0;
    }
    return 1;
}

/* Row bounds in one view, printed. */
static int show_rows(IpoptProblem prob, int view, const char *label)
{
    ipnumber gl[M], gu[M];
    if (GetPounceBounds(prob, 0, NULL, NULL, NULL, M, NULL, gl, gu, view, NULL)
        != POUNCE_BOUNDS_OK) {
        return 0;
    }
    printf("  %-9s rows: [%g, %g], [%g, %g]\n", label, gl[0], gu[0], gl[1], gu[1]);
    return 1;
}

int main(void)
{
    ipnumber x_L[N] = {0.0, 0.0, 0.0};
    ipnumber x_U[N] = {2.0, INF, INF};
    ipnumber g_L[M] = {-INF, -INF}; /* free rows: MOI's F */
    ipnumber g_U[M] = {INF, INF};
    IpoptProblem prob = CreateIpoptProblem(
        N, x_L, x_U, M, g_L, g_U, 4, 3, 0,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h);
    if (prob == NULL) {
        fprintf(stderr, "bounds_update.c: CreateIpoptProblem failed\n");
        return 1;
    }
    AddIpoptIntOption(prob, "print_level", 0);

    if (AddPounceComplementsConstraint(prob, 0, 1) != POUNCE_COMPLEMENTS_PAIR_LOWER
        || AddPounceComplementsConstraint(prob, 1, 2) != POUNCE_COMPLEMENTS_PAIR_LOWER) {
        return fail(prob, "AddPounceComplementsConstraint");
    }
    if (!show_rows(prob, POUNCE_BOUNDS_VIEW_USER, "user")
        || !show_rows(prob, POUNCE_BOUNDS_VIEW_EFFECTIVE, "effective")) {
        return fail(prob, "GetPounceBounds");
    }

    ipnumber obj;
    const ipnumber first[N] = {1.0, 1.0, 0.0};
    printf("solve:\n");
    if (!solve_to(prob, first, &obj)) return fail(prob, "first solve");

    /* Refused: y1 gaining an upper bound would move it from "lower only" to
     * "boxed" in the MOI table. The batch's first entry is fine; the second
     * is reported, and neither is applied. */
    {
        const ipindex idx[2] = {0, 1};
        const ipnumber lo[2] = {0.0, 0.0};
        const ipnumber hi[2] = {1.0, 5.0};
        ipindex bad = -1;
        int rc = SetPounceBounds(prob, 2, idx, lo, hi, 0, NULL, NULL, NULL, &bad);
        printf("refused: y1 <= 5 -> %d at entry %d\n", rc, (int)bad);
        if (rc != POUNCE_BOUNDS_ERR_COMPLEMENTS_CLASS || bad != 1) {
            return fail(prob, "expected POUNCE_BOUNDS_ERR_COMPLEMENTS_CLASS at 1");
        }
    }
    /* Refused: F >= 0 is the solve's view of row 0, not an input. */
    {
        const ipindex idx[1] = {0};
        const ipnumber lo[1] = {0.0};
        ipindex bad = -1;
        int rc = SetPounceBounds(prob, 0, NULL, NULL, NULL, 1, idx, lo, NULL, &bad);
        printf("refused: g_L[0] = 0 -> %d at entry %d\n", rc, (int)bad);
        if (rc != POUNCE_BOUNDS_ERR_COMPLEMENTS_ROW || bad != 0) {
            return fail(prob, "expected POUNCE_BOUNDS_ERR_COMPLEMENTS_ROW at 0");
        }
    }

    /* Accepted: x's box, one entry. */
    {
        const ipindex idx[1] = {0};
        const ipnumber lo[1] = {0.25};
        const ipnumber hi[1] = {0.75};
        if (SetPounceBounds(prob, 1, idx, lo, hi, 0, NULL, NULL, NULL, NULL)
            != POUNCE_BOUNDS_OK) {
            return fail(prob, "SetPounceBounds");
        }
    }
    const ipnumber second[N] = {0.75, 0.875, 0.125};
    printf("solve after x in [0.25, 0.75]:\n");
    if (!solve_to(prob, second, &obj) || fabs(obj - 3.0 / 32.0) > 1e-8) {
        return fail(prob, "second solve");
    }

    FreeIpoptProblem(prob);
    printf("bounds_update example: re-solved to (3/4, 7/8, 1/8) after an in-place update, 2 refusals\n");
    return 0;
}
