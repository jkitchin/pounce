/*
 * complementarity.c — declared complementarity through the C API (gh#1147).
 *
 * QpecSmall, the model of python/tests/test_complementarity.py and
 * crates/pounce-cli/tests/fixtures_mpcc/qpec_small.nl:
 *
 *     min (x-1)^2 + (y1-1)^2 + y2^2,   0 <= x <= 2,
 *         0 <= y1  _|_  2 y1 - 1 - x >= 0,      (row 0, variable 1)
 *         0 <= y2  _|_  x + 2 y2 - 1 >= 0.      (row 1, variable 2)
 *
 * Answer (1, 1, 0), the second pair biactive there; strongly stationary.
 *
 * The two pairs are declared the two ways the C API offers:
 *
 *   - row 0 is created with its native side (g_L = 0) and paired with y1
 *     by AddPounceComplementarityPair, as Python's Problem(complementarity=)
 *     and the .nl reader would declare it;
 *   - row 1 is created FREE and paired with y2 by
 *     AddPounceComplementsConstraint, MOI's [F, x] in Complements: y2 has
 *     only a lower bound, so the helper writes F >= 0 (g_L[1] = 0) into the
 *     handle's bounds and declares the native pair. The model is then
 *     QpecSmall exactly.
 *
 * Then: a few refusals (each returns a negative code and changes nothing),
 * the session refusal, IpoptSolve through the complementarity driver, the
 * three outcome getters, and the JSON solve report with its
 * "complementarity" block.
 *
 * Build and run every example in this directory with the strict flags the
 * test suite uses (`-std=c99 -Wall -Wextra -Werror -pedantic`):
 *
 *     make c-api-examples
 *
 * `tests/c_examples_build_and_run.rs` compiles and runs this file on every
 * `cargo test -p pounce-cinterface`, and checks the last line it prints.
 * The JSON report is written next to the binary (argv[0] + ".report.json").
 */

#include "pounce.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    fprintf(stderr, "complementarity.c: %s\n", what);
    FreeIpoptProblem(prob);
    return 1;
}

/* GetPounceComplementarityText into a malloc'd buffer sized by a first call
 * (snprintf contract). NULL when there is no such text. */
static char *cc_text(IpoptProblem prob, int which)
{
    ipindex len = GetPounceComplementarityText(prob, which, NULL, 0);
    if (len < 0) return NULL;
    char *buf = malloc((size_t)len + 1);
    if (buf == NULL) return NULL;
    if (GetPounceComplementarityText(prob, which, buf, len + 1) != len) {
        free(buf);
        return NULL;
    }
    return buf;
}

static const char *class_name(ipindex c)
{
    switch (c) {
    case POUNCE_CC_CLASS_S:            return "S";
    case POUNCE_CC_CLASS_W:            return "W";
    case POUNCE_CC_CLASS_NONE:         return "none";
    case POUNCE_CC_CLASS_NOT_SEARCHED: return "not_searched";
    default:                           return "absent";
    }
}

/* ---- driver -------------------------------------------------------------- */

int main(int argc, char **argv)
{
    double x_l[N] = {0.0, 0.0, 0.0};
    double x_u[N] = {2.0, INF, INF};
    double g_l[M] = {0.0, -INF};   /* row 1 free: MOI's F */
    double g_u[M] = {INF, INF};

    (void)argc;

    IpoptProblem prob = CreateIpoptProblem(
        N, x_l, x_u, M, g_l, g_u,
        /* nele_jac */ 4, /* nele_hess */ 3, /* index_style */ 0,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h);
    if (prob == NULL) {
        fprintf(stderr, "CreateIpoptProblem failed\n");
        return 1;
    }
    AddIpoptIntOption(prob, "print_level", 0);

    /* Before any solve there is no driver record. The outcome structs are
     * extensible: struct_size = sizeof first, or the getter refuses. */
    PounceComplementarityStats st;
    st.struct_size = sizeof st;
    if (GetPounceComplementarityStats(prob, &st))
        return fail(prob, "stats reported before any solve");

    /* 1. Native pair: row 0 has one finite side, y1 one finite bound. */
    int rc = AddPounceComplementarityPair(prob, 0, 1);
    if (rc != POUNCE_OK) return fail(prob, "native pair (0, 1) refused");

    /* 2. MOI pair on the free row 1: y2 >= 0 only, so F >= 0 is written. */
    rc = AddPounceComplementsConstraint(prob, 1, 2);
    if (rc != POUNCE_COMPLEMENTS_PAIR_LOWER)
        return fail(prob, "Complements (1, 2) did not return PAIR_LOWER");

    /* 3. Refusals: each is a negative code and leaves the two pairs. */
    if (AddPounceComplementarityPair(prob, 1, 1) != POUNCE_ERR_VAR_ALREADY_PAIRED)
        return fail(prob, "y1 paired twice was not ERR_VAR_ALREADY");
    if (AddPounceComplementarityPair(prob, M, 0) != POUNCE_ERR_OUT_OF_RANGE)
        return fail(prob, "row m was not ERR_OUT_OF_RANGE");
    /* Row 1 was used in step 2 (and bounded by it): already paired. */
    if (AddPounceComplementsConstraint(prob, 1, 0) != POUNCE_ERR_ROW_ALREADY_PAIRED)
        return fail(prob, "used row 1 was not ERR_ROW_ALREADY_PAIRED");

    ipindex rows[2], vars[2];
    if (GetPounceComplementarityPairCount(prob) != 2
        || !GetPounceComplementarityPairs(prob, rows, vars)
        || rows[0] != 0 || vars[0] != 1 || rows[1] != 1 || vars[1] != 2)
        return fail(prob, "declared pairs are not (0, 1), (1, 2)");
    printf("declared pairs: (%d, %d), (%d, %d)\n",
           rows[0], vars[0], rows[1], vars[1]);

    /* 4. Sessions refuse pairs: NULL, and the handle stays the caller's. */
    IpoptProblem keep = prob;
    IpoptSolver sol = IpoptCreateSolver(&prob);
    if (sol != NULL || prob != keep) {
        if (sol != NULL) IpoptFreeSolver(sol);
        fprintf(stderr, "IpoptCreateSolver accepted a problem with pairs\n");
        return 1;
    }

    /* 5. Solve through the complementarity driver. */
    double x[N] = {0.0, 0.5, 0.5};
    double g[M], obj = 0.0;
    int status = IpoptSolve(prob, x, g, &obj, NULL, NULL, NULL, NULL);
    printf("status = %d   x = (%.6f, %.6f, %.6f)   f = %.3e   "
           "reported solve: %d iterations\n",
           status, x[0], x[1], x[2], obj, GetIpoptIterCount(prob));
    if (status != Solve_Succeeded) return fail(prob, "solve did not succeed");
    if (fabs(x[0] - 1.0) > 1e-6 || fabs(x[1] - 1.0) > 1e-6 || fabs(x[2]) > 1e-6)
        return fail(prob, "x is not (1, 1, 0)");

    /* 6. What the driver did. */
    if (!GetPounceComplementarityStats(prob, &st))
        return fail(prob, "no stats after a driver solve");
    printf("strategy %s, accepted by \"%s\", %d lowered solves, "
           "%d iterations in all; certificate %s, %d pairs, %d biactive\n",
           st.strategy, st.accepted_by, st.n_solves, st.total_iterations,
           class_name(st.certificate_class), st.n_pairs, st.n_biactive);
    if (st.reported != POUNCE_CC_REPORTED_ANSWER
        || st.certificate_class != POUNCE_CC_CLASS_S || st.n_pairs != 2)
        return fail(prob, "not an S answer over 2 pairs");

    for (ipindex k = 0; k < st.n_solves; k++) {
        PounceComplementaritySolve s;
        s.struct_size = sizeof s;
        if (!GetPounceComplementaritySolve(prob, k, &s))
            return fail(prob, "missing lowered-solve record");
        if (s.relaxed)
            printf("  solve %d: %-22s tau = %.1e  status %d  %d iterations\n",
                   k, s.stage, s.tau, s.status, s.iterations);
        else
            printf("  solve %d: %-22s             status %d  %d iterations\n",
                   k, s.stage, s.status, s.iterations);
    }
    {
        PounceComplementaritySolve s;
        s.struct_size = sizeof s;
        if (GetPounceComplementaritySolve(prob, st.n_solves, &s))
            return fail(prob, "record k = n_solves accepted");
    }

    char *summary = cc_text(prob, POUNCE_CC_TEXT_SUMMARY);
    if (summary == NULL) return fail(prob, "no summary text");
    printf("summary: %s\n", summary);
    free(summary);
    /* No B-check ran on an S answer, so there is no B-check reason. */
    if (GetPounceComplementarityText(prob, POUNCE_CC_TEXT_B_REASON, NULL, 0) != -1)
        return fail(prob, "B-check reason present on an S answer");

    /* 7. The JSON report carries the same outcome as its complementarity
     *    block. */
    char path[4096];
    if (snprintf(path, sizeof path, "%s.report.json", argv[0]) >= (int)sizeof path)
        return fail(prob, "report path too long");
    if (!IpoptWriteSolveReport(prob, path, "summary"))
        return fail(prob, "IpoptWriteSolveReport failed");
    {
        FILE *fp = fopen(path, "rb");
        char chunk[1 << 16];
        size_t got;
        if (fp == NULL) return fail(prob, "cannot reopen the report");
        got = fread(chunk, 1, sizeof chunk - 1, fp);
        fclose(fp);
        chunk[got] = '\0';
        if (strstr(chunk, "\"complementarity\"") == NULL)
            return fail(prob, "report has no complementarity block");
    }

    FreeIpoptProblem(prob);
    printf("complementarity example: S-stationary answer (1, 1, 0) certified\n");
    return 0;
}
