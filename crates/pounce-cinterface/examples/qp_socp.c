/*
 * qp_socp.c — the matrix-form convex QP / conic entry (PounceQp*, gh#1147).
 *
 * Standard form (pounce.h, "Matrix-form convex QP / conic entry"):
 *
 *     minimize    1/2 x'P x + c'x
 *     subject to  A x = b,   G x <= h  (G x <=_K h with cones),   lb <= x <= ub
 *
 * Four small problems, every answer known by hand and checked here:
 *
 *   1. A convex QP with one equality, one inequality and a lower box:
 *          min 1/2 (x0^2 + x1^2) - x0 - 2 x1
 *          s.t. x0 - x1 = 0,  x0 + x1 <= 1,  x >= 0
 *      x = (1/2, 1/2), obj = -5/4, y = -1/2, z = 1 (the multiplier signs
 *      of P x + c + A'y + G'z - z_lb + z_ub = 0). Solution, residuals,
 *      certificate (none), and the JSON report (engine "cvx-qp").
 *   2. A second-order-cone program: the distance from (2, -1) to the line
 *      x1 + x2 = 0, as  min t  s.t.  (t, x1 - 2, x2 + 1) in SOC(3).
 *      t = 1/sqrt(2), x = (3/2, -3/2); the report's engine is "cvx-qcqp".
 *   3. A primal-infeasible LP (x <= -1 with x >= 0): status
 *      POUNCE_QP_STATUS_PRIMAL_INFEASIBLE and a Farkas certificate.
 *   4. Refusals, each a negative code before any engine runs: an indefinite
 *      P under method=ipm (POUNCE_QP_ERR_INDEFINITE), cones that do not
 *      partition the G rows (POUNCE_QP_ERR_CONES), a power-cone exponent
 *      outside (0, 1) (POUNCE_QP_ERR_ARGUMENT); and the getters return FALSE
 *      after a refused solve.
 *
 * Build and run every example in this directory with the strict flags the
 * test suite uses (`-std=c99 -Wall -Wextra -Werror -pedantic`):
 *
 *     make c-api-examples
 *
 * `tests/c_examples_build_and_run.rs` compiles and runs this file on every
 * `cargo test -p pounce-cinterface`, and checks the last line it prints.
 * The JSON reports are written next to the binary (argv[0] + ".qp.json",
 * argv[0] + ".socp.json").
 */

#include "pounce.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TOL 1e-6

static int fail(PounceQp qp, const char *what)
{
    fprintf(stderr, "qp_socp example: %s\n", what);
    PounceFreeQp(qp);
    return 1;
}

static int close_to(ipnumber got, ipnumber want)
{
    return fabs(got - want) <= TOL * (1.0 + fabs(want));
}

/* Write the report to `path` and check it names `engine` (quoted). */
static int report_names_engine(PounceQp qp, const char *path, const char *engine)
{
    char chunk[1 << 16];
    char quoted[64];
    size_t got;
    FILE *fp;
    if (!PounceQpWriteSolveReport(qp, path, "summary")) return 0;
    fp = fopen(path, "rb");
    if (fp == NULL) return 0;
    got = fread(chunk, 1, sizeof chunk - 1, fp);
    fclose(fp);
    chunk[got] = '\0';
    if (snprintf(quoted, sizeof quoted, "\"%s\"", engine) >= (int)sizeof quoted)
        return 0;
    return strstr(chunk, quoted) != NULL
        && strstr(chunk, "pounce.solve-report/v1") != NULL;
}

/* ---- 1. Convex QP -------------------------------------------------------- */

static int convex_qp(const char *report_path)
{
    /* Lower triangle of P = I; 0-based triplets (index_style 0). */
    const ipindex  p_row[] = {0, 1}, p_col[] = {0, 1};
    const ipnumber p_val[] = {1.0, 1.0}, c[] = {-1.0, -2.0};
    const ipindex  a_row[] = {0, 0}, a_col[] = {0, 1};
    const ipnumber a_val[] = {1.0, -1.0}, b[] = {0.0};
    const ipindex  g_row[] = {0, 0}, g_col[] = {0, 1};
    const ipnumber g_val[] = {1.0, 1.0}, h[] = {1.0};
    const ipnumber lb[] = {0.0, 0.0};
    ipnumber x[2], y[1], z[1], z_lb[2], z_ub[2], obj;
    ipindex iters;
    PounceQpResiduals res;
    PounceQpCertificate cert;
    int status;

    PounceQp qp = PounceCreateQp(2, 1, 1, 0);
    if (qp == NULL) return fail(qp, "PounceCreateQp failed");
    if (!PounceQpSetObjective(qp, 2, p_row, p_col, p_val, c)
        || !PounceQpSetEqualities(qp, 2, a_row, a_col, a_val, b)
        || !PounceQpSetInequalities(qp, 2, g_row, g_col, g_val, h)
        || !PounceQpSetBounds(qp, lb, NULL)) /* NULL: no upper bounds */
        return fail(qp, "a QP builder refused valid input");

    /* Nothing to read before a solve. */
    if (PounceQpGetSolution(qp, x, NULL, NULL, NULL, NULL, &obj, &iters))
        return fail(qp, "solution reported before any solve");

    status = PounceQpSolve(qp);
    if (status != POUNCE_QP_STATUS_OPTIMAL) return fail(qp, "QP not optimal");
    if (!PounceQpGetSolution(qp, x, y, z, z_lb, z_ub, &obj, &iters))
        return fail(qp, "no QP solution");
    printf("QP:   x = (%.6f, %.6f), obj = %.6f, y = %.6f, z = %.6f, %d iterations\n",
           x[0], x[1], obj, y[0], z[0], iters);
    if (!close_to(x[0], 0.5) || !close_to(x[1], 0.5) || !close_to(obj, -1.25)
        || !close_to(y[0], -0.5) || !close_to(z[0], 1.0)
        || !close_to(z_lb[0], 0.0) || !close_to(z_lb[1], 0.0))
        return fail(qp, "QP answer is not x = (1/2, 1/2), y = -1/2, z = 1");

    if (!PounceQpGetResiduals(qp, &res)) return fail(qp, "no QP residuals");
    if (!(res.kkt_error <= TOL && res.kkt_error_raw <= TOL))
        return fail(qp, "QP KKT residual above tolerance");
    if (!PounceQpGetCertificate(qp, &cert)) return fail(qp, "no QP certificate");
    if (cert.kind != POUNCE_QP_CERT_NONE || !isnan(cert.scale))
        return fail(qp, "certificate on an optimal QP");

    if (!report_names_engine(qp, report_path, "cvx-qp"))
        return fail(qp, "QP report missing or engine is not cvx-qp");
    PounceFreeQp(qp);
    return 0;
}

/* ---- 2. Second-order cone ------------------------------------------------ */

static int socp(const char *report_path)
{
    /* Variables (t, x1, x2). min t, x1 + x2 = 0, s = h - G x in SOC(3). */
    const ipnumber c[] = {1.0, 0.0, 0.0};
    const ipindex  a_row[] = {0, 0}, a_col[] = {1, 2};
    const ipnumber a_val[] = {1.0, 1.0}, b[] = {0.0};
    const ipindex  g_row[] = {0, 1, 2}, g_col[] = {0, 1, 2};
    const ipnumber g_val[] = {-1.0, -1.0, -1.0}, h[] = {0.0, -2.0, 1.0};
    ipnumber x[3], obj;
    ipindex iters;
    PounceQpResiduals res;

    PounceQp qp = PounceCreateQp(3, 1, 3, 0);
    if (qp == NULL) return fail(qp, "PounceCreateQp failed");
    if (!PounceQpSetObjective(qp, 0, NULL, NULL, NULL, c) /* P = 0 */
        || !PounceQpSetEqualities(qp, 2, a_row, a_col, a_val, b)
        || !PounceQpSetInequalities(qp, 3, g_row, g_col, g_val, h))
        return fail(qp, "an SOCP builder refused valid input");
    if (PounceQpAddCone(qp, POUNCE_CONE_SOC, 3, 0.0) != 0)
        return fail(qp, "SOC cone not added at position 0");

    if (PounceQpSolve(qp) != POUNCE_QP_STATUS_OPTIMAL)
        return fail(qp, "SOCP not optimal");
    if (!PounceQpGetSolution(qp, x, NULL, NULL, NULL, NULL, &obj, &iters))
        return fail(qp, "no SOCP solution");
    printf("SOCP: t = %.6f, x = (%.6f, %.6f), %d iterations\n", x[0], x[1], x[2], iters);
    if (!close_to(x[0], 1.0 / sqrt(2.0)) || !close_to(obj, 1.0 / sqrt(2.0))
        || !close_to(x[1], 1.5) || !close_to(x[2], -1.5))
        return fail(qp, "SOCP answer is not t = 1/sqrt(2), x = (3/2, -3/2)");
    if (!PounceQpGetResiduals(qp, &res) || !(res.kkt_error_raw <= TOL))
        return fail(qp, "SOCP KKT residual missing or above tolerance");

    if (!report_names_engine(qp, report_path, "cvx-qcqp"))
        return fail(qp, "SOCP report missing or engine is not cvx-qcqp");
    PounceFreeQp(qp);
    return 0;
}

/* ---- 3. Primal infeasible ------------------------------------------------ */

static int infeasible(void)
{
    const ipnumber c[] = {1.0};
    const ipindex  g_row[] = {0}, g_col[] = {0};
    const ipnumber g_val[] = {1.0}, h[] = {-1.0}; /* x <= -1 */
    const ipnumber lb[] = {0.0};                 /* x >= 0  */
    PounceQpCertificate cert;

    PounceQp qp = PounceCreateQp(1, 0, 1, 0);
    if (qp == NULL) return fail(qp, "PounceCreateQp failed");
    if (!PounceQpSetObjective(qp, 0, NULL, NULL, NULL, c)
        || !PounceQpSetEqualities(qp, 0, NULL, NULL, NULL, NULL) /* no rows */
        || !PounceQpSetInequalities(qp, 1, g_row, g_col, g_val, h)
        || !PounceQpSetBounds(qp, lb, NULL))
        return fail(qp, "an LP builder refused valid input");
    if (PounceQpSolve(qp) != POUNCE_QP_STATUS_PRIMAL_INFEASIBLE)
        return fail(qp, "infeasible LP not reported PRIMAL_INFEASIBLE");
    if (!PounceQpGetCertificate(qp, &cert)) return fail(qp, "no certificate");
    printf("LP:   primal infeasible, Farkas ray scale %.3g\n", cert.scale);
    if (cert.kind != POUNCE_QP_CERT_PRIMAL_INFEASIBLE
        || !(cert.scale > 0.0 && isfinite(cert.scale)))
        return fail(qp, "no primal-infeasibility certificate");
    PounceFreeQp(qp);
    return 0;
}

/* ---- 4. Refusals --------------------------------------------------------- */

static int refusals(void)
{
    /* P = diag(1, -1): indefinite, refused under the default method=ipm. */
    const ipindex  p_row[] = {0, 1}, p_col[] = {0, 1};
    const ipnumber p_val[] = {1.0, -1.0}, c[] = {0.0, 0.0};
    const ipindex  g_row[] = {0, 1}, g_col[] = {0, 1};
    const ipnumber g_val[] = {1.0, 1.0}, h[] = {1.0, 1.0};
    ipnumber x[2];
    int rc;

    PounceQp qp = PounceCreateQp(2, 0, 2, 0);
    if (qp == NULL) return fail(qp, "PounceCreateQp failed");
    if (!PounceQpSetObjective(qp, 2, p_row, p_col, p_val, c)
        || !PounceQpSetInequalities(qp, 2, g_row, g_col, g_val, h))
        return fail(qp, "a builder refused valid input");
    if ((rc = PounceQpSolve(qp)) != POUNCE_QP_ERR_INDEFINITE)
        return fail(qp, "indefinite P not refused with POUNCE_QP_ERR_INDEFINITE");
    if (PounceQpGetSolution(qp, x, NULL, NULL, NULL, NULL, NULL, NULL))
        return fail(qp, "solution reported after a refused solve");
    printf("refused: indefinite P under method=ipm (%d)\n", rc);

    /* Under method=active-set the same P is no longer refused. */
    if (!PounceQpSetStrOption(qp, "method", "active-set"))
        return fail(qp, "method=active-set refused");
    if (PounceQpSolve(qp) < 0) return fail(qp, "active-set refused an indefinite P");

    /* One SOC(1) cone over two G rows does not partition them (checked
     * before the PSD screen). */
    if (!PounceQpSetStrOption(qp, "method", "ipm"))
        return fail(qp, "method=ipm refused");
    if (PounceQpAddCone(qp, POUNCE_CONE_SOC, 1, 0.0) != 0)
        return fail(qp, "SOC(1) cone not added");
    if ((rc = PounceQpSolve(qp)) != POUNCE_QP_ERR_CONES)
        return fail(qp, "cones over 1 of 2 rows not refused with POUNCE_QP_ERR_CONES");
    printf("refused: cones covering 1 of 2 inequality rows (%d)\n", rc);

    /* A power-cone exponent must lie in (0, 1); nothing is added. */
    if ((rc = PounceQpAddCone(qp, POUNCE_CONE_POW, 3, 1.5)) != POUNCE_QP_ERR_ARGUMENT)
        return fail(qp, "alpha = 1.5 not refused with POUNCE_QP_ERR_ARGUMENT");
    printf("refused: power cone with alpha = 1.5 (%d)\n", rc);

    PounceFreeQp(qp);
    return 0;
}

int main(int argc, char **argv)
{
    char qp_path[4096], socp_path[4096];
    (void)argc;
    if (snprintf(qp_path, sizeof qp_path, "%s.qp.json", argv[0]) >= (int)sizeof qp_path
        || snprintf(socp_path, sizeof socp_path, "%s.socp.json", argv[0])
               >= (int)sizeof socp_path) {
        fprintf(stderr, "qp_socp example: report path too long\n");
        return 1;
    }
    if (convex_qp(qp_path) || socp(socp_path) || infeasible() || refusals())
        return 1;
    printf("qp_socp example: QP (1/2, 1/2), SOC distance 1/sqrt(2), "
           "infeasibility certified, 3 refusals\n");
    return 0;
}
