/*
 * A minimal custom linear solver for pounce (linear_solver=custom, gh#1175).
 *
 * Build as a shared library and point pounce at it:
 *
 *     cc -std=c99 -O2 -shared -fPIC -I<pounce>/include dense_solver.c \
 *        -o libdense_solver.so -lm
 *
 *     linear_solver                custom
 *     custom_linear_solver_lib     ./libdense_solver.so
 *
 * It factors the KKT matrix densely by cyclic Jacobi eigendecomposition,
 * A = V diag(lam) V^T, which gives the exact inertia and solves by
 * x = V diag(1/lam) V^T b. That is O(dim^3) per factorization and only
 * sensible for small systems; it is here to show the contract, not to be
 * fast. A real solver keeps the same callbacks and replaces the numerics.
 *
 * It also records what pounce told it (the KKT layout, the metadata keys
 * and their values, how many factorizations it did) and exports getters
 * for that. pounce's own tests read them; they also show a solver author
 * exactly what arrives.
 */
#include "pounce.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ipindex  dim, nnz;
    ipindex *irow, *jcol;  /* 1-based triplets, copied */
    double  *a;            /* dim*dim scratch, then eigenvectors V */
    double  *lam;          /* eigenvalues */
    int      singular;
} Solver;

/* ---- what pounce told us (process-global: tests read it) ---- */
#define MAX_KEYS 8
#define MAX_LEN  256
static ipindex g_layout[4] = {-1, -1, -1, -1};
static int     g_n_keys = 0;
static char    g_keys[MAX_KEYS][64];
static ipindex g_vals[MAX_KEYS][MAX_LEN];
static ipindex g_lens[MAX_KEYS];
static long    g_factors = 0;
static char    g_options[256];

static void *create(const char *options) {
    Solver *s = (Solver *)calloc(1, sizeof(Solver));
    if (options) {
        strncpy(g_options, options, sizeof(g_options) - 1);
        g_options[sizeof(g_options) - 1] = '\0';
    }
    return s;
}

static void destroy(void *state) {
    Solver *s = (Solver *)state;
    free(s->irow);
    free(s->jcol);
    free(s->a);
    free(s->lam);
    free(s);
}

static int set_layout(void *state, ipindex n_x, ipindex n_s, ipindex n_c, ipindex n_d) {
    (void)state;
    g_layout[0] = n_x;
    g_layout[1] = n_s;
    g_layout[2] = n_c;
    g_layout[3] = n_d;
    return 0;
}

static int set_metadata(void *state, const char *key, const ipindex *values, ipindex len) {
    int k;
    (void)state;
    for (k = 0; k < g_n_keys; k++)
        if (strcmp(g_keys[k], key) == 0) break;
    if (k == g_n_keys) {
        if (g_n_keys == MAX_KEYS) return 0;
        g_n_keys++;
        strncpy(g_keys[k], key, sizeof(g_keys[k]) - 1);
        g_keys[k][sizeof(g_keys[k]) - 1] = '\0';
    }
    g_lens[k] = len < MAX_LEN ? len : MAX_LEN;
    memcpy(g_vals[k], values, (size_t)g_lens[k] * sizeof(ipindex));
    return 0;
}

static int initialize_structure(void *state, ipindex dim, ipindex nnz,
                                const ipindex *irow, const ipindex *jcol) {
    Solver *s = (Solver *)state;
    s->dim = dim;
    s->nnz = nnz;
    s->irow = (ipindex *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(ipindex));
    s->jcol = (ipindex *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(ipindex));
    s->a = (double *)malloc((size_t)(dim > 0 ? dim * dim : 1) * sizeof(double));
    s->lam = (double *)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(double));
    if (!s->irow || !s->jcol || !s->a || !s->lam) return -1;
    memcpy(s->irow, irow, (size_t)nnz * sizeof(ipindex));
    memcpy(s->jcol, jcol, (size_t)nnz * sizeof(ipindex));
    return 0;
}

/* Cyclic Jacobi on the dense symmetric s->a; eigenvalues to lam,
 * eigenvectors (columns) back into s->a. */
static void jacobi(Solver *s) {
    ipindex n = s->dim, i, j, k, p, q;
    double *a = s->a;
    double *v = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    int sweep;
    for (i = 0; i < n; i++) v[i * n + i] = 1.0;
    for (sweep = 0; sweep < 100; sweep++) {
        double off = 0.0, tot = 0.0;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++) {
                tot += a[i * n + j] * a[i * n + j];
                if (i != j) off += a[i * n + j] * a[i * n + j];
            }
        if (off <= 1e-30 * (tot > 0 ? tot : 1)) break;
        for (p = 0; p < n; p++)
            for (q = p + 1; q < n; q++) {
                double apq = a[p * n + q], theta, t, c, sn;
                if (fabs(apq) < 1e-300) continue;
                theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                c = 1.0 / sqrt(t * t + 1.0);
                sn = t * c;
                for (k = 0; k < n; k++) { /* columns p, q */
                    double akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - sn * akq;
                    a[k * n + q] = sn * akp + c * akq;
                }
                for (k = 0; k < n; k++) { /* rows p, q */
                    double apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - sn * aqk;
                    a[q * n + k] = sn * apk + c * aqk;
                }
                for (k = 0; k < n; k++) {
                    double vkp = v[k * n + p], vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - sn * vkq;
                    v[k * n + q] = sn * vkp + c * vkq;
                }
            }
    }
    for (i = 0; i < n; i++) s->lam[i] = a[i * n + i];
    memcpy(a, v, (size_t)(n * n) * sizeof(double));
    free(v);
}

static int factor(void *state, const ipnumber *values, ipindex nnz,
                  int check_inertia, ipindex expected_neg, ipindex *n_neg) {
    Solver *s = (Solver *)state;
    ipindex n = s->dim, k, neg = 0;
    double big = 0.0;
    g_factors++;
    memset(s->a, 0, (size_t)(n * n) * sizeof(double));
    for (k = 0; k < nnz; k++) {
        ipindex i = s->irow[k] - 1, j = s->jcol[k] - 1;
        s->a[i * n + j] += values[k];
        if (i != j) s->a[j * n + i] += values[k];
    }
    jacobi(s);
    for (k = 0; k < n; k++)
        if (fabs(s->lam[k]) > big) big = fabs(s->lam[k]);
    s->singular = 0;
    for (k = 0; k < n; k++) {
        if (fabs(s->lam[k]) <= 1e-13 * (big > 1 ? big : 1)) s->singular = 1;
        if (s->lam[k] < 0) neg++;
    }
    *n_neg = neg;
    if (s->singular) return POUNCE_LINSOL_SINGULAR;
    if (check_inertia && neg != expected_neg) return POUNCE_LINSOL_WRONG_INERTIA;
    return POUNCE_LINSOL_SUCCESS;
}

static int solve(void *state, ipindex dim, ipindex nrhs, ipnumber *rhs) {
    Solver *s = (Solver *)state;
    ipindex r, i, k, n = s->dim;
    double *t = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!t || dim != n || s->singular) {
        free(t);
        return -1;
    }
    for (r = 0; r < nrhs; r++) {
        double *b = rhs + (size_t)r * (size_t)n;
        for (k = 0; k < n; k++) { /* t = diag(1/lam) V^T b */
            double acc = 0.0;
            for (i = 0; i < n; i++) acc += s->a[i * n + k] * b[i];
            t[k] = acc / s->lam[k];
        }
        for (i = 0; i < n; i++) { /* b = V t */
            double acc = 0.0;
            for (k = 0; k < n; k++) acc += s->a[i * n + k] * t[k];
            b[i] = acc;
        }
    }
    free(t);
    return 0;
}

static const PounceLinearSolver VTABLE = {
    sizeof(PounceLinearSolver),
    POUNCE_LINEAR_SOLVER_ABI_VERSION,
    "dense-jacobi",
    1, /* provides_inertia */
    create,
    destroy,
    set_layout,
    set_metadata,
    initialize_structure,
    factor,
    solve,
    NULL /* increase_quality */
};

const PounceLinearSolver *pounce_linear_solver(void) { return &VTABLE; }

/* ---- introspection ---- */
void dense_solver_layout(ipindex out[4]) { memcpy(out, g_layout, sizeof(g_layout)); }

ipindex dense_solver_metadata(const char *key, ipindex *out, ipindex cap) {
    int k;
    for (k = 0; k < g_n_keys; k++)
        if (strcmp(g_keys[k], key) == 0) {
            ipindex n = g_lens[k] < cap ? g_lens[k] : cap;
            memcpy(out, g_vals[k], (size_t)n * sizeof(ipindex));
            return g_lens[k];
        }
    return -1;
}

long dense_solver_factor_count(void) { return g_factors; }

const char *dense_solver_options(void) { return g_options; }

void dense_solver_reset(void) {
    int i;
    for (i = 0; i < 4; i++) g_layout[i] = -1;
    g_n_keys = 0;
    g_factors = 0;
    g_options[0] = '\0';
}
