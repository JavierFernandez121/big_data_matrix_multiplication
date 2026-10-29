/*
 * Assignment 2, Part B: sparse (CSR) versus dense matrix multiplication in C.
 *
 * The program generates A and B itself, so that every method uses exactly the
 * same logical matrices: for each element, in row-major order, a uniform
 * number u is drawn; the element is nonzero if u < density, and its value is
 * then drawn uniformly in [0, 1). A is generated first, then B, from one
 * splitmix64 stream seeded with (seed, n, density). Generation is not timed.
 *
 * Methods (timed region in brackets):
 *   dense    dense arrays with zeros, cache-blocked ikj multiply (block 64),
 *            the fastest method of Part A [zeroing C + multiplication]
 *   csr      A and B generated directly in CSR; CSR x CSR -> CSR with
 *            Gustavson's row-by-row algorithm [multiplication, including
 *            building the CSR arrays of C]
 *   convert  dense arrays generated, then converted to CSR
 *            [conversion of A and B from dense to CSR] -- the conversion cost
 *            is measured separately, not included in "csr"
 *
 * Usage:
 *   sparse <dense|csr|convert> <n> <density_percent> [--seed S] [--warmup W]
 *          [--reps R] [--dump DIR]
 *   --dump writes A, B and C as dense float64 files (used by the tests).
 *
 * Output, one line per run:
 *   method=.. n=.. density=.. phase=.. rep=.. time_s=.. checksum=..
 *   nnz_a=.. nnz_b=.. nnz_c=.. struct_mb=..
 * struct_mb is the size of the data structures of A, B and C:
 *   dense: 3 * 8 n^2 bytes; CSR: 12 bytes per nonzero + 4 (n + 1) per matrix.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

static long long now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (long long)((double)counter.QuadPart * 1e9 / (double)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (p == NULL) {
        fprintf(stderr, "Error: out of memory (%zu bytes)\n", bytes);
        exit(1);
    }
    return p;
}

/* ---------------- random generator (splitmix64) ---------------- */

static uint64_t rng_state;

static double next_uniform(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0); /* [0, 1) */
}

/* ---------------- CSR matrix ---------------- */

typedef struct {
    int n;
    size_t nnz, cap;
    int *row_ptr;   /* n + 1 entries */
    int *col;       /* nnz entries   */
    double *val;    /* nnz entries   */
} Csr;

static void csr_init(Csr *m, int n, size_t cap) {
    m->n = n;
    m->nnz = 0;
    m->cap = cap ? cap : 16;
    m->row_ptr = xmalloc((size_t)(n + 1) * sizeof(int));
    m->col = xmalloc(m->cap * sizeof(int));
    m->val = xmalloc(m->cap * sizeof(double));
    m->row_ptr[0] = 0;
}

static void csr_push(Csr *m, int j, double v) {
    if (m->nnz == m->cap) {
        m->cap *= 2;
        m->col = realloc(m->col, m->cap * sizeof(int));
        m->val = realloc(m->val, m->cap * sizeof(double));
        if (m->col == NULL || m->val == NULL) {
            fprintf(stderr, "Error: out of memory\n");
            exit(1);
        }
    }
    m->col[m->nnz] = j;
    m->val[m->nnz] = v;
    m->nnz++;
}

static void csr_free(Csr *m) {
    free(m->row_ptr);
    free(m->col);
    free(m->val);
}

static double csr_bytes(const Csr *m) {
    return 12.0 * (double)m->nnz + 4.0 * (m->n + 1);
}

/* ---------------- generation (not timed) ---------------- */

static double *generate_dense(int n, double d) {
    double *m = xmalloc((size_t)n * n * sizeof(double));
    for (size_t i = 0; i < (size_t)n * n; i++)
        m[i] = next_uniform() < d ? next_uniform() : 0.0;
    return m;
}

static void generate_csr(Csr *m, int n, double d) {
    csr_init(m, n, (size_t)((double)n * n * d * 1.1) + 16);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            if (next_uniform() < d)
                csr_push(m, j, next_uniform());
        m->row_ptr[i + 1] = (int)m->nnz;
    }
}

/* ---------------- kernels (timed) ---------------- */

static void dense_blocked(const double *a, const double *b, double *c, int n, int bs) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bs)
        for (int kk = 0; kk < n; kk += bs)
            for (int jj = 0; jj < n; jj += bs) {
                int i_end = ii + bs < n ? ii + bs : n;
                int k_end = kk + bs < n ? kk + bs : n;
                int j_end = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < i_end; i++)
                    for (int k = kk; k < k_end; k++) {
                        double aik = a[i * n + k];
                        for (int j = jj; j < j_end; j++)
                            c[i * n + j] += aik * b[k * n + j];
                    }
            }
}

static void dense_to_csr(const double *m, int n, Csr *out) {
    size_t nnz = 0;
    for (size_t i = 0; i < (size_t)n * n; i++)
        nnz += m[i] != 0.0;
    csr_init(out, n, nnz);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            if (m[(size_t)i * n + j] != 0.0)
                csr_push(out, j, m[(size_t)i * n + j]);
        out->row_ptr[i + 1] = (int)out->nnz;
    }
}

/*
 * Gustavson: row i of C = sum over nonzeros a_ik of a_ik * (row k of B).
 * acc holds the partial row, marker[j] == i means column j is already in row i.
 * Columns inside each row of C are left in the order they first appear.
 */
static void csr_multiply(const Csr *a, const Csr *b, Csr *c, double *acc, int *marker,
                         int *cols) {
    int n = a->n;
    c->nnz = 0;
    c->row_ptr[0] = 0;
    for (int j = 0; j < n; j++)
        marker[j] = -1;
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int p = a->row_ptr[i]; p < a->row_ptr[i + 1]; p++) {
            int k = a->col[p];
            double aik = a->val[p];
            for (int q = b->row_ptr[k]; q < b->row_ptr[k + 1]; q++) {
                int j = b->col[q];
                if (marker[j] != i) {
                    marker[j] = i;
                    acc[j] = aik * b->val[q];
                    cols[count++] = j;
                } else {
                    acc[j] += aik * b->val[q];
                }
            }
        }
        for (int t = 0; t < count; t++)
            csr_push(c, cols[t], acc[cols[t]]);
        c->row_ptr[i + 1] = (int)c->nnz;
    }
}

/* ---------------- helpers ---------------- */

static void write_dense(const char *dir, const char *name, const double *m, int n) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (f == NULL || fwrite(m, sizeof(double), (size_t)n * n, f) != (size_t)n * n) {
        fprintf(stderr, "Error: cannot write %s\n", path);
        exit(1);
    }
    fclose(f);
}

static double *csr_to_dense(const Csr *m) {
    int n = m->n;
    double *d = calloc((size_t)n * n, sizeof(double));
    if (d == NULL) { fprintf(stderr, "Error: out of memory\n"); exit(1); }
    for (int i = 0; i < n; i++)
        for (int p = m->row_ptr[i]; p < m->row_ptr[i + 1]; p++)
            d[(size_t)i * n + m->col[p]] = m->val[p];
    return d;
}

static size_t count_nonzero(const double *m, int n) {
    size_t nnz = 0;
    for (size_t i = 0; i < (size_t)n * n; i++)
        nnz += m[i] != 0.0;
    return nnz;
}

int main(int argc, char **argv) {
    const char *positional[3];
    int n_positional = 0, warmup = 0, reps = 1;
    uint64_t seed = 42;
    const char *dump = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump = argv[++i];
        else if (n_positional < 3) positional[n_positional++] = argv[i];
        else n_positional = 4;
    }
    if (n_positional != 3 || (strcmp(positional[0], "dense") != 0 &&
        strcmp(positional[0], "csr") != 0 && strcmp(positional[0], "convert") != 0)) {
        printf("Usage: %s <dense|csr|convert> <n> <density_percent> [--seed S] "
               "[--warmup W] [--reps R] [--dump DIR]\n", argv[0]);
        return 1;
    }
    const char *method = positional[0];
    int n = atoi(positional[1]);
    double density_pct = atof(positional[2]);
    double d = density_pct / 100.0;
    /* One stream per (seed, n, density): A first, then B */
    rng_state = seed * 1000003ULL + (uint64_t)n * 7919ULL + (uint64_t)(density_pct * 1e6);

    double *ad = NULL, *bd = NULL, *cd = NULL;
    Csr a, b, c;
    double *acc = NULL;
    int *marker = NULL, *cols = NULL;
    if (strcmp(method, "csr") == 0) {
        generate_csr(&a, n, d);
        generate_csr(&b, n, d);
        csr_init(&c, n, 16);
        acc = xmalloc((size_t)n * sizeof(double));
        marker = xmalloc((size_t)n * sizeof(int));
        cols = xmalloc((size_t)n * sizeof(int));
    } else {
        ad = generate_dense(n, d);
        bd = generate_dense(n, d);
        if (strcmp(method, "dense") == 0)
            cd = xmalloc((size_t)n * n * sizeof(double));
    }

    const char *phases[2] = {"warmup", "measure"};
    int counts[2] = {warmup, reps};
    for (int p = 0; p < 2; p++)
        for (int rep = 1; rep <= counts[p]; rep++) {
            size_t nnz_a, nnz_b, nnz_c;
            double checksum = 0.0, struct_mb;
            long long start = now_ns(), elapsed;

            if (strcmp(method, "dense") == 0) {
                dense_blocked(ad, bd, cd, n, 64);
                elapsed = now_ns() - start;
                for (size_t i = 0; i < (size_t)n * n; i++) checksum += cd[i];
                nnz_a = count_nonzero(ad, n);
                nnz_b = count_nonzero(bd, n);
                nnz_c = count_nonzero(cd, n);
                struct_mb = 24.0 * n * (double)n / 1e6;
            } else if (strcmp(method, "csr") == 0) {
                csr_multiply(&a, &b, &c, acc, marker, cols);
                elapsed = now_ns() - start;
                for (size_t i = 0; i < c.nnz; i++) checksum += c.val[i];
                nnz_a = a.nnz; nnz_b = b.nnz; nnz_c = c.nnz;
                struct_mb = (csr_bytes(&a) + csr_bytes(&b) + csr_bytes(&c)) / 1e6;
            } else { /* convert */
                Csr ca, cb;
                dense_to_csr(ad, n, &ca);
                dense_to_csr(bd, n, &cb);
                elapsed = now_ns() - start;
                for (size_t i = 0; i < ca.nnz; i++) checksum += ca.val[i];
                for (size_t i = 0; i < cb.nnz; i++) checksum += cb.val[i];
                nnz_a = ca.nnz; nnz_b = cb.nnz; nnz_c = 0;
                struct_mb = (csr_bytes(&ca) + csr_bytes(&cb)) / 1e6;
                csr_free(&ca);
                csr_free(&cb);
            }
            printf("method=%s n=%d density=%g phase=%s rep=%d time_s=%.9f checksum=%.10e "
                   "nnz_a=%zu nnz_b=%zu nnz_c=%zu struct_mb=%.3f\n",
                   method, n, density_pct, phases[p], rep, elapsed / 1e9, checksum,
                   nnz_a, nnz_b, nnz_c, struct_mb);
            fflush(stdout);
        }

    if (dump != NULL && strcmp(method, "convert") != 0) {
        if (strcmp(method, "csr") == 0) {
            double *da = csr_to_dense(&a), *db = csr_to_dense(&b), *dc = csr_to_dense(&c);
            write_dense(dump, "A.bin", da, n);
            write_dense(dump, "B.bin", db, n);
            write_dense(dump, "C.bin", dc, n);
            free(da); free(db); free(dc);
        } else {
            write_dense(dump, "A.bin", ad, n);
            write_dense(dump, "B.bin", bd, n);
            write_dense(dump, "C.bin", cd, n);
        }
    }
    return 0;
}