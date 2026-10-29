/*
 * Assignment 2, Part A: three dense matrix multiplication methods in C.
 *
 *   ijk      basic triple loop (same algorithm as Assignment 1, the baseline)
 *   ikj      loop reordering: the two inner loops are swapped, so B and C
 *            are read and written row by row (contiguous memory)
 *   blocked  cache blocking: ikj applied to square blocks of size BS x BS,
 *            so each block is reused while it is still in the cache
 *
 * All methods do the same n^3 multiply-adds on the same row-major arrays.
 * The timed region of every method includes setting C to zero, because ikj
 * and blocked accumulate into C (ijk overwrites it, so for ijk this is only
 * a small O(n^2) cost added to keep the same timing boundary).
 *
 * Usage:
 *   dense <method> <n> [data_dir] [output_file] [--warmup W] [--reps R] [--block BS]
 *
 * Output, one line per run:
 *   method=<m> n=<n> block=<bs> phase=<warmup|measure> rep=<k> time_s=<s> checksum=<sum of C>
 */

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

static double *load_matrix(const char *path, int n) {
    size_t count = (size_t)n * n;
    double *m = malloc(count * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (m == NULL || f == NULL || fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "Error: cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    return m;
}

static void save_matrix(const char *path, const double *m, int n) {
    size_t count = (size_t)n * n;
    FILE *f = fopen(path, "wb");
    if (f == NULL || fwrite(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "Error: cannot write %s\n", path);
        exit(1);
    }
    fclose(f);
}

/* Baseline: inner loop reads B by columns (stride n). */
static void multiply_ijk(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += a[i * n + k] * b[k * n + j];
            c[i * n + j] = s;
        }
}

/* Loop reordering: inner loop reads row k of B and writes row i of C (stride 1). */
static void multiply_ikj(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double aik = a[i * n + k];
            for (int j = 0; j < n; j++)
                c[i * n + j] += aik * b[k * n + j];
        }
}

/* Cache blocking: ikj order inside blocks of bs x bs elements. */
static void multiply_blocked(const double *a, const double *b, double *c, int n, int bs) {
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

static void run_method(const char *method, const double *a, const double *b,
                       double *c, int n, int bs) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    if (strcmp(method, "ijk") == 0)
        multiply_ijk(a, b, c, n);
    else if (strcmp(method, "ikj") == 0)
        multiply_ikj(a, b, c, n);
    else
        multiply_blocked(a, b, c, n, bs);
}

int main(int argc, char **argv) {
    const char *positional[4];
    int n_positional = 0, warmup = 0, reps = 1, bs = 64;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc)
            bs = atoi(argv[++i]);
        else if (n_positional < 4)
            positional[n_positional++] = argv[i];
        else
            n_positional = 5;
    }
    if (n_positional < 2 || n_positional > 4 || bs < 1 ||
        (strcmp(positional[0], "ijk") != 0 && strcmp(positional[0], "ikj") != 0 &&
         strcmp(positional[0], "blocked") != 0)) {
        printf("Usage: %s <ijk|ikj|blocked> <n> [data_dir] [output_file] "
               "[--warmup W] [--reps R] [--block BS]\n", argv[0]);
        return 1;
    }
    const char *method = positional[0];
    int n = atoi(positional[1]);
    const char *data_dir = n_positional >= 3 ? positional[2] : "assignment2/data";
    const char *output_file = n_positional == 4 ? positional[3] : NULL;

    /* Not timed: reading inputs and allocating C */
    char path_a[1024], path_b[1024];
    snprintf(path_a, sizeof path_a, "%s/A_%d.bin", data_dir, n);
    snprintf(path_b, sizeof path_b, "%s/B_%d.bin", data_dir, n);
    double *a = load_matrix(path_a, n);
    double *b = load_matrix(path_b, n);
    double *c = malloc((size_t)n * n * sizeof(double));
    if (c == NULL) {
        fprintf(stderr, "Error: cannot allocate C\n");
        return 1;
    }

    const char *phases[2] = {"warmup", "measure"};
    int counts[2] = {warmup, reps};
    for (int p = 0; p < 2; p++)
        for (int rep = 1; rep <= counts[p]; rep++) {
            /* Timed: zeroing C + multiplication */
            long long start = now_ns();
            run_method(method, a, b, c, n, bs);
            long long elapsed_ns = now_ns() - start;

            double checksum = 0.0;
            for (size_t i = 0; i < (size_t)n * n; i++)
                checksum += c[i];
            printf("method=%s n=%d block=%d phase=%s rep=%d time_s=%.9f checksum=%.10e\n",
                   method, n, bs, phases[p], rep, elapsed_ns / 1e9, checksum);
            fflush(stdout);
        }

    if (output_file != NULL)
        save_matrix(output_file, c, n);
    free(a);
    free(b);
    free(c);
    return 0;
}