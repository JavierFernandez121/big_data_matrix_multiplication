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

/* Read n*n float64 values (little-endian, as on x86) into a new array. */
static double *load_matrix(const char *path, int n) {
    size_t count = (size_t)n * n;
    double *m = malloc(count * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (m == NULL || f == NULL) {
        fprintf(stderr, "Error: cannot open or allocate %s\n", path);
        exit(1);
    }
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "Error: %s is too short\n", path);
        exit(1);
    }
    fclose(f);
    return m;
}

/* C = A x B using the basic i-j-k triple loop. C must be pre-allocated. */
static void multiply(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/* Build "<directory of the executable>/../data/<name>". */
static void data_path(char *out, size_t size, const char *argv0, const char *name) {
    const char *slash = strrchr(argv0, '/');
    const char *backslash = strrchr(argv0, '\\');
    if (backslash > slash) slash = backslash;
    int dir_len = slash ? (int)(slash - argv0) : 0;
    if (dir_len > 0)
        snprintf(out, size, "%.*s/../data/%s", dir_len, argv0, name);
    else
        snprintf(out, size, "../data/%s", name);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);

    /* Not timed: reading inputs and allocating the result */
    char path_a[1024], path_b[1024], name[64];
    snprintf(name, sizeof name, "A_%d.bin", n);
    data_path(path_a, sizeof path_a, argv[0], name);
    snprintf(name, sizeof name, "B_%d.bin", n);
    data_path(path_b, sizeof path_b, argv[0], name);

    double *a = load_matrix(path_a, n);
    double *b = load_matrix(path_b, n);
    double *c = calloc((size_t)n * n, sizeof(double));
    if (c == NULL) {
        fprintf(stderr, "Error: cannot allocate C\n");
        return 1;
    }

    /* Timed: only the multiplication kernel */
    long long start = now_ns();
    multiply(a, b, c, n);
    long long elapsed_ns = now_ns() - start;

    double checksum = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        checksum += c[i];
    }
    printf("n=%d time_s=%.6f checksum=%.10e\n", n, elapsed_ns / 1e9, checksum);

    free(a);
    free(b);
    free(c);
    return 0;
}