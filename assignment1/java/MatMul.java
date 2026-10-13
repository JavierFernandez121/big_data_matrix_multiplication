import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MatMul {

    /** Read n*n little-endian float64 values and return an n x n matrix. */
    static double[][] loadMatrix(Path path, int n) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = buffer.getDouble();
            }
        }
        return m;
    }

    /** Write an n x n matrix as little-endian float64 values (row-major). */
    static void saveMatrix(double[][] m, Path path, int n) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(8 * n * n).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                buffer.putDouble(m[i][j]);
            }
        }
        Files.write(path, buffer.array());
    }

    /** C = A x B using the basic i-j-k triple loop. C must be pre-allocated. */
    static void multiply(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i][k] * b[k][j];
                }
                c[i][j] = s;
            }
        }
    }

    /** Locate assignment1/data relative to where MatMul.class lives. */
    static Path dataDir() throws URISyntaxException {
        Path classDir = Path.of(MatMul.class.getProtectionDomain()
                .getCodeSource().getLocation().toURI());
        return classDir.getParent().resolve("data");
    }

    public static void main(String[] args) throws Exception {
        // Split arguments into positional ones and the --warmup / --reps options
        List<String> positional = new ArrayList<>();
        int warmup = 0;
        int reps = 1;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--warmup")) {
                warmup = Integer.parseInt(args[++i]);
            } else if (args[i].equals("--reps")) {
                reps = Integer.parseInt(args[++i]);
            } else {
                positional.add(args[i]);
            }
        }
        if (positional.isEmpty() || positional.size() > 3) {
            System.out.println("Usage: java -cp assignment1/java MatMul <n> [data_dir] "
                    + "[output_file] [--warmup W] [--reps R]");
            System.exit(1);
        }
        int n = Integer.parseInt(positional.get(0));
        Path data = positional.size() >= 2 ? Path.of(positional.get(1)) : dataDir();
        Path outputFile = positional.size() == 3 ? Path.of(positional.get(2)) : null;

        // Not timed: reading inputs and allocating the result
        double[][] a = loadMatrix(data.resolve("A_" + n + ".bin"), n);
        double[][] b = loadMatrix(data.resolve("B_" + n + ".bin"), n);
        double[][] c = new double[n][n];

        String[] phases = {"warmup", "measure"};
        int[] counts = {warmup, reps};
        for (int p = 0; p < phases.length; p++) {
            for (int rep = 1; rep <= counts[p]; rep++) {
                // Timed: only the multiplication kernel
                long start = System.nanoTime();
                multiply(a, b, c, n);
                long elapsedNs = System.nanoTime() - start;

                double checksum = 0.0;
                for (double[] row : c) {
                    for (double v : row) {
                        checksum += v;
                    }
                }
                // Locale.ROOT: always use '.' as decimal separator (e.g. on Spanish Windows)
                System.out.printf(Locale.ROOT, "n=%d phase=%s rep=%d time_s=%.9f checksum=%.10e%n",
                        n, phases[p], rep, elapsedNs / 1e9, checksum);
                System.out.flush();
            }
        }

        if (outputFile != null) {
            saveMatrix(c, outputFile, n);
        }
    }
}