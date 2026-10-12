import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
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
        if (args.length != 1) {
            System.out.println("Usage: java -cp assignment1/java MatMul <n>");
            System.exit(1);
        }
        int n = Integer.parseInt(args[0]);

        // Not timed: reading inputs and allocating the result
        Path data = dataDir();
        double[][] a = loadMatrix(data.resolve("A_" + n + ".bin"), n);
        double[][] b = loadMatrix(data.resolve("B_" + n + ".bin"), n);
        double[][] c = new double[n][n];

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
        System.out.printf(Locale.ROOT, "n=%d time_s=%.6f checksum=%.10e%n",
                n, elapsedNs / 1e9, checksum);
    }
}