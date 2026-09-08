import java.lang.management.ManagementFactory;
import java.util.Arrays;
import jdistlib.Beta;
import jdistlib.Gamma;
import jdistlib.Poisson;
import jdistlib.math.MathFunctions;

/** Standalone warmed scalar benchmark; compile against either checkout's classes. */
public final class CoefficientBenchmark {
    private static volatile double sink;
    private static double value(int kind, double x) {
        switch (kind) {
        case 0: return MathFunctions.lgammacor(10.0 + x);
        case 1: return MathFunctions.gammafn(0.125 + x * 0.09);
        case 2: return MathFunctions.lgammafn(0.125 + x * 0.09);
        case 3: return MathFunctions.stirlerr(0.5 + x);
        case 4: return Gamma.density(0.125 + x * 0.09, 2.75, 1.5, false);
        case 5: return Poisson.density(Math.floor(x), 12.75, false);
        default: return Beta.density(0.01 + x * 0.009, 2.75, 3.125, false);
        }
    }
    private static double run(int kind, int count) {
        double sum = 0.0;
        for (int i = 0; i < count; ++i) sum += value(kind, (i & 4095) * (100.0 / 4096));
        sink = sum;
        return sum;
    }
    public static void main(String[] args) {
        int count = args.length == 0 ? 1000000 : Integer.parseInt(args[0]);
        String[] names = {"lgammacor", "gamma", "lgamma", "stirlerr", "dgamma", "dpois", "dbeta"};
        com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        bean.setThreadAllocatedMemoryEnabled(true);
        long id = Thread.currentThread().getId();
        System.out.println("function,median_ns_per_call,median_bytes_per_call,checksum,fingerprint");
        for (int kind = 0; kind < names.length; ++kind) {
            for (int warm = 0; warm < 4; ++warm) run(kind, count);
            double[] times = new double[7], bytes = new double[7];
            double sum = 0;
            for (int repeat = 0; repeat < times.length; ++repeat) {
                long allocation = bean.getThreadAllocatedBytes(id), start = System.nanoTime();
                sum = run(kind, count);
                times[repeat] = (System.nanoTime() - start) / (double) count;
                bytes[repeat] = (bean.getThreadAllocatedBytes(id) - allocation) / (double) count;
            }
            long hash = 1;
            for (int i = 0; i < 65536; ++i)
                hash = 31 * hash + Double.doubleToLongBits(value(kind, i * (100.0 / 65536)));
            Arrays.sort(times);
            Arrays.sort(bytes);
            System.out.println(names[kind] + "," + times[3] + "," + bytes[3] + "," + Double.toHexString(sum) + "," + Long.toHexString(hash));
        }
    }
}
