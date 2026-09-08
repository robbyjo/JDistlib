import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import jdistlib.*;

/** Paired with compare-r.R; no file I/O, parsing or startup in timed regions. */
public final class DistributionAudit {
    private static volatile double sink;
    private DistributionAudit() { }
    private static final class Workload {
        final String family;
        final char op;
        final boolean lower, log;
        final List<Double> inputs = new ArrayList<>(), expected = new ArrayList<>();
        Workload(String key) {
            String[] p = key.split(":"); family = p[0]; op = p[1].charAt(0);
            lower = p[2].equals("1"); log = p[3].equals("1");
        }
        double value(double x) {
            switch (family) {
            case "normal": return op == 'd' ? Normal.density(x,.3,1.7,log) : op == 'p' ? Normal.cumulative(x,.3,1.7,lower,log) : Normal.quantile(x,.3,1.7,lower,log);
            case "gamma": return op == 'd' ? Gamma.density(x,2.7,1.3,log) : op == 'p' ? Gamma.cumulative(x,2.7,1.3,lower,log) : Gamma.quantile(x,2.7,1.3,lower,log);
            case "beta": return op == 'd' ? Beta.density(x,.7,3.2,log) : op == 'p' ? Beta.cumulative(x,.7,3.2,lower,log) : Beta.quantile(x,.7,3.2,lower,log);
            case "t": return op == 'd' ? T.density(x,5.5,log) : op == 'p' ? T.cumulative(x,5.5,lower,log) : T.quantile(x,5.5,lower,log);
            case "f": return op == 'd' ? F.density(x,4.5,11,log) : op == 'p' ? F.cumulative(x,4.5,11,lower,log) : F.quantile(x,4.5,11,lower,log);
            case "exponential": return op == 'd' ? Exponential.density(x,1.3,log) : op == 'p' ? Exponential.cumulative(x,1.3,lower,log) : Exponential.quantile(x,1.3,lower,log);
            case "weibull": return op == 'd' ? Weibull.density(x,.7,1.3,log) : op == 'p' ? Weibull.cumulative(x,.7,1.3,lower,log) : Weibull.quantile(x,.7,1.3,lower,log);
            case "lognormal": return op == 'd' ? LogNormal.density(x,.3,1.7,log) : op == 'p' ? LogNormal.cumulative(x,.3,1.7,lower,log) : LogNormal.quantile(x,.3,1.7,lower,log);
            case "cauchy": return op == 'd' ? Cauchy.density(x,.3,1.7,log) : op == 'p' ? Cauchy.cumulative(x,.3,1.7,lower,log) : Cauchy.quantile(x,.3,1.7,lower,log);
            case "logistic": return op == 'd' ? Logistic.density(x,.3,1.7,log) : op == 'p' ? Logistic.cumulative(x,.3,1.7,lower,log) : Logistic.quantile(x,.3,1.7,lower,log);
            case "poisson": return op == 'd' ? Poisson.density(x,17.3,log) : op == 'p' ? Poisson.cumulative(x,17.3,lower,log) : Poisson.quantile(x,17.3,lower,log);
            case "binomial": return op == 'd' ? Binomial.density(x,73,.27,log) : op == 'p' ? Binomial.cumulative(x,73,.27,lower,log) : Binomial.quantile(x,73,.27,lower,log);
            case "negbinomial": return op == 'd' ? NegBinomial.density(x,4.5,.27,log) : op == 'p' ? NegBinomial.cumulative(x,4.5,.27,lower,log) : NegBinomial.quantile(x,4.5,.27,lower,log);
            case "ncchisquare": return op == 'd' ? NonCentralChiSquare.density(x,4.5,12,log) : op == 'p' ? NonCentralChiSquare.cumulative(x,4.5,12,lower,log) : NonCentralChiSquare.quantile(x,4.5,12,lower,log);
            case "ncbeta": return op == 'd' ? NonCentralBeta.density(x,2.5,7,11,log) : op == 'p' ? NonCentralBeta.cumulative(x,2.5,7,11,lower,log) : NonCentralBeta.quantile(x,2.5,7,11,lower,log);
            default: throw new IllegalArgumentException(family);
            }
        }
        double run(double[] x, int repetitions) {
            double sum = 0;
            for (int r = 0; r < repetitions; r++) for (double v : x) sum += value(v);
            sink = sum; return sum;
        }
    }
    public static void main(String[] args) throws Exception {
        if (args.length < 2) throw new IllegalArgumentException("reference.tsv output-directory [accuracy-only]");
        Path out = Paths.get(args[1]); Files.createDirectories(out);
        Map<String, Workload> work = new LinkedHashMap<>();
        try (BufferedReader in = Files.newBufferedReader(Paths.get(args[0]), StandardCharsets.UTF_8)) {
            in.readLine(); String line;
            while ((line = in.readLine()) != null) {
                String[] p = line.split("\t"); Workload w = work.get(p[0]);
                if (w == null) { w = new Workload(p[0]); work.put(p[0], w); }
                w.inputs.add(number(p[1])); w.expected.add(number(p[2]));
            }
        }
        int total = 0, differences = 0;
        try (BufferedWriter accuracy = Files.newBufferedWriter(out.resolve("java-accuracy.tsv"));
             BufferedWriter timing = Files.newBufferedWriter(out.resolve("java-timing.tsv"));
             BufferedWriter mismatch = Files.newBufferedWriter(out.resolve("differences.tsv"))) {
            accuracy.write("key\tcount\tdifferences\tmax_scaled_error\n");
            timing.write("key\tns_per_value\trepetitions\tchecksum\n");
            mismatch.write("key\tx\tR\tJava\tscaled_error\n");
            for (Map.Entry<String, Workload> entry : work.entrySet()) {
                Workload w = entry.getValue(); String key = entry.getKey();
                double[] x = new double[w.inputs.size()]; int bad = 0; double max = 0;
                // Near-zero log probabilities need relative accuracy too. This is
                // a discrepancy screen, not permission to copy an erroneous R value.
                double tolerance = key.startsWith("nc") ? 2e-7 : 5e-10;
                for (int i = 0; i < x.length; i++) {
                    x[i] = w.inputs.get(i); double expected = w.expected.get(i), actual = w.value(x[i]);
                    double error = actual == expected || (Double.isNaN(actual) && Double.isNaN(expected)) ? 0 :
                        !Double.isFinite(actual) || !Double.isFinite(expected) ? Double.POSITIVE_INFINITY :
                        Math.abs(actual - expected) / Math.max(1e-300, Math.abs(expected));
                    max = Math.max(max, error);
                    if (error > tolerance) { bad++; mismatch.write(key + "\t" + x[i] + "\t" + expected + "\t" + actual + "\t" + error + "\n"); }
                }
                total += x.length; differences += bad;
                accuracy.write(key + "\t" + x.length + "\t" + bad + "\t" + max + "\n");
                if (args.length < 3 || !args[2].equals("accuracy-only")) {
                    long warm = System.nanoTime(); do { w.run(x, 1); } while (System.nanoTime() - warm < 100_000_000L);
                    int reps = 1;
                    for (;;) { long start = System.nanoTime(); w.run(x, reps);
                        if (System.nanoTime() - start >= 40_000_000L || reps >= 8192) break; reps *= 2; }
                    double[] samples = new double[5];
                    for (int i = 0; i < samples.length; i++) { long start = System.nanoTime(); w.run(x, reps);
                        samples[i] = (System.nanoTime() - start) / ((double) reps * x.length); }
                    Arrays.sort(samples);
                    timing.write(String.format(Locale.ROOT, "%s\t%.9g\t%d\t%.17g%n", key, samples[2], reps, w.run(x, 1)));
                }
                System.out.println(key + " differences=" + bad + " max_scaled_error=" + max);
            }
        }
        System.out.println("Compared " + total + " values; " + differences + " discrepancies require investigation.");
        if (differences != 0) System.exit(1);
    }
    private static double number(String text) {
        if (text.equals("Inf")) return Double.POSITIVE_INFINITY;
        if (text.equals("-Inf")) return Double.NEGATIVE_INFINITY;
        if (text.equals("NA")) return Double.NaN;
        return Double.parseDouble(text);
    }
}
