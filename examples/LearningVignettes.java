import jdistlib.DiagnosticPreset;
import jdistlib.DistributionAnalysis;
import jdistlib.LogNormal;
import jdistlib.NumericalContinuousDistribution;
import jdistlib.ProbabilityInterval;
import jdistlib.disttest.DistributionTest;
import jdistlib.rng.MersenneTwister;

/** Compilable counterparts to the beginner tutorials and vignettes. */
public final class LearningVignettes {
    private LearningVignettes() {}

    public static void responseTimeVignette() {
        LogNormal responseTime = new LogNormal(1.6, 0.45);
        double withinEight = responseTime.cumulative(8.0, true, false);
        double overTen = responseTime.cumulative(10.0, false, false);
        double p95 = responseTime.quantile(0.95, true, false);
        double[] observed = {3.8, 4.2, 4.7, 5.1, 5.6,
                6.0, 6.8, 7.4, 8.5, 9.7};
        double[] fit = DistributionTest.anderson_darling_test(
                observed, responseTime, 99, new MersenneTwister(11L));
        System.out.println(withinEight + " " + overTen + " " + p95
                + " " + fit[1]);
    }

    public static void customDistributionVignette() {
        NumericalContinuousDistribution error =
                NumericalContinuousDistribution.builder()
                        .kernel(x -> Math.exp(-Math.abs(x) / 0.55))
                        .support(-2.0, 2.0)
                        .singularities(0.0)
                        .diagnosticPreset(DiagnosticPreset.THOROUGH)
                        .build();
        double withinTolerance = error.cumulative(0.5, true, false)
                - error.cumulative(-0.5, true, false);
        ProbabilityInterval central = error.probabilityInterval(0.95);
        DistributionAnalysis report = error.analyzeDistribution();
        error.setRandomEngine(new MersenneTwister(20260826L));
        double[] draws = error.random(10);
        System.out.println(withinTolerance + " " + central.getLower()
                + " " + report.hasErrors() + " " + draws[0]);
    }

    public static void main(String[] arguments) {
        responseTimeVignette();
        customDistributionVignette();
    }
}
