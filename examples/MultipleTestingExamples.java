import java.util.Arrays;

import jdistlib.disttest.MultipleTesting;
import jdistlib.disttest.MultipleTesting.AdaptiveFdrResult;
import jdistlib.disttest.MultipleTesting.CensoredTestResult;
import jdistlib.disttest.MultipleTesting.Method;
import jdistlib.disttest.MultipleTesting.StepDownFdrResult;
import jdistlib.disttest.DiscreteFdr;
import jdistlib.disttest.DiscretePValueDistribution;
import jdistlib.disttest.online.LordPlusPlus;
import jdistlib.disttest.online.OnlineFdr;
import jdistlib.disttest.online.OnlineFdrDecision;
import jdistlib.disttest.online.Saffron;

/** Minimal multiple-testing and q-value examples compiled by the check task. */
public final class MultipleTestingExamples {
    private MultipleTestingExamples() {}

    public static void main(String[] arguments) {
        double[] pValues = {0.01, 0.04, 0.03, 0.002, 0.50, Double.NaN};

        double[] bh = MultipleTesting.adjust(
                pValues, Method.BENJAMINI_HOCHBERG);
        boolean[] discoveries = MultipleTesting.reject(
                pValues, 0.05, Method.BENJAMINI_HOCHBERG);
        double rawCutoff = MultipleTesting.threshold(
                pValues, 0.05, Method.BENJAMINI_HOCHBERG);

        AdaptiveFdrResult bky = MultipleTesting
                .benjaminiKriegerYekutieli(pValues, 0.05);
        double[] weightedBh = MultipleTesting
                .adjustWeightedBenjaminiHochberg(pValues,
                        new double[] {2.0, 1.0, 0.5, 0.5, 1.0, 1.0});
        double[] weightedHolm = MultipleTesting.adjustWeightedHolm(
                pValues, new double[] {2.0, 1.0, 0.5, 0.5, 1.0, 1.0});
        StepDownFdrResult gbs = MultipleTesting
                .gavrilovBenjaminiSarkar(pValues, 0.05);
        double[] logAdjusted = MultipleTesting.adjustLog(
                new double[] {Math.log(0.01), -1000.0, Math.log(0.2)},
                Method.BENJAMINI_HOCHBERG);
        CensoredTestResult censored = MultipleTesting.testRightCensored(
                new double[] {0.001, 0.004, 0.01, 0.03},
                0.05, 10, 0.05, Method.BENJAMINI_HOCHBERG);

        MultipleTesting.GroupedFdrResult grouped = MultipleTesting
                .selectiveGroupedBenjaminiHochberg(
                        new double[] {0.001, 0.02, 0.01, 0.9},
                        new int[] {1, 1, 2, 2}, 0.05, 0.05);

        double[] gamma = OnlineFdr.polynomialGamma(10_000, 1.6);
        LordPlusPlus lord = new LordPlusPlus(0.05, 0.025, gamma);
        OnlineFdrDecision onlineDecision = lord.test(0.001);
        Saffron saffron = new Saffron(0.05, 0.01, 0.5, gamma);
        saffron.test(0.001);

        DiscretePValueDistribution[] discreteNulls = {
            DiscretePValueDistribution.exact(
                    new double[] {0.01, 0.05, 0.20, 1.0}),
            DiscretePValueDistribution.exact(
                    new double[] {0.02, 0.10, 0.50, 1.0})
        };
        DiscreteFdr.Result discrete = DiscreteFdr.dbhStepDown(
                new double[] {0.02, 0.05}, discreteNulls, 0.05);

        double[] calibrationPValues = new double[200];
        for (int i = 0; i < calibrationPValues.length; i++) {
            calibrationPValues[i] = (i + 0.5) / calibrationPValues.length;
        }
        double pi0 = MultipleTesting.estimateNullProportion(
                calibrationPValues);
        double[] qValues = MultipleTesting.qValues(pValues, pi0);

        System.out.println(Arrays.toString(bh));
        System.out.println(Arrays.toString(discoveries));
        System.out.println("BKY discoveries: " + bky.getRejectedCount());
        System.out.println("weighted BH: " + Arrays.toString(weightedBh));
        System.out.println("weighted Holm: " + Arrays.toString(weightedHolm));
        System.out.println("GBS discoveries: " + gbs.getRejectedCount());
        System.out.println("log adjusted: " + Arrays.toString(logAdjusted));
        System.out.println("censored decisions exact: "
                + censored.areDecisionsExact());
        System.out.println("selected groups: " + grouped.getSelectedGroupCount());
        System.out.println("first online level: " + onlineDecision.getTestLevel());
        System.out.println("discrete DBH discoveries: "
                + discrete.getRejectedCount());
        System.out.println(rawCutoff + " " + pi0 + " "
                + Arrays.toString(qValues));
    }
}
