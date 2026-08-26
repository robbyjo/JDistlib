import java.util.Arrays;

import jdistlib.disttest.MultipleTesting;
import jdistlib.disttest.MultipleTesting.AdaptiveFdrResult;
import jdistlib.disttest.MultipleTesting.CensoredTestResult;
import jdistlib.disttest.MultipleTesting.Method;

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
        double[] logAdjusted = MultipleTesting.adjustLog(
                new double[] {Math.log(0.01), -1000.0, Math.log(0.2)},
                Method.BENJAMINI_HOCHBERG);
        CensoredTestResult censored = MultipleTesting.testRightCensored(
                new double[] {0.001, 0.004, 0.01, 0.03},
                0.05, 10, 0.05, Method.BENJAMINI_HOCHBERG);

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
        System.out.println("log adjusted: " + Arrays.toString(logAdjusted));
        System.out.println("censored decisions exact: "
                + censored.areDecisionsExact());
        System.out.println(rawCutoff + " " + pi0 + " "
                + Arrays.toString(qValues));
    }
}
