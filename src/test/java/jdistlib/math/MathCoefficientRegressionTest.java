package jdistlib.math;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

/** Identities independent of R protect the shared gamma approximation tables. */
public class MathCoefficientRegressionTest {
    @Test
    public void gammaMatchesHalfIntegerValuesAndRecurrence() {
        double expected = Math.sqrt(Math.PI);
        for (int i = 0; i < 60; ++i) {
            double x = i + 0.5;
            assertEquals("gamma(" + x + ")", expected,
                    MathFunctions.gammafn(x), Math.abs(expected) * 4e-14);
            assertEquals(Math.log(expected), MathFunctions.lgammafn(x),
                    Math.max(1.0, Math.abs(Math.log(expected))) * 4e-15);
            expected *= x;
        }
        for (double x : new double[] {0.01, 0.125, 0.3, 0.875, 2.3, 8.75, 10.25, 37.125}) {
            double value = MathFunctions.gammafn(x);
            assertEquals(x * value, MathFunctions.gammafn(x + 1), Math.abs(x * value) * 4e-14);
        }
    }

    @Test
    public void gammaReflectionChecksNegativeArguments() {
        for (double x : new double[] {0.125, 0.3, 0.7, 1.25, 2.625, 8.75, 12.375}) {
            double expected = Math.PI / MathFunctions.sinpi(x);
            assertEquals(expected, MathFunctions.gammafn(x) * MathFunctions.gammafn(1 - x),
                    Math.abs(expected) * 4e-14);
        }
    }

    @Test
    public void stirlingAndLogGammaCorrectionsMatchTheirDefinitions() {
        for (double x : new double[] {0.125, 0.5, 1, 2.75, 5.125, 6.05, 6.4, 6.9, 8.5,
                10.125, 12.5, 13.25, 23.25, 26.5, 29.5, 100.5}) {
            double expected = MathFunctions.lgammafn(x)
                    - ((x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI));
            assertEquals("stirlerr(" + x + ")", expected, MathFunctions.stirlerr(x), 1e-13);
            if (x >= 10) assertEquals(expected, MathFunctions.lgammacor(x), 1e-13);
        }
        for (double x : new double[] {-0.49, -0.25, -0.01, 0, 0.01, 0.25, 0.49})
            assertEquals(MathFunctions.lgammafn(1 + x), MathFunctions.lgamma1p(x), 8e-16);
    }
}
