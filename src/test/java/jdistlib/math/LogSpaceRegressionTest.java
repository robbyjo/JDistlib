package jdistlib.math;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class LogSpaceRegressionTest {
    private static final double NEG_INF = Double.NEGATIVE_INFINITY;
    private static final double POS_INF = Double.POSITIVE_INFINITY;

    @Test
    public void additionHandlesZeroAndInfiniteTerms() {
        assertEquals(NEG_INF, MathFunctions.logspace_add(NEG_INF, NEG_INF), 0);
        assertEquals(POS_INF, MathFunctions.logspace_add(POS_INF, POS_INF), 0);
        assertEquals(POS_INF, MathFunctions.logspace_add(POS_INF, NEG_INF), 0);
        assertEquals(-1000, MathFunctions.logspace_add(-1000, NEG_INF), 0);
        assertEquals(-1000 + Math.log(2), MathFunctions.logspace_add(-1000, -1000), 0);
    }

    @Test
    public void sumsHandleZeroAndInfiniteTermsWithoutMaskingNaN() {
        assertEquals(NEG_INF, MathFunctions.logspace_sum(new double[0]), 0);
        assertEquals(NEG_INF, MathFunctions.logspace_sum(new double[] {NEG_INF, NEG_INF, NEG_INF}), 0);
        assertEquals(POS_INF, MathFunctions.logspace_sum(new double[] {NEG_INF, 3, POS_INF}), 0);
        assertEquals(-1000 + Math.log(3), MathFunctions.logspace_sum(new double[] {-1000, -1000, -1000}), 0);
        for (double infinity : new double[] {NEG_INF, POS_INF}) {
            assertTrue(Double.isNaN(MathFunctions.logspace_add(infinity, Double.NaN)));
            assertTrue(Double.isNaN(MathFunctions.logspace_add(Double.NaN, infinity)));
            for (int index = 0; index < 3; ++index) {
                double[] values = {infinity, infinity, infinity};
                values[index] = Double.NaN;
                assertTrue(Double.isNaN(MathFunctions.logspace_sum(values)));
            }
        }
    }

    @Test
    public void subtractingTwoZeroTermsHasZeroMass() {
        // exp(-Inf) - exp(-Inf) is exactly 0 - 0, whose logarithm is -Inf.
        assertEquals(NEG_INF, MathFunctions.logspace_sub(NEG_INF, NEG_INF), 0);
        assertEquals(NEG_INF, MathFunctions.logspace_sub(-1000, -1000), 0);
        assertEquals(-1000, MathFunctions.logspace_sub(-1000, NEG_INF), 0);
        assertTrue(Double.isNaN(MathFunctions.logspace_sub(POS_INF, POS_INF)));
        assertTrue(Double.isNaN(MathFunctions.logspace_sub(NEG_INF, -1000)));
        assertTrue(Double.isNaN(MathFunctions.logspace_sub(NEG_INF, Double.NaN)));
    }
}
