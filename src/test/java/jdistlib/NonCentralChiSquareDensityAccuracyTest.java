package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

/** Independent Bessel-function and squared-normal density references. */
public class NonCentralChiSquareDensityAccuracyTest {
    @Test
    public void densitiesAgreeWithOneHundredDigitBesselReferences() {
        double[][] cases = {
            {10000, 1, 10, -4694.989489882914776219344190324},
            {10000, 0, 10, -4696.717616436992197300528084024},
            {1e-200, 10, 10, -1853.711864128384219381126265956},
            {1000, 1, 10, -410.065963353255686577224549040},
            {17, 4.5, 80, -16.023767798130901732742477507},
            {1, 0, 10, -3.521033594996730006518275129},
            {1e-100, 5, 2000, -1347.405314770979635035874293176},
            {1, 1000, 2000, -3951.190436004798724854334468419}
        };
        for (double[] c : cases) {
            double actual = NonCentralChiSquare.density(c[0], c[1], c[2], true);
            assertEquals(c[3], actual, 4 * Math.ulp(c[3]));
            assertEquals(Math.exp(c[3]), NonCentralChiSquare.density(c[0], c[1], c[2], false),
                Math.exp(c[3]) * 2e-13);
        }
    }

    @Test
    public void oneDegreeOfFreedomIsTheSquareOfAShiftedNormal() {
        for (double ncp : new double[] {0.01, 1, 10, 100, 2000}) {
            for (double x : new double[] {1e-200, 0.01, 1, 10, 1000, 10000}) {
                double a = Math.sqrt(x) - Math.sqrt(ncp);
                double b = Math.sqrt(x) + Math.sqrt(ncp);
                double first = -0.5 * a * a;
                double second = -0.5 * b * b;
                double expected = first + Math.log1p(Math.exp(second - first))
                    - 0.5 * Math.log(2 * Math.PI) - Math.log(2) - 0.5 * Math.log(x);
                assertEquals(expected, NonCentralChiSquare.density(x, 1, ncp, true),
                    2e-12 + 8 * Math.ulp(expected));
            }
        }
    }

    @Test
    public void zeroDegreesAndZeroEndpointsRespectTheAtomAndContinuousPart() {
        assertEquals(Double.POSITIVE_INFINITY, NonCentralChiSquare.density(0, 0, 10, true), 0);
        assertEquals(0, NonCentralChiSquare.density(1, 0, 0, false), 0);
        assertEquals(-5 - Math.log(2), NonCentralChiSquare.density(0, 2, 10, true), 0);
        assertEquals(0, NonCentralChiSquare.density(0, 3, 10, false), 0);
        assertTrue(Double.isNaN(NonCentralChiSquare.density(1, -1, 10, true)));
        assertEquals(Math.log(Double.MIN_VALUE) - 2 * Math.log(2) - 0.5,
            NonCentralChiSquare.density(1, 0, Double.MIN_VALUE, true), 2e-13);
    }

    @Test
    public void upperCdfRetainsTheComponentsBeyondThePoissonMassCutoff() {
        double[][] cases = {
            {1711.55384277344, -718.681839751221211490380541140},
            {1736.00391113281, -729.882167817043486978547271772},
            {1760.45397949219, -741.089714901001829723840917761},
            {1800, -759.231920965539164373904277510}
        };
        for (double[] c : cases) {
            assertEquals(c[1], NonCentralChiSquare.cumulative(c[0], 4.5, 12, false, true), 1e-12);
            double upper = Math.exp(c[1]);
            assertEquals(upper, NonCentralChiSquare.cumulative(c[0], 4.5, 12, false, false),
                Math.max(2 * Double.MIN_VALUE, upper * 1e-12));
            assertEquals(-upper, NonCentralChiSquare.cumulative(c[0], 4.5, 12, true, true),
                Math.max(2 * Double.MIN_VALUE, upper * 1e-12));
        }
    }

    @Test
    public void oddSubnormalNoncentralitiesKeepTheirUnroundedLogPoissonMean() {
        for (double ncp : new double[] {Double.MIN_VALUE, 3 * Double.MIN_VALUE,
                5 * Double.MIN_VALUE, 99 * Double.MIN_VALUE}) {
            double logLambda = Math.log(ncp) - Math.log(2);
            assertEquals(logLambda - Math.log(2) - 0.5,
                NonCentralChiSquare.density(1, 0, ncp, true), 3e-13);
            for (double x : new double[] {0, 1, 100}) {
                // Higher Poisson terms are smaller by O(ncp*x) here.
                assertEquals(logLambda - 0.5 * x,
                    NonCentralChiSquare.cumulative(x, 0, ncp, false, true), 3e-13);
            }
        }
    }

    @Test
    public void noncentralFInfiniteDenominatorUsesTheLogJacobian() {
        for (double df : new double[] {1, 2, 5, 12}) {
            double density = NonCentralF.density(0.7, df, Double.POSITIVE_INFINITY, 3, false);
            assertEquals(Math.log(density),
                NonCentralF.density(0.7, df, Double.POSITIVE_INFINITY, 3, true), 2e-15);
        }
        assertEquals(Double.NEGATIVE_INFINITY,
            NonCentralF.density(Double.POSITIVE_INFINITY, 5, 10, 3, true), 0);
        assertTrue(Double.isNaN(NonCentralF.random(5, 10, Double.NaN, new MersenneTwister(1))));
    }
}
