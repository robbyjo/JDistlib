package jdistlib;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.disttest.DistributionTest;
import jdistlib.disttest.TestKind;
import jdistlib.math.Bessel;
import jdistlib.rng.MersenneTwister;

public class RCurrentBugzillaRegressionTest {
	@Test
	public void tiedMoodAndAnsariTestsUseTheCorrectExchangeableScores() {
		double[] first = {3, 2, 3};
		double[] second = {4, 1, 5};

		double[] mood = DistributionTest.mood_test(first, second);
		assertEquals(-1.7928429140015905, mood[0], 2e-15);
		assertEquals(0.07299804543011551, mood[1], 2e-15);

		double[] ansari = DistributionTest.ansari_bradley_test(first, second,
				false);
		assertEquals(9.0, ansari[0], 0.0);
		assertEquals(0.0697253691835128, ansari[1], 2e-15);
		double lower = DistributionTest.ansari_bradley_test(first, second,
				false, TestKind.LOWER)[1];
		double greater = DistributionTest.ansari_bradley_test(first, second,
				false, TestKind.GREATER)[1];
		assertEquals(1.0, lower + greater, 2e-16);
	}

	@Test
	public void negativeHalfIntegerBesselOrdersRetainTheirConnectionTerm() {
		double x = 1.5;
		double scale = sqrt(2.0 / (PI * x));
		assertEquals(scale * cos(x), Bessel.j(x, -0.5), 2e-15);
		assertEquals(scale * sin(x), Bessel.y(x, -0.5), 2e-15);
		assertEquals(Double.POSITIVE_INFINITY, Bessel.j(x, -1700.5), 0.0);
	}

	@Test
	public void noncentralTDensityAvoidsCdfCancellation() {
		assertEquals(0.23896620520494657,
				NonCentralT.density(2e-7, 20, 1, false), 3e-15);
		assertEquals(0.23896622940201886,
				NonCentralT.density(3e-7, 20, 1, false), 3e-15);
		assertEquals(0.23896625359909118,
				NonCentralT.density(4e-7, 20, 1, false), 3e-15);
		assertEquals(0.3961263744414466,
				NonCentralT.density(1.081715148, 150, 1, false), 5e-15);
	}

	@Test
	public void flignerTestIsStableUnderAffineRescaling() {
		double[] original = {2, 2, 6, 6, 1, 4, 5, 3, 5, 6, 5, 5};
		double[] rescaled = new double[original.length];
		int[] group = new int[original.length];
		for (int i = 0; i < original.length; i++) {
			rescaled[i] = (original[i] - 1) / 5.0;
			group[i] = i < 6 ? 1 : 2;
		}

		double[] first = DistributionTest.fligner_test(original, group);
		double[] second = DistributionTest.fligner_test(rescaled, group);
		assertEquals(first[0], second[0], 0.0);
		assertEquals(first[1], second[1], 0.0);
		assertEquals(4.279354289480157, first[0], 2e-15);
		assertEquals(0.038578001144839606, first[1], 2e-15);

		// Raw binary64 ranking remains available for compatibility.
		double[] raw = DistributionTest.fligner_test(rescaled, group,
				Double.POSITIVE_INFINITY);
		assertEquals(4.814848002245868, raw[0], 2e-15);
	}

	@Test
	public void fixedProbabilityNegativeBinomialHandlesInfiniteAndHugeSize() {
		double infinity = Double.POSITIVE_INFINITY;
		assertEquals(1.0, NegBinomial.density(0, infinity, 1, false), 0.0);
		assertEquals(0.0, NegBinomial.density(1, infinity, 1, false), 0.0);
		assertEquals(0.0, NegBinomial.cumulative(3, infinity, 0.999,
				true, false), 0.0);
		assertEquals(1.0, NegBinomial.cumulative(3, infinity, 0.999,
				false, false), 0.0);
		assertEquals(Double.POSITIVE_INFINITY,
				NegBinomial.quantile(0.3, infinity, 0.999, true, false), 0.0);
		assertEquals(Double.POSITIVE_INFINITY,
				NegBinomial.random(infinity, 0.999, new MersenneTwister(1)), 0.0);

		double maximum = Double.MAX_VALUE;
		for (double size : new double[] {maximum, maximum / 4, maximum / 8}) {
			for (double x : new double[] {maximum / 16, maximum / 4, maximum}) {
				double logDensity = NegBinomial.density(x, size, 0.999, true);
				double cumulative = NegBinomial.cumulative(x, size, 0.999,
						true, false);
				assertFalse(Double.isNaN(logDensity));
				assertFalse(Double.isNaN(cumulative));
				assertTrue(logDensity <= 0.0);
				assertEquals(1.0, cumulative, 0.0);
			}
		}
	}
}
