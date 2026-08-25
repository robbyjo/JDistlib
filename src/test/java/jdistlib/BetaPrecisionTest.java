package jdistlib;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class BetaPrecisionTest {

	@Test
	public void toms708KeepsExtremeFiniteShapeLogsFinite() {
		double left = Beta.cumulative(0.4, 1e308, 1e308, true, true);
		double mirrored = Beta.cumulative(0.6, 1e308, 1e308, false, true);

		assertTrue(left < 0.0);
		assertFalse("left=" + left, Double.isInfinite(left));
		assertFalse("left=" + left + ", mirrored=" + mirrored, Double.isNaN(left));
		assertEquals(left, mirrored, 8.0 * Math.ulp(left));
		assertEquals(0.5, Beta.cumulative(0.5, 1e308, 1e308, true, false), 2e-15);
	}

	@Test
	public void toms708RetainsReferenceLogTailPrecision() {
		double actual = Beta.cumulative(0.9833, 43779, 0.06728, true, true);
		assertEquals(-746.0986886924, actual, abs(actual) * 1e-12);

		double[] x = {.01, .10, .25, .40, .55, .71, .98};
		double[] expected = {
			-0.04605755624088, -0.3182809860569, -0.7503593555585,
			-1.241555830932, -1.851527837938, -2.76044482378,
			-8.149862739881
		};
		for (int i = 0; i < x.length; i++)
			assertEquals(expected[i],
				Beta.cumulative(x[i], 0.8, 2, false, true),
				abs(expected[i]) * 2e-12);
	}

	@Test
	public void quantileRecoversMpfrReferencePoints() {
		double[] exponents = {-3, -15, -100, -300, -600, -1000};
		double[] logProbabilities = {
			-40.7588797271766572448, -248.063200048177428608,
			-1721.00081201679567511, -5186.73671481652222237,
			-10385.3405690161120427, -17316.8123746155651368
		};

		for (int i = 0; i < exponents.length; i++) {
			double expected = pow(2, exponents[i]);
			double actual = Beta.quantile(logProbabilities[i], 25, 6, true, true);
			/* TOMS 708 is certified at roughly 14 significant digits in its
			 * least favorable parameter ratios. */
			assertEquals(expected, actual, abs(expected) * 5e-14);
		}
	}

	@Test
	public void quantileIsAccurateForDifficultSkewAndSmallShapes() {
		double x = Beta.quantile(0.6948886, 0.0672788, 226390, true, false);
		assertTrue(x < 2e-8);
		assertEquals(0.6948886,
			Beta.cumulative(x, 0.0672788, 226390, true, false), 3e-15);

		double a = 0.125;
		for (int i = 4; i <= 160; i += 13) {
			double b = pow(2, -i);
			double probability = b / 4.;
			double quantile = Beta.quantile(probability, a, b, true, false);
			double roundTrip = Beta.cumulative(quantile, a, b, true, false);
			assertEquals(probability, roundTrip, probability * 8e-15);
		}

		double nearOneLogP = -exp(901. / 256.);
		double quantile = Beta.quantile(nearOneLogP, 43779, 0.06728, true, true);
		double roundTrip = Beta.cumulative(quantile, 43779, 0.06728, true, true);
		assertEquals(nearOneLogP, roundTrip, abs(nearOneLogP) * 2e-15);
	}
}
