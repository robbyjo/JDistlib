package jdistlib;

import static java.lang.Math.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Independent analytic identities: several of these also expose R 4.6.1 bugs. */
public class ContinuousExtremeScaleTest {
	private static void relative(double expected, double actual) {
		assertTrue("expected a positive finite reference", expected > 0 && Double.isFinite(expected));
		assertEquals(expected, actual, expected * 3e-13);
	}

	@Test
	public void exponentialSupportRespectsBothTails() {
		for (double x : new double[] {Double.NEGATIVE_INFINITY, -1, -0., 0.}) {
			assertEquals(0, Exponential.cumulative(x, 2, true, false), 0);
			assertEquals(1, Exponential.cumulative(x, 2, false, false), 0);
			assertEquals(Double.NEGATIVE_INFINITY, Exponential.cumulative(x, 2, true, true), 0);
			assertEquals(0, Exponential.cumulative(x, 2, false, true), 0);
		}
		assertEquals(log(Double.MIN_VALUE) - log(2),
				Exponential.cumulative(Double.MIN_VALUE, 2, true, true), 1e-13);
		double scale = 1e-300, x = 800 * scale;
		relative(exp(-x / scale - log(scale)), Exponential.density(x, scale, false));
	}

	@Test
	public void gammaScalingPreservesPositiveSubnormalInputs() {
		// Gamma(1/2,2): density exp(-x/2)/sqrt(2*pi*x),
		// CDF erf(sqrt(x/2)) = sqrt(2*x/pi) * (1 + O(x)).
		double x = Double.MIN_VALUE;
		double logDensity = -.5 * log(x) - .5 * log(2 * PI);
		double logCdf = .5 * log(x) + .5 * log(2 / PI);
		assertEquals(logDensity, Gamma.density(x, .5, 2, true), 1e-13);
		relative(exp(logDensity), Gamma.density(x, .5, 2, false));
		assertEquals(logCdf, Gamma.cumulative(x, .5, 2, true, true), 1e-13);
		relative(exp(logCdf), Gamma.cumulative(x, .5, 2, true, false));
		assertEquals(log1p(-exp(logCdf)), Gamma.cumulative(x, .5, 2, false, true), 1e-174);
		// Gamma(shape,1) ~ shape/x for shape and x both tiny.
		relative(1, Gamma.density(1e-320, 1e-320, 1, false));
		assertEquals(-log(2), Gamma.density(x, 1, 2, true), 1e-13);
		// Gamma(1,scale) is exponential, including rescued density tails.
		double scale = 1e-300, tailX = 800 * scale;
		relative(exp(-tailX / scale - log(scale)), Gamma.density(tailX, 1, scale, false));
		// At x=scale, f(x)=exp(-1)/(scale*Gamma(shape));
		// Gamma(shape) = 1/shape * (1 + O(shape)) for this tiny shape.
		assertEquals(-1 - log(1e100) + log(1e-300),
				Gamma.density(1e100, 1e-300, 1e100, true), 2e-13);
	}

	@Test
	public void weibullUsesTheHazardWithoutIntermediateUnderflow() {
		// Preserve the point-mass limit at scale as shape tends to infinity.
		assertEquals(Double.POSITIVE_INFINITY, Weibull.density(2, Double.POSITIVE_INFINITY, 2, true), 0);
		assertEquals(Double.POSITIVE_INFINITY, Weibull.density(2, Double.POSITIVE_INFINITY, 2, false), 0);
		assertEquals(log(5) + 4 * log(1e-200), Weibull.density(1e-200, 5, 1, true), 3e-13);
		assertEquals(5 * log(1e-200), Weibull.cumulative(1e-200, 5, 1, true, true), 3e-13);
		assertEquals(Double.NEGATIVE_INFINITY, Weibull.density(1e200, 5, 1, true), 0);
		assertEquals(0, Weibull.density(1e200, 5, 1, false), 0);
		// A square root rescues an otherwise underflowed x/scale.
		double logHazard = .5 * (log(Double.MIN_VALUE) - log(2));
		relative(exp(logHazard), Weibull.cumulative(Double.MIN_VALUE, .5, 2, true, false));
		relative(exp(-500), Weibull.quantile(-1000, 2, 1, true, true));
		relative(exp(-372), Weibull.quantile(-744, 2, 1, true, true));
		assertEquals(2 * log(1e-160), Weibull.cumulative(1e-160, 2, 1, true, true), 2e-13);
		relative(exp(log(1e300) - 1000), Weibull.quantile(-1000, 1, 1e300, true, true));
		// Scale cancels an overflowing unscaled quantile.
		relative(exp(log(1e-300) + 1000), Weibull.quantile(-exp(100), .1, 1e-300, false, true));
	}

	@Test
	public void lognormalDensitySeparatesTheJacobianFactors() {
		// At x=exp(meanlog), density is exactly 1/(sqrt(2*pi)*x*sdlog).
		assertEquals(-.5 * log(2 * PI) - log(1e300) - log(1e100),
				LogNormal.density(1e300, log(1e300), 1e100, true), 2e-13);
		assertEquals(-.5 * log(2 * PI) - log(1e-300) - log(1e-100),
				LogNormal.density(1e-300, log(1e-300), 1e-100, true), 2e-13);
		// Normal exponential underflows, but division by x rescues the density.
		double x = exp(-500), mean = log(x) - 40;
		double logDensity = -.5 * log(2 * PI) - 800 - log(x);
		relative(exp(logDensity), LogNormal.density(x, mean, 1, false));
	}

	@Test
	public void cauchyDensityAvoidsSquaringOverflow() {
		assertEquals(-log(PI) - 400 * log(10), Cauchy.density(1e200, 0, 1, true), 2e-13);
		relative((1 / PI) / 1e308, Cauchy.density(0, 0, 1e308, false));
		// Standardized distance is exactly 2, although x-location overflows.
		assertEquals(-log(PI) - log(1e308) - log(5),
				Cauchy.density(1e308, -1e308, 1e308, true), 3e-13);
	}

	@Test
	public void logisticDensityRetainsTheScaleAndTail() {
		assertEquals(-log(4) - log(1e308), Logistic.density(0, 0, 1e308, true), 2e-13);
		relative(.25 / 1e308, Logistic.density(0, 0, 1e308, false));
		double scale = 1e-300, x = 800 * scale;
		relative(exp(-x / scale - log(scale)), Logistic.density(x, 0, scale, false));
	}
}
