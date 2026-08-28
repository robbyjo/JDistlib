/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationResult;
import jdistlib.rng.MersenneTwister;

public class WishartProbabilityTest {
	private static final MultivariateProbabilityOptions ACCURATE =
			new MultivariateProbabilityOptions(2e-7, 2e-7, 524288, 16);

	@Test
	public void directionalQuadraticFormReducesExactlyToChiSquare() {
		double[][] scale = {{2.0, 0.3}, {0.3, 1.0}};
		double[] direction = {1.0, -2.0};
		double degreesOfFreedom = 7.5;
		double directionalScale = 4.8;
		double upper = 31.0;
		double expected = ChiSquare.cumulative(upper / directionalScale,
				degreesOfFreedom, true, false);
		assertEquals(expected, Wishart.quadraticFormCumulative(upper, direction,
				degreesOfFreedom, scale, true, false), 3e-16);
		assertEquals(Math.log(expected), Wishart.quadraticFormCumulative(upper,
				direction, degreesOfFreedom, scale, true, true), 5e-16);
		assertTrue(Double.isNaN(Wishart.quadraticFormCumulative(upper,
				new double[] {0.0, 0.0}, degreesOfFreedom, scale, true, false)));
	}

	@Test
	public void standardizedTraceReducesExactlyToChiSquare() {
		double[][] scale = {{2.0, 0.3}, {0.3, 1.0}};
		double degreesOfFreedom = 6.25;
		double upper = 15.0;
		assertEquals(ChiSquare.cumulative(upper, 12.5, true, false),
				Wishart.standardizedTraceCumulative(upper, degreesOfFreedom, scale,
						true, false), 0.0);
		assertEquals(ChiSquare.cumulative(upper, 12.5, false, true),
				Wishart.standardizedTraceCumulative(upper, degreesOfFreedom, scale,
						false, true), 0.0);
	}

	@Test
	public void scalarDeterminantProbabilityIsExactChiSquareReduction() {
		double degreesOfFreedom = 5.5;
		double scale = 3.0;
		double lower = 2.0;
		double upper = 13.0;
		double expected = ChiSquare.cumulative(upper / scale, degreesOfFreedom,
				true, false) - ChiSquare.cumulative(lower / scale, degreesOfFreedom,
				true, false);
		MultivariateProbabilityResult result = Wishart.determinantProbability(lower,
				upper, degreesOfFreedom, new double[][] {{scale}});
		assertEquals(expected, result.probability, 2e-16);
		assertEquals(0, result.evaluations);
		assertTrue(result.isSuccess());
	}

	@Test
	public void bivariateDeterminantMatchesIndependentBartlettQuadrature() {
		double degreesOfFreedom = 7.0;
		double[][] scale = {{2.0, 0.4}, {0.4, 1.5}};
		double scaleDeterminant = 2.84;
		double upper = 95.0;
		double standardizedUpper = upper / scaleDeterminant;
		IntegrationResult reference = Integrate.integrate(x -> {
			if (!(x > 0.0)) return 0.0;
			return ChiSquare.density(x, degreesOfFreedom, false) *
					ChiSquare.cumulative(standardizedUpper / x,
							degreesOfFreedom - 1.0, true, false);
		}, 0.0, Double.POSITIVE_INFINITY, 1e-11, 1e-11, 300);
		assertTrue(reference.isSuccess());
		MultivariateProbabilityResult actual = Wishart.determinantCumulative(upper,
				degreesOfFreedom, scale, ACCURATE, new MersenneTwister(410L));
		assertEquals(reference.result, actual.probability, 8e-7);
		assertTrue(actual.absoluteError <= 2e-7);
	}

	@Test
	public void logDeterminantIntervalsAreScaleInvariantAndTailSafe() {
		double degreesOfFreedom = 8.0;
		double[][] identity = {{1.0, 0.0}, {0.0, 1.0}};
		double[][] scale = {{4.0, 1.0}, {1.0, 3.0}};
		double logScaleDeterminant = Math.log(11.0);
		double lower = 2.0;
		double upper = 4.8;
		MultivariateProbabilityResult standardized = Wishart.logDeterminantProbability(
				lower, upper, degreesOfFreedom, identity, ACCURATE,
				new MersenneTwister(411L));
		MultivariateProbabilityResult transformed = Wishart.logDeterminantProbability(
				lower + logScaleDeterminant, upper + logScaleDeterminant,
				degreesOfFreedom, scale, ACCURATE, new MersenneTwister(411L));
		assertEquals(standardized.probability, transformed.probability, 0.0);
		assertEquals(standardized.absoluteError, transformed.absoluteError, 0.0);

		MultivariateProbabilityResult extreme = Wishart.logDeterminantCumulative(
				-1000.0, degreesOfFreedom, identity,
				new MultivariateProbabilityOptions(1e-15, 1e-12, 4096, 8),
				new MersenneTwister(412L));
		assertEquals(0.0, extreme.probability, 0.0);
		assertFalse(extreme.isSuccess());
		assertTrue(extreme.absoluteError > 0.0);
	}

	@Test
	public void determinantApisRejectAmbiguousOrInvalidInputs() {
		double[][] identity = {{1.0, 0.0}, {0.0, 1.0}};
		assertEquals(0.0, Wishart.determinantCumulative(0.0, 5.0, identity)
				.probability, 0.0);
		assertEquals(1.0, Wishart.logDeterminantProbability(
				Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 5.0, identity)
				.probability, 0.0);
		MultivariateProbabilityResult invalid = Wishart.determinantCumulative(2.0,
				0.5, identity);
		assertEquals(MultivariateProbabilityStatus.INVALID_INPUT,
				invalid.getStatus());
		assertTrue(Double.isNaN(invalid.probability));
	}
}
