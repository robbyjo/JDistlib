/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.math.IntegrationResult;
import jdistlib.rng.MersenneTwister;

public class NumericalDistributionTest {
	@Test
	public void continuousKernelIsNormalizedOnFiniteSupport() {
		NumericalContinuousDistribution distribution =
				new NumericalContinuousDistribution(x -> x, 0.0, 1.0,
						1e-12, 1e-12, 200);

		assertEquals(0.5, distribution.getNormalizationConstant(), 2e-15);
		assertEquals(Math.log(0.5), distribution.getLogNormalizationConstant(),
				2e-15);
		assertEquals(0.5, distribution.density(0.25, false), 2e-15);
		assertEquals(Math.log(0.5), distribution.density(0.25, true), 2e-15);
		assertEquals(0.0, distribution.density(-1.0, false), 0.0);
		assertEquals(Double.NEGATIVE_INFINITY,
				distribution.density(2.0, true), 0.0);

		for (double x : new double[] {0.01, 0.2, 0.5, 0.9, 0.99}) {
			assertEquals(x * x, distribution.cumulative(x, true, false), 3e-14);
			assertEquals(1.0 - x * x,
					distribution.cumulative(x, false, false), 3e-14);
			assertEquals(Math.log1p(-x * x),
					distribution.cumulative(x, false, true), 3e-14);
		}
		for (double p : new double[] {1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6}) {
			assertEquals(Math.sqrt(p), distribution.quantile(p, true, false),
					3e-12);
			assertEquals(Math.sqrt(1.0 - p),
					distribution.quantile(p, false, false), 3e-12);
		}
	}

	@Test
	public void continuousKernelSupportsInfiniteIntervalsAndDiagnostics() {
		NumericalContinuousDistribution distribution =
				new NumericalContinuousDistribution(
						x -> Math.exp(-0.5 * x * x),
						Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
						1e-11, 1e-11, 300);

		assertEquals(Math.sqrt(2.0 * Math.PI),
				distribution.getNormalizationConstant(), 2e-11);
		for (double x : new double[] {-3.0, -1.0, 0.0, 1.0, 3.0}) {
			assertEquals(Normal.cumulative(x, 0.0, 1.0, true, false),
					distribution.cumulative(x, true, false), 2e-10);
		}
		for (double p : new double[] {0.01, 0.25, 0.5, 0.9, 0.99}) {
			assertEquals(Normal.quantile(p, 0.0, 1.0, true, false),
					distribution.quantile(p, true, false), 2e-9);
		}

		IntegrationResult diagnostics = distribution.getNormalizationResult();
		assertTrue(diagnostics.isSuccess());
		assertTrue(diagnostics.neval > 0);
		diagnostics.result = -1.0;
		assertTrue(distribution.getNormalizationConstant() > 0.0);
	}

	@Test
	public void continuousRandomUsesConfiguredEngine() {
		NumericalContinuousDistribution distribution =
				new NumericalContinuousDistribution(x -> 1.0, -2.0, 3.0,
						1e-12, 1e-12, 100);
		MersenneTwister expected = new MersenneTwister(42L);
		distribution.setRandomEngine(new MersenneTwister(42L));
		assertEquals(-2.0 + 5.0 * expected.nextDouble(), distribution.random(),
				2e-12);
	}

	@Test
	public void continuousConstructionRejectsInvalidKernelsAndSupports() {
		assertThrows(IllegalArgumentException.class,
				() -> new NumericalContinuousDistribution(x -> 1.0, 1.0, 1.0));
		assertThrows(IllegalArgumentException.class,
				() -> new NumericalContinuousDistribution(x -> -1.0, 0.0, 1.0));
		assertThrows(IllegalArgumentException.class,
				() -> new NumericalContinuousDistribution(x -> 0.0, 0.0, 1.0));
		assertThrows(IllegalArgumentException.class,
				() -> new NumericalContinuousDistribution(x -> Double.NaN,
						0.0, 1.0));
	}

	@Test
	public void discreteFormulaIsNormalizedOverArbitraryOutcomes() {
		NumericalDiscreteDistribution distribution =
				new NumericalDiscreteDistribution(x -> x * x,
						new double[] {3.0, -1.0, 2.0});

		assertArrayEquals(new double[] {-1.0, 2.0, 3.0},
				distribution.getSupport(), 0.0);
		assertArrayEquals(new double[] {1.0 / 14.0, 4.0 / 14.0, 9.0 / 14.0},
				distribution.getProbabilities(), 2e-16);
		assertEquals(14.0, distribution.getNormalizationConstant(), 0.0);
		assertEquals(Math.log(14.0), distribution.getLogNormalizationConstant(),
				2e-15);
		assertEquals(4.0 / 14.0, distribution.density(2.0, false), 2e-16);
		assertEquals(0.0, distribution.density(1.0, false), 0.0);
		assertEquals(5.0 / 14.0,
				distribution.cumulative(2.0, true, false), 2e-16);
		assertEquals(9.0 / 14.0,
				distribution.cumulative(2.0, false, false), 2e-16);
		assertEquals(-1.0, distribution.quantile(0.05, true, false), 0.0);
		assertEquals(2.0, distribution.quantile(0.3, true, false), 0.0);
		assertEquals(3.0, distribution.quantile(0.6, false, false), 0.0);
		assertEquals(3.0, distribution.quantile(Math.log(0.6), false, true), 0.0);
	}

	@Test
	public void discreteIntegerRangeAndScaledSummationAreStable() {
		NumericalDiscreteDistribution integers =
				new NumericalDiscreteDistribution(x -> x + 1.0, 0, 3);
		assertArrayEquals(new double[] {0.0, 1.0, 2.0, 3.0},
				integers.getSupport(), 0.0);
		assertEquals(10.0, integers.getNormalizationConstant(), 0.0);
		assertEquals(2.0, integers.quantile(0.5, true, false), 0.0);

		NumericalDiscreteDistribution huge =
				new NumericalDiscreteDistribution(x -> Double.MAX_VALUE,
						new double[] {0.0, 1.0});
		assertTrue(Double.isInfinite(huge.getNormalizationConstant()));
		assertTrue(Double.isFinite(huge.getLogNormalizationConstant()));
		assertEquals(0.5, huge.density(0.0, false), 0.0);
		assertEquals(0.5, huge.density(1.0, false), 0.0);
	}

	@Test
	public void discreteRandomAndValidationBehavePredictably() {
		NumericalDiscreteDistribution distribution =
				new NumericalDiscreteDistribution(x -> x == 10.0 ? 1.0 : 3.0,
						new double[] {10.0, 20.0});
		distribution.setRandomEngine(new MersenneTwister(8675309L));
		for (int i = 0; i < 100; i++) {
			double draw = distribution.random();
			assertTrue(draw == 10.0 || draw == 20.0);
		}

		assertThrows(IllegalArgumentException.class,
				() -> new NumericalDiscreteDistribution(x -> 1.0,
						new double[] {1.0, 1.0}));
		assertThrows(IllegalArgumentException.class,
				() -> new NumericalDiscreteDistribution(x -> x,
						new double[] {-1.0, 1.0}));
		assertThrows(IllegalArgumentException.class,
				() -> new NumericalDiscreteDistribution(x -> 0.0,
						new double[] {1.0, 2.0}));
		assertThrows(IllegalArgumentException.class,
				() -> new NumericalDiscreteDistribution(x -> 1.0, 2, 1));
		assertFalse(Double.isNaN(distribution.cumulative(15.0, true, false)));
	}
}
