/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Test;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationResult;
import jdistlib.rng.MersenneTwister;

public class MultivariateProbabilityTest {
	private static final MultivariateProbabilityOptions ACCURATE =
			new MultivariateProbabilityOptions(2e-6, 2e-6, 262144, 12);

	@Test
	public void normalRectangleProbabilitiesCoverExactSpecialCases() {
		double[] mean = {0.0, 0.0};
		double[][] independent = {{1.0, 0.0}, {0.0, 1.0}};
		MultivariateProbabilityResult independentResult =
				MultivariateNormal.cumulative(new double[] {0.0, 0.0}, mean,
						independent, ACCURATE, new MersenneTwister(1L));
		assertEquals(0.25, independentResult.probability, 2e-8);
		assertTrue(independentResult.absoluteError <= 2e-6);
		assertTrue(independentResult.evaluations > 0);

		double[][] correlated = {{1.0, 0.5}, {0.5, 1.0}};
		MultivariateProbabilityResult correlatedResult =
				MultivariateNormal.cumulative(new double[] {0.0, 0.0}, mean,
						correlated, ACCURATE, new MersenneTwister(2L));
		assertEquals(1.0 / 3.0, correlatedResult.probability, 8e-6);

		MultivariateProbabilityResult scalar = MultivariateNormal.probability(
				new double[] {-1.0}, new double[] {2.0}, new double[] {0.3},
				new double[][] {{2.25}});
		double expected = Normal.cumulative(2.0, 0.3, 1.5, true, false) -
				Normal.cumulative(-1.0, 0.3, 1.5, true, false);
		assertEquals(expected, scalar.probability, 2e-16);
		assertEquals(0, scalar.evaluations);
	}

	@Test
	public void studentTAndCauchyProbabilitiesUseTheCentralEllipticalLaw() {
		double[] location = {0.0, 0.0};
		double[][] scale = {{1.0, 0.5}, {0.5, 1.0}};
		MultivariateProbabilityResult t = MultivariateStudentT.cumulative(
				new double[] {0.0, 0.0}, location, scale, 7.0, ACCURATE,
				new MersenneTwister(3L));
		assertEquals(1.0 / 3.0, t.probability, 2e-5);
		MultivariateProbabilityResult cauchy = MultivariateCauchy.cumulative(
				new double[] {0.0, 0.0}, location, scale, ACCURATE,
				new MersenneTwister(4L));
		assertEquals(1.0 / 3.0, cauchy.probability, 3e-5);

		MultivariateProbabilityResult scalar = MultivariateStudentT.probability(
				new double[] {-2.0}, new double[] {0.7}, new double[] {0.1},
				new double[][] {{1.44}}, 5.0);
		double expected = T.cumulative((0.7 - 0.1) / 1.2, 5.0, true, false) -
				T.cumulative((-2.0 - 0.1) / 1.2, 5.0, true, false);
		assertEquals(expected, scalar.probability, 2e-15);
	}

	@Test
	public void arbitraryBivariateRectanglesMatchIndependentQuadrature() {
		double lowerX = -0.7;
		double upperX = 1.1;
		double lowerY = -1.3;
		double upperY = 0.4;
		double correlation = 0.65;
		double conditionalScale = Math.sqrt(1.0 - correlation * correlation);
		IntegrationResult normalReference = Integrate.integrate(x ->
				Normal.density(x, 0.0, 1.0, false) *
				(Normal.cumulative((upperY - correlation * x) / conditionalScale,
						0.0, 1.0, true, false) -
				 Normal.cumulative((lowerY - correlation * x) / conditionalScale,
						0.0, 1.0, true, false)), lowerX, upperX, 1e-12, 1e-12, 200);
		assertTrue(normalReference.isSuccess());
		MultivariateProbabilityResult normal = MultivariateNormal.probability(
				new double[] {lowerX, lowerY}, new double[] {upperX, upperY},
				new double[] {0.0, 0.0},
				new double[][] {{1.0, correlation}, {correlation, 1.0}}, ACCURATE,
				new MersenneTwister(10L));
		assertEquals(normalReference.result, normal.probability, 1e-5);

		double degreesOfFreedom = 6.0;
		IntegrationResult tReference = Integrate.integrate(x -> {
			double scale = Math.sqrt((degreesOfFreedom + x * x) /
					(degreesOfFreedom + 1.0) *
					(1.0 - correlation * correlation));
			double conditional = T.cumulative((upperY - correlation * x) / scale,
					degreesOfFreedom + 1.0, true, false) -
					T.cumulative((lowerY - correlation * x) / scale,
							degreesOfFreedom + 1.0, true, false);
			return T.density(x, degreesOfFreedom, false) * conditional;
		}, lowerX, upperX, 1e-11, 1e-11, 200);
		assertTrue(tReference.isSuccess());
		MultivariateProbabilityResult t = MultivariateStudentT.probability(
				new double[] {lowerX, lowerY}, new double[] {upperX, upperY},
				new double[] {0.0, 0.0},
				new double[][] {{1.0, correlation}, {correlation, 1.0}},
				degreesOfFreedom, ACCURATE, new MersenneTwister(11L));
		assertEquals(tReference.result, t.probability, 2e-5);
	}

	@Test
	public void logNormalProbabilitiesTransformBounds() {
		double[] meanLog = {0.1, -0.2};
		double[][] covariance = {{1.0, 0.3}, {0.3, 0.8}};
		double[] upper = {Math.exp(0.4), Math.exp(0.7)};
		MultivariateProbabilityResult expected = MultivariateNormal.cumulative(
				new double[] {0.4, 0.7}, meanLog, covariance, ACCURATE,
				new MersenneTwister(5L));
		MultivariateProbabilityResult actual = MultivariateLogNormal.cumulative(
				upper, meanLog, covariance, ACCURATE, new MersenneTwister(5L));
		assertEquals(expected.probability, actual.probability, 0.0);
		assertEquals(0.0, MultivariateLogNormal.cumulative(
				new double[] {0.0, 1.0}, meanLog, covariance).probability, 0.0);
	}

	@Test
	public void namedQuantilesHaveUnambiguousDefinitions() {
		double[] mean = {0.0, 0.0};
		double[][] independent = {{1.0, 0.0}, {0.0, 1.0}};
		double q = MultivariateNormal.equicoordinateQuantile(0.9, mean,
				independent, ACCURATE, new MersenneTwister(6L));
		assertEquals(Normal.quantile(Math.sqrt(0.9), 0.0, 1.0, true, false), q,
				2e-5);
		assertEquals(Math.sqrt(ChiSquare.quantile(0.95, 3.0, true, false)),
				MultivariateNormal.radialQuantile(0.95, 3, true, false), 0.0);
		assertEquals(Math.sqrt(2.0 * F.quantile(0.95, 2.0, 7.0, true, false)),
				MultivariateStudentT.radialQuantile(0.95, 2, 7.0, true, false), 0.0);
		double expectedPower = Math.pow(2.0 * Gamma.quantile(0.95, 1.5, 1.0,
				true, false), 0.5);
		assertEquals(expectedPower, MultivariatePowerExponential.radialQuantile(
				0.95, 3, 1.0, true, false), 0.0);
	}

	@Test
	public void probabilityResultsReportInvalidInputAndEvaluationLimits() {
		MultivariateProbabilityResult invalid = MultivariateNormal.cumulative(
				new double[] {0.0, 0.0}, new double[] {0.0}, new double[][] {{1.0}});
		assertEquals(2, invalid.status);
		assertEquals(MultivariateProbabilityStatus.INVALID_INPUT,
				invalid.getStatus());
		assertFalse(invalid.isSuccess());
		assertFalse(invalid.hasEstimate());
		assertTrue(Double.isNaN(invalid.probability));

		MultivariateProbabilityOptions tiny = new MultivariateProbabilityOptions(
				1e-16, 1e-16, 96, 12);
		MultivariateProbabilityResult limited = MultivariateNormal.cumulative(
				new double[] {0.1, 0.2, 0.3}, new double[] {0.0, 0.0, 0.0},
				new double[][] {{1.0, 0.4, 0.2}, {0.4, 1.0, 0.3},
					{0.2, 0.3, 1.0}}, tiny, new MersenneTwister(7L));
		assertEquals(1, limited.status);
		assertEquals(MultivariateProbabilityStatus.MAX_EVALUATIONS_REACHED,
				limited.getStatus());
		assertFalse(limited.isConverged());
		assertTrue(limited.hasEstimate());
		assertEquals(96, limited.evaluations);
	}

	@Test
	public void explicitRandomStreamsRemainReproducibleUnderConcurrency()
			throws Exception {
		ExecutorService pool = Executors.newFixedThreadPool(4);
		try {
			List<Callable<MultivariateProbabilityResult>> calls = new ArrayList<>();
			for (int i = 0; i < 8; i++) {
				calls.add(() -> MultivariateNormal.cumulative(
						new double[] {0.2, -0.1, 0.7},
						new double[] {0.0, 0.0, 0.0},
						new double[][] {{1.0, 0.4, 0.1}, {0.4, 1.0, 0.3},
							{0.1, 0.3, 1.0}}, ACCURATE,
						new MersenneTwister(12345L)));
			}
			List<Future<MultivariateProbabilityResult>> results =
					pool.invokeAll(calls);
			double expected = results.get(0).get().probability;
			for (Future<MultivariateProbabilityResult> future : results) {
				MultivariateProbabilityResult result = future.get();
				assertEquals(expected, result.probability, 0.0);
				assertTrue(result.absoluteError >= 0.0);
			}
		} finally {
			pool.shutdownNow();
		}
	}

	@Test
	public void wishartDensityReducesToScaledChiSquare() {
		double degreesOfFreedom = 6.5;
		double scale = 4.0;
		double x = 9.0;
		double expected = ChiSquare.density(x / scale, degreesOfFreedom, false) /
				scale;
		assertEquals(expected, Wishart.density(new double[][] {{x}},
				degreesOfFreedom, new double[][] {{scale}}, false), 2e-16);
		assertEquals(Math.log(expected), Wishart.densityFromCholesky(
				new double[][] {{x}}, degreesOfFreedom, new double[][] {{2.0}}, true),
				2e-15);
		assertEquals(0.0, Wishart.density(new double[][] {{0.0}},
				degreesOfFreedom, new double[][] {{scale}}, false), 0.0);
	}

	@Test
	public void wishartSamplerMatchesBartlettReductionAndMoments() {
		MersenneTwister wishartRandom = new MersenneTwister(8L);
		MersenneTwister chiSquareRandom = new MersenneTwister(8L);
		double[][] scalar = Wishart.random(7.0, new double[][] {{2.0}},
				wishartRandom);
		assertNotNull(scalar);
		assertEquals(4.0 * ChiSquare.random(7.0, chiSquareRandom), scalar[0][0],
				1e-14);

		double[][] scale = {{2.0, 0.6}, {0.6, 1.0}};
		MersenneTwister random = new MersenneTwister(9L);
		int count = 30000;
		double[] sums = new double[3];
		for (int i = 0; i < count; i++) {
			double[][] draw = Wishart.randomFromScale(8.0, scale, random);
			sums[0] += draw[0][0];
			sums[1] += draw[0][1];
			sums[2] += draw[1][1];
		}
		assertEquals(16.0, sums[0] / count, 0.15);
		assertEquals(4.8, sums[1] / count, 0.10);
		assertEquals(8.0, sums[2] / count, 0.10);
		assertNull(Wishart.random(0.5, new double[][] {{1.0, 0.0},
				{0.0, 1.0}}, random));
	}
}
