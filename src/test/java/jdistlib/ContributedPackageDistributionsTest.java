/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.evd.Rayleigh;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;

public class ContributedPackageDistributionsTest {
	@Test
	public void maxwellBoltzmannScaleMatchesMaxwellRateParameterization() {
		double sigma = 2.5;
		double rate = 1.0 / (sigma * sigma);
		double x = 3.0;
		assertEquals(Maxwell.density(x, rate, false),
				MaxwellBoltzmann.density(x, sigma, false), 2e-16);
		assertEquals(Maxwell.density(x, rate, true),
				MaxwellBoltzmann.density(x, sigma, true), 2e-15);
		assertEquals(Maxwell.cumulative(x, rate, true, false),
				MaxwellBoltzmann.cumulative(x, sigma, true, false), 2e-16);
		assertEquals(Maxwell.cumulative(x, rate, false, true),
				MaxwellBoltzmann.cumulative(x, sigma, false, true), 2e-15);

		for (double probability : new double[] {1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8}) {
			double quantile = MaxwellBoltzmann.quantile(probability, sigma,
					true, false);
			assertEquals(probability, MaxwellBoltzmann.cumulative(quantile, sigma,
					true, false), 3e-14);
		}

		MersenneTwister scaleRandom = new MersenneTwister(42L);
		MersenneTwister rateRandom = new MersenneTwister(42L);
		assertEquals(Maxwell.random(rate, rateRandom),
				MaxwellBoltzmann.random(sigma, scaleRandom), 2e-15);
		assertEquals(0.0, MaxwellBoltzmann.density(0.0, sigma, false), 0.0);
		assertTrue(Double.isNaN(MaxwellBoltzmann.density(1.0, 0.0, false)));
	}

	@Test
	public void distributions3ModifiedPoissonFamiliesMatchDefinitions() {
		double lambda = 2.5;
		double pi = 0.25;
		double p0 = Math.exp(-lambda);
		double p3 = p0 * Math.pow(lambda, 3.0) / 6.0;

		assertEquals(pi + (1.0 - pi) * p0,
				ZeroInflatedPoisson.density(0.0, lambda, pi, false), 2e-16);
		assertEquals((1.0 - pi) * p3,
				ZeroInflatedPoisson.density(3.0, lambda, pi, false), 2e-16);
		assertEquals(p3 / (1.0 - p0),
				ZeroTruncatedPoisson.density(3.0, lambda, false), 2e-16);
		assertEquals(1.0 - pi,
				HurdlePoisson.density(0.0, lambda, pi, false), 0.0);
		assertEquals(pi * p3 / (1.0 - p0),
				HurdlePoisson.density(3.0, lambda, pi, false), 2e-16);

		assertEquals(0.0, ZeroTruncatedPoisson.cumulative(0.0, lambda,
				true, false), 0.0);
		assertEquals(1.0, ZeroTruncatedPoisson.quantile(0.0, lambda,
				true, false), 0.0);
		assertEquals(0.0, HurdlePoisson.quantile(0.7, lambda, pi,
				true, false), 0.0);
		assertEquals(3.0, HurdlePoisson.quantile(0.9, lambda, pi,
				true, false), 0.0);
	}

	@Test
	public void distributions3ModifiedNegativeBinomialFamiliesMatchDefinitions() {
		// size 1 and mean 2.5 is geometric with success probability 1/3.5.
		double mu = 2.5;
		double size = 1.0;
		double pi = 0.3;
		double success = 1.0 / 3.5;
		double p0 = success;
		double p2 = success * Math.pow(1.0 - success, 2.0);
		assertEquals(pi + (1.0 - pi) * p0,
				ZeroInflatedNegativeBinomial.density(0.0, mu, size, pi, false),
				2e-16);
		assertEquals(p2 / (1.0 - p0),
				ZeroTruncatedNegativeBinomial.density(2.0, mu, size, false),
				2e-16);
		assertEquals(1.0 - pi,
				HurdleNegativeBinomial.density(0.0, mu, size, pi, false), 0.0);
		assertEquals(pi * p2 / (1.0 - p0),
				HurdleNegativeBinomial.density(2.0, mu, size, pi, false), 2e-16);

		assertEquals(1.0, ZeroTruncatedNegativeBinomial.density(1.0, 0.0,
				size, false), 0.0);
		assertEquals(0.7, HurdleNegativeBinomial.density(0.0, 0.0, size,
				pi, false), 0.0);
		assertEquals(0.3, HurdleNegativeBinomial.density(1.0, 0.0, size,
				pi, false), 0.0);
	}

	@Test
	public void finiteAndEmpiricalFamiliesAreExact() {
		double[] probabilities = {0.2, 0.5, 0.7};
		assertEquals(0.12, PoissonBinomial.density(0.0, probabilities, false),
				2e-16);
		assertEquals(0.43, PoissonBinomial.density(1.0, probabilities, false),
				2e-16);
		assertEquals(0.38, PoissonBinomial.density(2.0, probabilities, false),
				2e-16);
		assertEquals(0.07, PoissonBinomial.density(3.0, probabilities, false),
				2e-16);
		assertEquals(2.0, PoissonBinomial.quantile(0.8, probabilities, true,
				false), 0.0);

		double[] outcomes = {20.0, 10.0, 30.0};
		double[] weights = {0.5, 0.4, 0.1};
		assertEquals(0.5, Categorical.density(20.0, outcomes, weights, false), 0.0);
		assertEquals(0.9, Categorical.cumulative(20.0, outcomes, weights, true,
				false), 2e-16);
		assertEquals(20.0, Categorical.quantile(0.8, outcomes, weights, true,
				false), 0.0);

		assertEquals(0.135, Multinomial.density(new double[] {1, 2, 2}, 5,
				new double[] {0.2, 0.3, 0.5}, false), 2e-15);

		Empirical empirical = new Empirical(new double[] {3, 1, 3, 7});
		assertEquals(0.5, empirical.density(3.0, false), 0.0);
		assertEquals(0.75, empirical.cumulative(3.0, true, false), 0.0);
		assertEquals(3.0, empirical.quantile(0.5, true, false), 0.0);
	}

	@Test
	public void sinhArcsinhContainsNormalAndInvertsSkewedCases() {
		double[] x = {-4.0, -1.25, 0.0, 2.75, 6.0};
		for (double value : x) {
			assertEquals(Normal.density(value, 1.3, 2.1, false),
					SinhArcsinh.density(value, 1.3, 2.1, 1.0, 1.0, false), 2e-15);
			assertEquals(Normal.cumulative(value, 1.3, 2.1, true, false),
					SinhArcsinh.cumulative(value, 1.3, 2.1, 1.0, 1.0, true, false),
					2e-15);
		}
		for (double probability : new double[] {1e-8, 0.01, 0.5, 0.9, 1.0 - 1e-8}) {
			double value = SinhArcsinh.quantile(probability, 3.0, 2.0, 0.7,
					1.3, true, false);
			assertEquals(probability, SinhArcsinh.cumulative(value, 3.0, 2.0,
					0.7, 1.3, true, false), 3e-14);
		}
	}

	@Test
	public void vgamApplicationFamiliesReduceToKnownDistributions() {
		for (double x : new double[] {0.1, 0.7, 2.0, 8.0}) {
			assertEquals(Gamma.density(x, 2.3, 1.7, false),
					GeneralizedGamma.density(x, 1.7, 1.0, 2.3, false), 3e-15);
			assertEquals(Gamma.cumulative(x, 2.3, 1.7, true, false),
					GeneralizedGamma.cumulative(x, 1.7, 1.0, 2.3, true, false),
					3e-15);

			assertEquals(LogLogistic.density(x, 1.8, 2.2, false),
					GeneralizedBetaSecondKind.density(x, 2.2, 1.8, 1.0, 1.0,
							false), 3e-15);
			assertEquals(LogLogistic.cumulative(x, 1.8, 2.2, true, false),
					GeneralizedBetaSecondKind.cumulative(x, 2.2, 1.8, 1.0, 1.0,
							true, false), 3e-15);

			assertEquals(HalfNormal.density(x, 1.4, false),
					FoldedNormal.density(x, 0.0, 1.4, 1.0, 1.0, false), 3e-15);
			assertEquals(HalfNormal.cumulative(x, 1.4, true, false),
					PositiveNormal.cumulative(x, 0.0, 1.4, true, false), 3e-15);

			assertEquals(Rayleigh.density(x, 1.3, false),
					Rice.density(x, 1.3, 0.0, false), 3e-15);
			assertEquals(Rayleigh.cumulative(x, 1.3, true),
					Rice.cumulative(x, 1.3, 0.0, true, false), 3e-13);
		}

		assertEquals(Gompertz.cumulative(2.0, 0.4, 0.7, true, false),
				Makeham.cumulative(2.0, 0.4, 0.7, 0.0, true, false), 2e-15);
		assertEquals(0.2103043927580732,
				Makeham.density(2.0, 0.3, 0.5, 0.1, false), 3e-16);
		assertEquals(0.20800399335336728,
				Makeham.cumulative(2.0, 0.3, 0.5, 0.1, false, false), 3e-16);
		assertEquals(0.21323384529068323, Lindley.density(2.0, 0.7, false),
				3e-16);
		assertEquals(0.44967681659940006,
				Lindley.cumulative(2.0, 0.7, false, false), 3e-16);

		double pigP0 = Math.exp((1.0 - Math.sqrt(1.0 + 2.0 * 0.8 * 1.3 * 1.3))
				/ (0.8 * 1.3));
		assertEquals(pigP0, PoissonInverseGaussian.density(0.0, 1.3, 0.8,
				false), 2e-16);
		assertEquals(pigP0 * 1.3 / Math.sqrt(1.0 + 2.0 * 0.8 * 1.3 * 1.3),
				PoissonInverseGaussian.density(1.0, 1.3, 0.8, false), 2e-16);
		assertEquals(Math.exp(-Math.sqrt(2.0 / 0.8)),
				PoissonInverseGaussian.density(0.0, Double.POSITIVE_INFINITY,
						0.8, false), 2e-16);
		assertEquals(1.0, PoissonInverseGaussian.density(0.0, 1.3,
				Double.POSITIVE_INFINITY, false), 0.0);
	}

	@Test
	public void allNewScalarQuantilesInvertLoggedUpperTails() {
		GenericDistribution[] distributions = {
			new ZeroInflatedPoisson(3.2, 0.2),
			new ZeroTruncatedPoisson(3.2),
			new HurdlePoisson(3.2, 0.8),
			new ZeroInflatedNegativeBinomial(3.2, 1.7, 0.2),
			new ZeroTruncatedNegativeBinomial(3.2, 1.7),
			new HurdleNegativeBinomial(3.2, 1.7, 0.8),
			new GeneralizedGamma(1.4, 0.8, 2.1),
			new GeneralizedBetaSecondKind(1.4, 1.8, 2.1, 3.2),
			new Makeham(0.3, 0.5, 0.1),
			new Lindley(0.7),
			new FoldedNormal(0.8, 1.2, 1.1, 0.9),
			new PositiveNormal(-0.4, 1.3),
			new Rice(1.2, 0.7),
			new Maxwell(0.9),
			new PoissonInverseGaussian(1.3, 0.8),
			new SinhArcsinh(0.3, 1.1, 0.8, 1.2)
		};
		for (GenericDistribution distribution : distributions) {
			double logProbability = Math.log(0.2);
			double value = distribution.quantile(logProbability, false, true);
			double actual = distribution.cumulative(value, false, true);
			if (Math.rint(value) == value && distribution instanceof ZeroInflatedPoisson) {
				assertTrue(actual <= logProbability);
			} else if (Math.rint(value) == value
					&& (distribution instanceof ZeroTruncatedPoisson
							|| distribution instanceof HurdlePoisson
							|| distribution instanceof ZeroInflatedNegativeBinomial
							|| distribution instanceof ZeroTruncatedNegativeBinomial
							|| distribution instanceof HurdleNegativeBinomial
							|| distribution instanceof PoissonInverseGaussian)) {
				assertTrue(actual <= logProbability);
			} else {
				assertEquals(logProbability, actual, 2e-10);
			}
		}
	}

	@Test
	public void continuousFamiliesHaveStableInfiniteEndpoints() {
		GenericDistribution[] distributions = {
			new GeneralizedGamma(1.4, 0.8, 2.1),
			new GeneralizedBetaSecondKind(1.4, 1.8, 2.1, 3.2),
			new Makeham(0.3, 0.5, 0.0),
			new Lindley(0.7),
			new FoldedNormal(0.8, 1.2, 1.1, 0.9),
			new PositiveNormal(-0.4, 1.3),
			new Rice(1.2, 0.7),
			new Maxwell(0.9),
			new SinhArcsinh(0.3, 1.1, 0.8, 1.2)
		};
		for (GenericDistribution distribution : distributions) {
			assertEquals(0.0, distribution.density(Double.POSITIVE_INFINITY,
					false), 0.0);
			assertEquals(1.0, distribution.cumulative(Double.POSITIVE_INFINITY,
					true, false), 0.0);
			assertEquals(Double.NEGATIVE_INFINITY,
					distribution.cumulative(Double.POSITIVE_INFINITY, false, true),
					0.0);
		}
	}

	@Test
	public void randomGeneratorsRespectSupportAndMoments() {
		MersenneTwister random = new MersenneTwister(20260824L);
		int n = 20_000;
		double poissonBinomialSum = 0.0;
		double maxwellSquaredSum = 0.0;
		int hurdleZeros = 0;
		double pigSum = 0.0;
		for (int i = 0; i < n; i++) {
			poissonBinomialSum += PoissonBinomial.random(
					new double[] {0.2, 0.5, 0.7}, random);
			double value = Maxwell.random(0.9, random);
			maxwellSquaredSum += value * value;
			if (HurdlePoisson.random(2.5, 0.7, random) == 0.0) hurdleZeros++;
			pigSum += PoissonInverseGaussian.random(1.3, 0.8, random);
		}
		assertEquals(1.4, poissonBinomialSum / n, 0.025);
		assertEquals(3.0 / 0.9, maxwellSquaredSum / n, 0.08);
		assertEquals(0.3, hurdleZeros / (double) n, 0.015);
		assertEquals(1.3, pigSum / n, 0.04);

		int[] draw = Multinomial.random(17, new double[] {2, 3, 5}, random);
		assertArrayEquals(new int[] {17}, new int[] {draw[0] + draw[1] + draw[2]});
	}
}
