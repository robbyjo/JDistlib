/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;

public class BacklogDistributionsTest {
	@Test
	public void generalizedFMatchesFlexsurvVectorsAndSpecialCases() {
		double[] expectedDensity = {
			0.353553390593274, 0.140288989252053,
			0.067923038519582, 0.038247711235678
		};
		double[] expectedCdf = {
			0.5, 0.727159434644773, 0.825443507527843, 0.876588815661789
		};
		for (int i = 0; i < expectedDensity.length; i++) {
			double x = i + 1.0;
			assertEquals(expectedDensity[i],
					GeneralizedF.density(x, 0.0, 1.0, 0.0, 1.0, false), 2e-14);
			assertEquals(expectedCdf[i],
					GeneralizedF.cumulative(x, 0.0, 1.0, 0.0, 1.0,
							true, false), 2e-14);
		}
		assertEquals(0.459858613264917,
				GeneralizedF.quantile(0.25, 0.0, 1.0, 0.0, 1.0,
						true, false), 3e-14);
		for (double x : new double[] {0.1, 0.7, 2.0, 20.0}) {
			assertEquals(LogNormal.density(x, 0.4, 1.2, false),
					GeneralizedF.density(x, 0.4, 1.2, 0.0, 0.0, false), 3e-15);
			assertEquals(LogNormal.cumulative(x, 0.4, 1.2, true, false),
					GeneralizedF.cumulative(x, 0.4, 1.2, 0.0, 0.0,
							true, false), 3e-15);
		}
		assertEquals(0.03214437, GeneralizedF.cumulative(80.0, 7.495875,
				0.35362, 0.4572124, 16.68415, true, false), 4e-8);
		assertEquals(0.9933716, GeneralizedF.cumulative(1e5, 7.495875,
				0.35362, 0.4572124, 16.68415, true, false), 1e-7);
	}

	@Test
	public void halfFamiliesReduceToTheirSymmetricParents() {
		for (double x : new double[] {0.0, 0.2, 1.0, 5.0}) {
			assertEquals(2.0 * Cauchy.density(x, 0.0, 1.7, false),
					HalfCauchy.density(x, 1.7, false), 2e-16);
			assertEquals(2.0 * T.density(x / 1.4, 3.5, false) / 1.4,
					HalfT.density(x, 3.5, 1.4, false), 3e-16);
		}
		for (double p : new double[] {1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8}) {
			double hc = HalfCauchy.quantile(p, 1.7, true, false);
			double ht = HalfT.quantile(p, 3.5, 1.4, true, false);
			assertEquals(p, HalfCauchy.cumulative(hc, 1.7, true, false), 3e-15);
			assertEquals(p, HalfT.cumulative(ht, 3.5, 1.4, true, false), 3e-13);
		}
		assertEquals(HalfCauchy.density(2.0, 1.4, false),
				HalfT.density(2.0, 1.0, 1.4, false), 2e-16);
	}

	@Test
	public void slashAndTukeyLambdaHaveCompleteDpqrApis() {
		assertEquals(0.3989422804014327 / (2.0 * 1.7),
				Slash.density(2.0, 2.0, 1.7, false), 2e-16);
		assertEquals(0.5, Slash.cumulative(2.0, 2.0, 1.7, true, false), 0.0);
		for (double p : new double[] {0.001, 0.1, 0.5, 0.9, 0.999}) {
			double value = Slash.quantile(p, -0.3, 1.8, true, false);
			assertEquals(p, Slash.cumulative(value, -0.3, 1.8, true, false), 2e-13);
		}

		for (double x : new double[] {-0.9, -0.2, 0.0, 0.4, 0.9}) {
			assertEquals((x + 1.0) / 2.0,
					TukeyLambda.cumulative(x, 1.0, true, false), 2e-15);
			assertEquals(0.5, TukeyLambda.density(x, 1.0, false), 2e-15);
		}
		assertEquals(0.5, TukeyLambda.density(-1.0, 1.0, false), 0.0);
		assertEquals(0.5, TukeyLambda.density(1.0, 1.0, false), 0.0);
		for (double p : new double[] {0.01, 0.2, 0.5, 0.8, 0.99}) {
			assertEquals(Logistic.quantile(p, 0.0, 1.0, true, false),
					TukeyLambda.quantile(p, 0.0, true, false), 3e-15);
		}
	}

	@Test
	public void discreteBacklogFamiliesMatchDefinitions() {
		double q = 0.7;
		for (int x = 0; x < 12; x++) {
			assertEquals((1.0 - q) * Math.pow(q, x),
					DiscreteWeibull.density(x, q, 1.0, false), 3e-16);
			assertEquals(1.0 - Math.pow(q, x + 1.0),
					DiscreteWeibull.cumulative(x, q, 1.0, true, false), 3e-16);
		}

		double total = 0.0;
		for (int x = 3; x <= 8; x++) {
			total += NegativeHypergeometric.density(x, 5, 7, 3, false);
			assertEquals(total, NegativeHypergeometric.cumulative(x, 5, 7, 3,
					true, false), 3e-15);
		}
		assertEquals(1.0, total, 3e-15);

		double betaNegativeTotal = 0.0;
		for (int x = 0; x < 100000; x++) {
			betaNegativeTotal += BetaNegativeBinomial.density(x, 2.5, 4.0,
					3.0, false);
		}
		assertEquals(1.0, betaNegativeTotal, 2e-13);
		assertEquals(BetaNegativeBinomial.cumulative(8, 2.5, 4.0, 3.0,
				true, false), BetaNegativeBinomial.cumulative(8.9, 2.5, 4.0,
				3.0, true, false), 0.0);
	}

	@Test
	public void skellamCdfQuantileAndDegenerateCasesAreConsistent() {
		assertEquals(0.2070019212239867, Skellam.density(0.0, 2.0, 2.0, false),
				2e-15);
		assertEquals(Poisson.density(4.0, 2.3, false),
				Skellam.density(-4.0, 0.0, 2.3, false), 0.0);
		assertEquals(0.5 + 0.5 * Skellam.density(0.0, 2.0, 2.0, false),
				Skellam.cumulative(0.0, 2.0, 2.0, true, false), 2e-14);
		for (double p : new double[] {0.001, 0.1, 0.5, 0.9, 0.999}) {
			double value = Skellam.quantile(p, 3.2, 1.7, true, false);
			assertTrue(Skellam.cumulative(value, 3.2, 1.7, true, false) >= p);
			assertTrue(Skellam.cumulative(value - 1.0, 3.2, 1.7, true, false) < p);
		}
	}

	@Test
	public void fellerParetoContainsTheUnitLogLogisticLaw() {
		for (double x : new double[] {0.0, 0.2, 1.0, 3.0, 20.0}) {
			assertEquals(1.0 / ((1.0 + x) * (1.0 + x)),
					FellerPareto.density(x, 0.0, 1.0, 1.0, 1.0, 1.0,
							false), 3e-16);
			assertEquals(x / (1.0 + x), FellerPareto.cumulative(x, 0.0,
					1.0, 1.0, 1.0, 1.0, true, false), 3e-16);
		}
		for (double p : new double[] {1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8}) {
			double value = FellerPareto.quantile(p, -2.0, 1.3, 0.8, 2.1,
					4.0, true, false);
			assertEquals(p, FellerPareto.cumulative(value, -2.0, 1.3, 0.8,
					2.1, 4.0, true, false), 3e-13);
		}
	}

	@Test
	public void phaseTypeReducesToErlangAndSupportsAnAtomAtZero() {
		double[] initial = {1.0, 0.0, 0.0};
		double[][] rates = {
			{-2.0, 2.0, 0.0},
			{0.0, -2.0, 2.0},
			{0.0, 0.0, -2.0}
		};
		PhaseType erlang = new PhaseType(initial, rates);
		for (double x : new double[] {0.1, 0.7, 2.0, 8.0}) {
			assertEquals(Gamma.density(x, 3.0, 0.5, false),
					erlang.density(x, false), 2e-13);
			assertEquals(Gamma.cumulative(x, 3.0, 0.5, true, false),
					erlang.cumulative(x, true, false), 2e-13);
		}
		assertEquals(2.0, erlang.quantile(Gamma.cumulative(2.0, 3.0, 0.5,
				true, false), true, false), 2e-12);

		PhaseType atom = new PhaseType(new double[] {0.7},
				new double[][] {{-2.0}});
		assertEquals(0.3, atom.atomProbability(0.0), 6e-17);
		assertEquals(0.3, atom.cumulative(0.0, true, false), 6e-17);
		assertEquals(0.3 + 0.7 * (1.0 - Math.exp(-2.0)),
				atom.cumulative(1.0, true, false), 3e-15);
		assertEquals(0.0, atom.quantile(0.2, true, false), 0.0);
	}

	@Test
	public void taskViewAdditionsAndRepairedBetaPrimeUseTransformIdentities() {
		double discreteTotal = 0.0;
		for (int k = -1000; k <= 1000; k++) {
			discreteTotal += DiscreteLaplace.density(1.5 + k, 1.5, 0.4, false);
		}
		assertEquals(1.0, discreteTotal, 3e-15);
		assertEquals(0.0, DiscreteLaplace.density(1.0, 1.5, 0.4, false), 0.0);
		for (double p : new double[] {1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8}) {
			double value = DiscreteLaplace.quantile(p, 1.5, 0.4, true, false);
			assertTrue(DiscreteLaplace.cumulative(value, 1.5, 0.4,
					true, false) >= p);
			assertTrue(DiscreteLaplace.cumulative(value - 1.0, 1.5, 0.4,
					true, false) < p);
		}

		for (double x : new double[] {0.1, 0.7, 2.0, 20.0}) {
			assertEquals(1.0 / ((1.0 + x) * (1.0 + x)),
					BetaPrime.density(x, 1.0, 1.0, false), 2e-16);
			assertEquals(x / (1.0 + x),
					BetaPrime.cumulative(x, 1.0, 1.0, true, false), 2e-16);
		}
		assertEquals(-2.0 * Math.log1p(1e100),
				BetaPrime.density(1e100, 1.0, 1.0, true), 0.0);
		assertEquals(-Math.log1p(1e100),
				BetaPrime.cumulative(1e100, 1.0, 1.0, false, true), 3e-14);
		for (double p : new double[] {1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8}) {
			double betaPrime = BetaPrime.quantile(p, 1.3, 2.7, true, false);
			assertEquals(p, BetaPrime.cumulative(betaPrime, 1.3, 2.7,
					true, false), 3e-13);
			double logitNormal = LogitNormal.quantile(p, -0.4, 1.7, true, false);
			assertEquals(p, LogitNormal.cumulative(logitNormal, -0.4, 1.7,
					true, false), 3e-13);
		}
	}

	@Test
	public void newRandomGeneratorsRespectSupportAndFirstMoments() {
		MersenneTwister random = new MersenneTwister(20260826L);
		int n = 20000;
		double discreteWeibull = 0.0;
		double skellam = 0.0;
		double phase = 0.0;
		double[][] rates = {{-2.0, 2.0}, {0.0, -2.0}};
		double[] initial = {1.0, 0.0};
		for (int i = 0; i < n; i++) {
			discreteWeibull += DiscreteWeibull.random(0.7, 1.0, random);
			skellam += Skellam.random(3.2, 1.7, random);
			phase += PhaseType.random(initial, rates, random);
		}
		assertEquals(0.7 / 0.3, discreteWeibull / n, 0.06);
		assertEquals(1.5, skellam / n, 0.06);
		assertEquals(1.0, phase / n, 0.025);
	}

	@Test
	public void newContinuousQuantilesInvertLoggedUpperTails() {
		GenericDistribution[] values = {
			new HalfCauchy(1.4), new HalfT(4.0, 1.2), new Slash(0.3, 1.1),
			new TukeyLambda(-0.2), new GeneralizedF(0.2, 0.9, -0.4, 1.3),
			new FellerPareto(-1.0, 1.3, 0.8, 2.2, 1.7),
			new BetaPrime(1.3, 2.7), new LogitNormal(-0.4, 1.7)
		};
		for (GenericDistribution distribution : values) {
			double logP = Math.log(0.2);
			double quantile = distribution.quantile(logP, false, true);
			assertEquals(logP, distribution.cumulative(quantile, false, true),
					3e-10);
		}
	}
}
