/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.*;

import org.junit.Test;
import jdistlib.finance.*;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.Complex;
import jdistlib.rng.MersenneTwister;

public class FinanceFollowupTest {
	@Test public void temperedStableAndMeixnerTransformsRespectDomains() {
		FourierInversionOptions inversion = FourierInversionOptions.defaults().withPanels(1024)
				.withFrequencyRange(20.0, 160.0).withMaximumRefinements(4).withTolerance(1e-5);
		CgmyDistribution cgmy = new CgmyDistribution(0.4, 5.0, 7.0, 0.7, 0.02, inversion);
		assertEquals(0.0, cgmy.logCharacteristic(0.0).abs(), 0.0);
		assertTrue(cgmy.momentGeneratingDomain().contains(1.0));
		assertFalse(cgmy.momentGeneratingDomain().contains(7.0));
		assertTrue(cgmy.densityResult(0.0).getValue() > 0.0);
		NormalTemperedStableDistribution nts = new NormalTemperedStableDistribution(0.6, 2.0, 0.8, -0.1, 0.3, 0.01, inversion);
		assertEquals(0.0, nts.logMomentGenerating(0.0).abs(), 0.0);
		double ntsProbability = nts.cumulativeResult(0.0).getValue();
		assertTrue(ntsProbability >= 0.0 && ntsProbability <= 1.0);
		MeixnerDistribution meixner = new MeixnerDistribution(0.4, 0.0, 1.5, 0.0, inversion);
		Complex positive = meixner.logCharacteristic(0.8), negative = meixner.logCharacteristic(-0.8);
		assertEquals(positive.real(), negative.real(), 1e-14);
		assertEquals(positive.imaginary(), -negative.imaginary(), 1e-14);
		assertEquals(0.0, meixner.quantile(0.5), 2e-5);
		MeixnerDistribution first = new MeixnerDistribution(0.4, 0.0, 1.5, 0.0, inversion);
		MeixnerDistribution second = new MeixnerDistribution(0.4, 0.0, 1.5, 0.0, inversion);
		first.setRandomEngine(new MersenneTwister(71)); second.setRandomEngine(new MersenneTwister(71));
		assertEquals(first.random(), second.random(), 0.0);
		assertEquals(Math.log(0.047533822460251667) - 700.0,
				GeneralizedHyperbolicDistribution.logBesselK(700.0, 2.25), 5e-14);
	}

	@Test public void adaptiveCosAndSaddlepointPathsMatchNormalReferences() {
		FourierInversionOptions options = FourierInversionOptions.defaults().withTolerance(1e-7).withPanels(2048);
		NumericalEstimate density = DistributionTransforms.densityAdaptive(new Normal(), 0.4, options);
		assertEquals(Normal.density(0.4, 0, 1, false), density.getValue(), 2e-5);
		NumericalEstimate saddlepoint = DistributionTransforms.saddlepointCumulative(new Normal(), 0.7);
		assertEquals(Normal.cumulative(0.7, 0, 1, true, false), saddlepoint.getValue(), 2e-5);
		DistributionApproximation cos = DistributionAggregation.cosInversion(new Normal(), -8, 8, 1024, 128);
		assertEquals(0.5, cos.getDistribution().cumulative(0.0), 3e-3);
		assertTrue(cos.getDiagnostics().getAbsoluteError() < 1e-5);
	}

	@Test public void exactFftAndPanjerAggregationMatchKnownMasses() {
		FiniteGridDistribution coin = new FiniteGridDistribution(0, 1, new double[] {0.7, 0.3});
		FiniteGridDistribution sum = (FiniteGridDistribution) DistributionAggregation
				.exactDiscreteConvolution(coin, coin).getDistribution();
		assertArrayEquals(new double[] {0.49, 0.42, 0.09}, sum.getProbabilities(), 1e-14);
		FiniteGridDistribution fft = (FiniteGridDistribution) DistributionAggregation
				.fftConvolution(new double[] {0.7, 0.3}, 0, new double[] {0.7, 0.3}, 0, 1).getDistribution();
		assertArrayEquals(sum.getProbabilities(), fft.getProbabilities(), 1e-13);
		double lambda = 2.0;
		FiniteGridDistribution poisson = (FiniteGridDistribution) DistributionAggregation
				.panjerCompound(0.0, lambda, Math.exp(-lambda), new double[] {0.0, 1.0}, 30).getDistribution();
		for (int k = 0; k < 10; k++) assertEquals(Poisson.density(k, lambda, false), poisson.density(k, false), 1e-12);
		FiniteGridDistribution binomialWithZeroSeverity = (FiniteGridDistribution) DistributionAggregation
				.panjerCompound(-2.0 / 3.0, 2.0, 0.36, new double[] {0.5, 0.5}, 2).getDistribution();
		assertArrayEquals(new double[] {0.64, 0.32, 0.04}, binomialWithZeroSeverity.getProbabilities(), 1e-13);
	}

	@Test public void multivariateConstructionsReduceToScalarFamilies() {
		double[][] covariance = {{1.0, 0.25}, {0.25, 2.0}};
		MultivariateFinancialDistribution vg = MultivariateFinancialDistribution.varianceGamma(
				1.7, new double[] {0.1, -0.2}, new double[] {0.02, -0.03}, covariance);
		assertTrue(vg.linearCombination(new double[] {0.6, 0.4}) instanceof VarianceGammaDistribution);
		assertArrayEquals(vg.random(91), vg.random(91), 0.0);
		MultivariateFinancialDistribution stable = MultivariateFinancialDistribution.stable(
				1.5, new double[] {0.0, 0.0}, covariance);
		StableDistribution projection = (StableDistribution) stable.linearCombination(new double[] {1.0, 0.0});
		assertEquals(1.0, projection.getScale(), 1e-15);
	}

	@Test public void stableSpecialCasesAndTailsAreAvailable() {
		StableDistribution levy = new StableDistribution(0.5, 1.0, 1.2, 0.3);
		assertEquals(Levy.density(2.0, 0.3, 1.2, false), levy.density(2.0, false), 1e-14);
		assertEquals(Levy.cumulative(2.0, 0.3, 1.2, true, false), levy.cumulative(2.0, true, false), 1e-14);
		assertTrue(levy.upperTailAsymptotic(100.0) > 0.0);
		StableDistribution reflected = new StableDistribution(0.5, -1.0, 1.2, 0.3);
		assertEquals(0.0, reflected.density(0.4, false), 0.0);
	}

	@Test public void smoothingRiskPathAndLevyCompositionExposeDiagnostics() {
		OptionCurve curve = OptionCurve.build(100, 1.0, 1.0,
				new OptionObservation(80, true, 21.2), new OptionObservation(90, true, 13.6),
				new OptionObservation(100, true, 8.0), new OptionObservation(110, true, 4.8),
				new OptionObservation(120, true, 2.9));
		SmoothOptionDistributionResult smooth = curve.smoothDistribution(2.0);
		assertEquals(1.0, smooth.getDistribution().getNormalizationConstant(), 2e-6);
		assertTrue(Double.isFinite(smooth.getMaximumPriceResidual()));
		assertEquals(0.25, AdvancedRiskMeasures.entropic(new Normal(), 0.5, RiskConvention.LOSS).getValue(), 1e-12);
		NumericalEstimate spectral = AdvancedRiskMeasures.spectral(new Normal(), RiskConvention.LOSS, p -> 1.0);
		assertEquals(0.0, spectral.getValue(), 2e-3);
		DistributionApproximation first = PathFunctionalDistributions.maximumDrawdown(new Normal(), 12, 500, 44);
		DistributionApproximation second = PathFunctionalDistributions.maximumDrawdown(new Normal(), 12, 500, 44);
		assertArrayEquals(((EmpiricalDistribution) first.getDistribution()).observations(),
				((EmpiricalDistribution) second.getDistribution()).observations(), 0.0);
		Normal unit = new Normal();
		LevyIncrementDistribution one = new LevyIncrementDistribution(unit, 0.5);
		LevyIncrementDistribution two = new LevyIncrementDistribution(unit, 1.25);
		assertEquals(1.75, one.plus(two).getTime(), 0.0);
		assertEquals(Normal.cumulative(0.5, 0, Math.sqrt(0.5), true, false), one.cumulative(0.5), 2e-5);
	}
}
