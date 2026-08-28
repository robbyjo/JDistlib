/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.*;

import org.junit.Test;
import jdistlib.finance.*;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;

public class Finance090Test {
	@Test public void tailRiskUsesLossAndReturnConventions() {
		Normal returns = new Normal(0.01, 0.2);
		assertEquals(-returns.quantile(0.01), FinancialRisk.valueAtRisk(returns, 0.99, RiskConvention.RETURN), 1e-12);
		NumericalEstimate es = FinancialRisk.expectedShortfall(new Normal(), 0.975, RiskConvention.LOSS);
		assertTrue(es.isConverged());
		assertEquals(2.3378, es.getValue(), 3e-3);
		assertEquals(0.5, FinancialRisk.shortfallProbability(new Normal(), 0.0), 1e-15);
		assertEquals(0.0, FinancialRisk.expectile(new Normal(), 0.5).getValue(), 3e-3);
	}

	@Test public void analyticTransformsAndTiltReduceCorrectly() {
		Normal normal = new Normal(2.0, 3.0);
		assertEquals(-4.5, normal.logCharacteristic(1.0).real(), 1e-15);
		assertEquals(2.0, DistributionTransforms.cumulant(normal, 1).getValue(), 1e-6);
		DistributionTransforms.TiltResult tilt = DistributionTransforms.esscherTilt(new Normal(), 0.5);
		assertTrue(tilt.getNormalization().isConverged());
		assertEquals(0.5, tilt.getDistribution().quantile(0.5), 2e-4);
		assertFalse(new T(3.0).momentGeneratingDomain().contains(0.1));
	}

	@Test public void heavyTailedFamiliesExposeReductionsAndSeededSampling() {
		StableDistribution stable = new StableDistribution(2.0, 0.7, 1.5, -0.2);
		assertEquals(Normal.density(0.4, -0.2, Math.sqrt(2.0) * 1.5, false), stable.density(0.4, false), 1e-15);
		StableDistribution cauchy = new StableDistribution(1.0, 0.0, 2.0, 1.0);
		assertEquals(Cauchy.cumulative(0.3, 1.0, 2.0, true, false), cauchy.cumulative(0.3, true, false), 1e-15);
		VarianceGammaDistribution first = new VarianceGammaDistribution(2.0, -0.1, 0.3, 0.0);
		VarianceGammaDistribution second = new VarianceGammaDistribution(2.0, -0.1, 0.3, 0.0);
		first.setRandomEngine(new MersenneTwister(91)); second.setRandomEngine(new MersenneTwister(91));
		assertEquals(first.random(), second.random(), 0.0);
		GeneralizedHyperbolicDistribution gh = new NormalInverseGaussianDistribution(2.0, -0.2, 0.5, 0.0);
		assertTrue(Double.isFinite(gh.density(0.0, true)));
		assertTrue(gh.momentGeneratingDomain().contains(0.0));
	}

	@Test public void aggregationAndConditionalLawsAreReproducible() {
		DistributionApproximation first = DistributionAggregation.weightedSum(
				new GenericDistribution[] {new Normal(), new Gamma(2, 1)}, new double[] {1, -0.2}, 500, 7);
		DistributionApproximation second = DistributionAggregation.weightedSum(
				new GenericDistribution[] {new Normal(), new Gamma(2, 1)}, new double[] {1, -0.2}, 500, 7);
		assertArrayEquals(((EmpiricalDistribution) first.getDistribution()).observations(),
				((EmpiricalDistribution) second.getDistribution()).observations(), 0.0);
		ConditionalDistribution conditional = new ConditionalDistribution(new Normal(), -1.0, 1.0);
		assertEquals(0.5, conditional.cumulative(0.0), 1e-15);
		assertEquals(Math.pow(0.5, 3), OrderStatisticDistribution.maximum(new Uniform(0, 1), 3).cumulative(0.5), 1e-15);
	}

	@Test public void asymmetricAndRotatedCopulasReportTailBehavior() {
		JoeCopula joe = new JoeCopula(2.0);
		assertEquals(0.5, joe.cumulative(new double[] {0.5, 1.0}), 1e-12);
		assertTrue(Double.isFinite(joe.logDensity(new double[] {0.4, 0.7})));
		BB1Copula bb1 = new BB1Copula(1.0, 1.5);
		assertTrue(bb1.kendallsTau(0, 1) > 0.0);
		assertEquals(new ClaytonCopula(2, 1.0).cumulative(new double[] {0.3, 0.7}),
				new BB1Copula(1.0, 1.0).cumulative(new double[] {0.3, 0.7}), 1e-14);
		RotatedCopula rotated = new RotatedCopula(new ClaytonCopula(2, 2.0), RotatedCopula.Rotation.SURVIVAL_180);
		assertTrue(CopulaTailAnalysis.upperTailDependence(rotated).getValue() > 0.5);
		assertEquals(0.25, CopulaTailAnalysis.stressProbability(new IndependenceCopula(2), Tail.UPPER, 0.5, Tail.UPPER, 0.5), 1e-15);
	}

	@Test public void impliedVolatilityAndCurveRecoveryAreChecked() {
		double price = ReferenceOptions.blackScholes(100, 105, 0.97, 0.5, 0.24, true);
		ImpliedVolatilityResult implied = ReferenceOptions.impliedBlackScholes(price, 100, 105, 0.97, 0.5, true);
		assertTrue(implied.isConverged()); assertEquals(0.24, implied.getVolatility(), 1e-9);
		assertEquals(ImpliedVolatilityResult.Status.PRICE_ABOVE_BOUND,
				ReferenceOptions.impliedBlackScholes(200, 100, 105, 0.97, 0.5, true).getStatus());
		double normalPrice=ReferenceOptions.bachelier(1.0,100.0,1.0,1.0,250.0,true);
		assertEquals(250.0,ReferenceOptions.impliedBachelier(normalPrice,1.0,100.0,1.0,1.0,true).getVolatility(),1e-8);
		OptionCurve curve = OptionCurve.build(100, 1.0, 1.0,
				new OptionObservation(80, true, 21.2), new OptionObservation(90, true, 13.6),
				new OptionObservation(100, true, 8.0), new OptionObservation(110, true, 4.8),
				new OptionObservation(120, true, 2.9));
		assertTrue(curve.getDiagnostics().isConvex());
		double mass=0.0;for(double value:curve.getDistribution().getAtomMasses())mass+=value;
		assertEquals(1.0,mass,1e-14);
		assertEquals(1.0,curve.getDistribution().cumulative(Double.POSITIVE_INFINITY),0.0);
	}

	@Test public void compoundCountLawsNormalizeAndHandleDegeneracy() {
		DelaporteDistribution delaporte = new DelaporteDistribution(1.2, 2.0, 0.6);
		PolyaAeppliDistribution polya = new PolyaAeppliDistribution(1.2, 0.6);
		double delaporteMass=0.0,polyaMass=0.0;for(int i=0;i<100;i++){delaporteMass+=delaporte.density(i,false);polyaMass+=polya.density(i,false);}
		assertEquals(1.0,delaporteMass,1e-10);assertEquals(1.0,polyaMass,1e-10);
		assertEquals(1.0,new DelaporteDistribution(0.0,2.0,1.0).density(0,false),0.0);
		assertEquals(0.0,new DelaporteDistribution(0.0,2.0,1.0).density(1,false),0.0);
		assertEquals(1.0,new PolyaAeppliDistribution(0.0,1.0).density(0,false),0.0);
	}

	@Test public void evtEstimatorsAndGeneralFitterReturnDiagnostics() {
		double[] tail={1,2,3,4,5,7,9,12,18,30};
		assertTrue(ExtremeValueInference.hill(tail,3).getValue()>0.0);
		assertTrue(Double.isFinite(ExtremeValueInference.pickands(tail,2).getValue()));
		ExtremeValueInference.ThresholdDiagnostics diagnostics=ExtremeValueInference.thresholds(tail,new double[]{3,7});
		assertArrayEquals(new int[]{7,4},diagnostics.getExceedances());
	}
}
