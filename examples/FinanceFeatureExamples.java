/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.*;
import jdistlib.finance.*;
import jdistlib.generic.GenericDistribution;

/** Small executable examples for the probability-first finance APIs in 0.9.0. */
public final class FinanceFeatureExamples {
	private FinanceFeatureExamples() {}

	public static void main(String[] args) {
		tailRiskAndPayoffs();
		transformsAndHeavyTails();
		aggregationAndExtremes();
		dependenceAndEvt();
	}

	static void tailRiskAndPayoffs() {
		GenericDistribution dailyReturn = new Normal(0.0004, 0.018);
		double lossVar = FinancialRisk.valueAtRisk(dailyReturn, 0.99, RiskConvention.RETURN);
		NumericalEstimate lossEs = FinancialRisk.expectedShortfall(dailyReturn, 0.99, RiskConvention.RETURN);
		NumericalEstimate downside = FinancialRisk.downsideDeviation(dailyReturn, 0.0);
		System.out.printf("99%% return-loss VaR %.5f, ES %.5f, downside deviation %.5f%n",
				lossVar, lossEs.valueOrThrow(), downside.valueOrThrow());

		GenericDistribution terminalPrice = new LogNormal(Math.log(100.0) - 0.5 * 0.2 * 0.2, 0.2);
		System.out.printf("E[(S-110)+] %.4f, E[(90-S)+] %.4f%n",
				FinancialRisk.callPayoff(terminalPrice, 110.0).valueOrThrow(),
				FinancialRisk.putPayoff(terminalPrice, 90.0).valueOrThrow());
	}

	static void transformsAndHeavyTails() {
		Normal physical = new Normal(-0.02, 0.25);
		DistributionTransforms.TiltResult riskNeutral = DistributionTransforms.esscherTilt(physical, 0.32);
		System.out.printf("tilted median %.4f, normalization error %.3g%n",
				riskNeutral.getDistribution().quantile(0.5),
				riskNeutral.getNormalization().getAbsoluteError());

		StableDistribution stable = StableDistribution.fromS0(1.7, -0.25, 0.03, 0.001);
		VarianceGammaDistribution varianceGamma = new VarianceGammaDistribution(1.5, -0.01, 0.025, 0.0);
		NormalInverseGaussianDistribution nig = new NormalInverseGaussianDistribution(25.0, -4.0, 0.04, 0.0);
		System.out.printf("stable CF %s, VG variance cumulant %.6g, NIG log density %.5f%n",
				stable.logCharacteristic(2.0), DistributionTransforms.cumulant(varianceGamma, 2).getValue(),
				nig.density(0.01, true));
	}

	static void aggregationAndExtremes() {
		DistributionApproximation portfolio = DistributionAggregation.weightedSum(
				new GenericDistribution[] {new Normal(0.001, 0.02), new T(5.0)},
				new double[] {0.7, 0.3 * 0.02}, 20_000, 20260901L);
		System.out.printf("portfolio 1%% quantile %.5f (%s, MC error %.4g)%n",
				portfolio.getDistribution().quantile(0.01),
				portfolio.getDiagnostics().getStrategy(), portfolio.getDiagnostics().getAbsoluteError());

		OrderStatisticDistribution worstOfTen = OrderStatisticDistribution.minimum(new Normal(), 10);
		System.out.printf("median worst-of-ten shock %.4f%n", worstOfTen.quantile(0.5));
		System.out.printf("Delaporte P(N<=3) %.4f, Polya-Aeppli P(N=0) %.4f%n",
				new DelaporteDistribution(0.8, 2.0, 0.65).cumulative(3),
				new PolyaAeppliDistribution(0.8, 0.65).density(0, false));
	}

	static void dependenceAndEvt() {
		Copula survivalClayton = new RotatedCopula(new ClaytonCopula(2, 2.0),
				RotatedCopula.Rotation.SURVIVAL_180);
		System.out.printf("upper tail dependence %.4f; joint upper 95%% stress %.4f%n",
				CopulaTailAnalysis.upperTailDependence(survivalClayton).getValue(),
				CopulaTailAnalysis.stressProbability(survivalClayton, Tail.UPPER, 0.95,
						Tail.UPPER, 0.95));
		double[] losses = {1.1,1.4,1.6,1.8,2.0,2.4,2.9,3.7,5.1,8.2,13.0};
		System.out.printf("Hill tail index %.4f, Pickands %.4f%n",
				ExtremeValueInference.hill(losses, 4).getValue(),
				ExtremeValueInference.pickands(losses, 2).getValue());
	}
}
