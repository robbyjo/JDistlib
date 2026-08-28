/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.*;
import jdistlib.finance.*;
import jdistlib.generic.GenericDistribution;

/** Executable examples for the advanced finance-distribution follow-ups. */
public final class AdvancedFinanceExamples {
	private AdvancedFinanceExamples() {}

	public static void main(String[] args) {
		temperedAndMultivariateReturns();
		checkedAggregation();
		smoothOptionLawAndRisk();
		pathAndLevyLaws();
	}

	static void temperedAndMultivariateReturns() {
		CgmyDistribution cgmy = new CgmyDistribution(0.4, 5.0, 7.0, 0.7, 0.001);
		MeixnerDistribution meixner = new MeixnerDistribution(0.03, -0.2, 1.5, 0.0);
		NormalTemperedStableDistribution nts =
				new NormalTemperedStableDistribution(0.6, 2.0, 0.8, -0.01, 0.03, 0.0);
		System.out.printf("CGMY log CF %s; Meixner MGF domain (%.3f, %.3f); NTS CDF error %.3g%n",
				cgmy.logCharacteristic(2.0), meixner.momentGeneratingDomain().getLower(),
				meixner.momentGeneratingDomain().getUpper(), nts.cumulativeResult(0.0).getAbsoluteError());

		MultivariateFinancialDistribution joint = MultivariateFinancialDistribution.varianceGamma(
				1.8, new double[] {0.001, -0.0003}, new double[] {-0.01, 0.005},
				new double[][] {{0.0004, 0.00012}, {0.00012, 0.0009}});
		GenericDistribution portfolio = joint.linearCombination(new double[] {0.65, 0.35});
		double[] draw = joint.random(20260828);
		System.out.printf("multivariate-VG portfolio 1%% quantile %.5f; seeded draw [%g, %g]%n",
				portfolio.quantile(0.01), draw[0], draw[1]);
	}

	static void checkedAggregation() {
		FiniteGridDistribution defaultLoss = new FiniteGridDistribution(0.0, 1.0,
				new double[] {0.96, 0.03, 0.01});
		DistributionApproximation exact = DistributionAggregation.exactDiscreteConvolution(
				defaultLoss, defaultLoss, defaultLoss);
		DistributionApproximation fft = DistributionAggregation.fftConvolution(
				new double[] {0.96, 0.03, 0.01}, 0.0,
				new double[] {0.96, 0.03, 0.01}, 0.0, 1.0);
		DistributionApproximation panjer = DistributionAggregation.panjerCompound(
				0.0, 2.0, Math.exp(-2.0), new double[] {0.0, 0.8, 0.2}, 40);
		DistributionApproximation cos = DistributionAggregation.cosInversion(
				new Normal(), -8.0, 8.0, 1024, 128);
		NumericalEstimate saddlepoint = DistributionTransforms.saddlepointCumulative(new Normal(), 1.2);
		System.out.printf("P(exact loss<=2) %.5f; FFT P(loss<=2) %.5f; Panjer tail bound %.3g%n",
				exact.getDistribution().cumulative(2.0), fft.getDistribution().cumulative(2.0),
				panjer.getDiagnostics().getAbsoluteError());
		System.out.printf("COS mass error %.3g; saddlepoint Phi(1.2) %.6f%n",
				cos.getDiagnostics().getAbsoluteError(), saddlepoint.getValue());
	}

	static void smoothOptionLawAndRisk() {
		OptionCurve curve = OptionCurve.build(100.0, 1.0, 1.0,
				new OptionObservation(80, true, 21.2), new OptionObservation(90, true, 13.6),
				new OptionObservation(100, true, 8.0), new OptionObservation(110, true, 4.8),
				new OptionObservation(120, true, 2.9));
		SmoothOptionDistributionResult smooth = curve.smoothDistribution(2.0);
		NumericalEstimate entropic = AdvancedRiskMeasures.entropic(new Normal(), 0.5, RiskConvention.LOSS);
		NumericalEstimate wang = AdvancedRiskMeasures.distortedExpectation(
				new Normal(), RiskConvention.LOSS, AdvancedRiskMeasures.wang(0.25));
		System.out.printf("smooth quote residual %.4g; differentiation uncertainty %.4g%n",
				smooth.getMaximumPriceResidual(), smooth.getDifferentiationUncertainty());
		System.out.printf("entropic risk %.5f; Wang-distorted risk %.5f%n",
				entropic.valueOrThrow(), wang.valueOrThrow());
	}

	static void pathAndLevyLaws() {
		DistributionApproximation drawdown = PathFunctionalDistributions.maximumDrawdown(
				new Normal(0.001, 0.02), 252, 10_000, 81L);
		Normal unitBrownianIncrement = new Normal();
		LevyIncrementDistribution quarter = new LevyIncrementDistribution(unitBrownianIncrement, 0.25);
		LevyIncrementDistribution year = quarter.plus(
				new LevyIncrementDistribution(unitBrownianIncrement, 0.75));
		System.out.printf("95%% maximum drawdown %.4f (MC error %.3g); annual increment CDF(0) %.3f%n",
				drawdown.getDistribution().quantile(0.95), drawdown.getDiagnostics().getAbsoluteError(),
				year.cumulative(0.0));
	}
}
