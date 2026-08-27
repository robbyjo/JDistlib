/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.ConstructionPolicy;
import jdistlib.DiagnosticPreset;
import jdistlib.Distributions;
import jdistlib.FunctionAnalysisOptions;
import jdistlib.MixtureDistribution;
import jdistlib.NumericalContinuousDistribution;
import jdistlib.NumericalDiscreteDistribution;
import jdistlib.NumericalDistributionBuildResult;

/** Analyze, build, and compose custom continuous and discrete laws. */
public final class CustomDistributionIntegrationExamples {
	private CustomDistributionIntegrationExamples() {}
	public static NumericalContinuousDistribution boundedSensorError() {
		return NumericalContinuousDistribution.builder()
				.logKernel(x -> -Math.abs(x) / 0.55)
				.support(-2, 2).singularities(0)
				.diagnosticPreset(DiagnosticPreset.THOROUGH).build();
	}
	public static NumericalDiscreteDistribution finiteCountLaw() {
		return NumericalDiscreteDistribution.builder()
				.weights(k -> 1.0 / (1.0 + k * k))
				.integerSupport(0, 100).build();
	}
	public static NumericalDistributionBuildResult analyzeHeavyTail() {
		FunctionAnalysisOptions options = DiagnosticPreset.THOROUGH.options()
				.toBuilder().constructionPolicy(ConstructionPolicy.WARNING).build();
		return NumericalContinuousDistribution.analyzeLogKernel(
				x -> -Math.log1p(x * x), -50, 50, options);
	}
	public static void main(String[] arguments) {
		NumericalContinuousDistribution error = boundedSensorError();
		MixtureDistribution contamination = Distributions.mixture(
				new double[] {0.95, 0.05}, error,
				NumericalContinuousDistribution.builder().kernel(x -> 1)
						.support(-2, 2).build());
		System.out.println("within tolerance="
				+ (contamination.cumulative(0.5) - contamination.cumulative(-0.5)));
		System.out.println("discrete median=" + finiteCountLaw().quantile(0.5));
		System.out.println("heavy-tail findings="
				+ analyzeHeavyTail().getAnalysis().getFindings().size());
	}
}
