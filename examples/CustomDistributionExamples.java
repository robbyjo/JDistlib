/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.ConstructionPolicy;
import jdistlib.DiagnosticFinding;
import jdistlib.DiagnosticPreset;
import jdistlib.Distributions;
import jdistlib.FunctionAnalysisOptions;
import jdistlib.MixtureDistribution;
import jdistlib.NumericalContinuousDistribution;
import jdistlib.NumericalDiscreteDistribution;
import jdistlib.NumericalDistributionBuildResult;
import jdistlib.ProbabilityInterval;

/** Compilable counterparts to the custom-distribution guide snippets. */
public final class CustomDistributionExamples {
	private CustomDistributionExamples() {}

	/** Truncated standard-normal density proportional to exp(-x^2 / 2). */
	public static NumericalContinuousDistribution continuous() {
		return NumericalContinuousDistribution.builder()
				.logKernel(x -> -0.5 * x * x)
				.support(-8.0, 8.0)
				.diagnosticPreset(DiagnosticPreset.STANDARD)
				.build();
	}

	/** Mass proportional to (k + 1)^2 for k in {0,1,2,3,4}. */
	public static NumericalDiscreteDistribution discrete() {
		return NumericalDiscreteDistribution.builder()
				.weights(k -> (k + 1.0) * (k + 1.0))
				.integerSupport(0, 4)
				.build();
	}

	public static void main(String[] arguments) {
		FunctionAnalysisOptions checks = DiagnosticPreset.THOROUGH.options()
				.toBuilder().constructionPolicy(ConstructionPolicy.WARNING).build();
		NumericalDistributionBuildResult candidate =
				NumericalContinuousDistribution.analyzeLogKernel(
						x -> -0.5 * x * x, -8.0, 8.0, checks);
		for (DiagnosticFinding finding : candidate.getAnalysis().getFindings()) {
			System.out.println(finding);
		}

		NumericalContinuousDistribution first = candidate.build();
		NumericalContinuousDistribution second =
				NumericalContinuousDistribution.builder().kernel(x -> 1.0)
						.support(-2.0, 2.0).build();
		MixtureDistribution mixture = Distributions.mixture(
				new double[] {0.75, 0.25}, first, second);
		ProbabilityInterval interval = first.probabilityInterval(0.95);
		System.out.println(mixture.cumulative(0.0));
		System.out.println(interval.getLower() + " to " + interval.getUpper());
		System.out.println(discrete().random());
	}
}
