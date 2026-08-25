/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.List;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationResult;
import jdistlib.math.IntegrationStabilityResult;

/** Self-consistency and moment diagnostics for numerical distributions. */
public final class NumericalDistributionAnalyzer {
	private static final double[] PROBES = {
		1e-6, 1e-4, 0.01, 0.05, 0.1, 0.25, 0.5,
		0.75, 0.9, 0.95, 0.99, 1.0 - 1e-4, 1.0 - 1e-6
	};

	private NumericalDistributionAnalyzer() {}

	public static DistributionAnalysis analyze(
			NumericalContinuousDistribution distribution) {
		if (distribution == null) {
			throw new IllegalArgumentException("distribution must not be null");
		}
		List<DiagnosticFinding> findings = new ArrayList<DiagnosticFinding>();
		IntegrationResult normalization = distribution.getNormalizationResult();
		double relativeError = normalization.result == 0.0 ? Double.POSITIVE_INFINITY
				: Math.abs(normalization.abserr / normalization.result);
		if (!normalization.isSuccess()) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NORMALIZATION_STATUS", normalization.detailedMessage()));
		} else if (relativeError > Math.max(1e-8,
				distribution.getIntegrationOptions().getRelativeTolerance() * 10.0)) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"NORMALIZATION_ERROR", "estimated relative normalization error is "
							+ relativeError));
		}

		double lowerEndpoint = distribution.cumulative(
				distribution.getLowerBound(), true, false);
		double upperEndpoint = distribution.cumulative(
				distribution.getUpperBound(), true, false);
		if (lowerEndpoint != 0.0 || upperEndpoint != 1.0) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"CDF_ENDPOINTS", "CDF endpoints are not exactly zero and one"));
		}

		double maxTail = 0.0;
		double maxRoundTrip = 0.0;
		double previousQuantile = Double.NEGATIVE_INFINITY;
		for (double probability : PROBES) {
			double quantile = distribution.quantile(probability, true, false);
			if (Double.isNaN(quantile) || quantile < previousQuantile) {
				findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
						"QUANTILE_MONOTONICITY",
						"quantile is NaN or decreases at p=" + probability));
				continue;
			}
			previousQuantile = quantile;
			double lower = distribution.cumulative(quantile, true, false);
			double upper = distribution.cumulative(quantile, false, false);
			if (!Double.isFinite(lower) || !Double.isFinite(upper)
					|| lower < 0.0 || lower > 1.0 || upper < 0.0 || upper > 1.0) {
				findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
						"CDF_RANGE", "CDF returned an invalid probability", quantile));
				continue;
			}
			maxTail = Math.max(maxTail, Math.abs((lower + upper) - 1.0));
			maxRoundTrip = Math.max(maxRoundTrip, Math.abs(lower - probability));
		}
		if (maxTail > 1e-8) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"TAIL_DISAGREEMENT", "direct lower and upper tails disagree by up to "
							+ maxTail));
		}
		if (maxRoundTrip > 1e-7) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"QUANTILE_ROUND_TRIP", "CDF/quantile round-trip error reached "
							+ maxRoundTrip));
		}

		IntegrationStabilityResult firstMoment = Integrate.assessStability(
				x -> x * distribution.density(x, false),
				distribution.getLowerBound(), distribution.getUpperBound(),
				distribution.getIntegrationOptions());
		double mean = firstMoment.getTightened().result;
		IntegrationStabilityResult firstAbsolute = Integrate.assessStability(
				x -> Math.abs(x) * distribution.density(x, false),
				distribution.getLowerBound(), distribution.getUpperBound(),
				distribution.getIntegrationOptions());
		double absoluteMean = firstAbsolute.getTightened().result;
		IntegrationStabilityResult secondAbsolute = Integrate.assessStability(
				x -> x * x * distribution.density(x, false),
				distribution.getLowerBound(), distribution.getUpperBound(),
				distribution.getIntegrationOptions());
		double second = secondAbsolute.getTightened().result;
		boolean momentsStable = firstMoment.isStable() && firstAbsolute.isStable()
				&& secondAbsolute.isStable() && Double.isFinite(mean)
				&& Double.isFinite(absoluteMean) && Double.isFinite(second);
		double variance = momentsStable ? Math.max(0.0, second - mean * mean)
				: Double.NaN;
		if (!momentsStable) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"ABSOLUTE_MOMENTS_UNSTABLE",
					"the first or second absolute moment did not converge stably; signed cancellation is not accepted as evidence of existence"));
		} else {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.INFO,
					"ABSOLUTE_MOMENTS_STABLE",
					"the first two absolute moments were stable under repeated integration"));
		}
		return new DistributionAnalysis(findings, relativeError, maxTail,
				maxRoundTrip, mean, variance, absoluteMean, second, momentsStable);
	}

	public static DistributionAnalysis analyze(
			NumericalDiscreteDistribution distribution) {
		if (distribution == null) {
			throw new IllegalArgumentException("distribution must not be null");
		}
		List<DiagnosticFinding> findings = new ArrayList<DiagnosticFinding>();
		double[] support = distribution.getSupport();
		double[] probabilities = distribution.getProbabilities();
		double sum = 0.0;
		double mean = 0.0;
		double absoluteMean = 0.0;
		double second = 0.0;
		for (int i = 0; i < support.length; i++) {
			if (!(probabilities[i] >= 0.0) || !Double.isFinite(probabilities[i])) {
				findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
						"INVALID_MASS", "invalid normalized probability", support[i]));
			}
			sum += probabilities[i];
			mean += support[i] * probabilities[i];
			absoluteMean += Math.abs(support[i]) * probabilities[i];
			second += support[i] * support[i] * probabilities[i];
		}
		double relativeError = Math.abs(sum - 1.0);
		if (relativeError > 1e-14) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"MASS_SUM", "normalized masses sum to " + sum));
		}
		double maxTail = 0.0;
		double maxRoundTrip = 0.0;
		for (double probability : PROBES) {
			double quantile = distribution.quantile(probability, true, false);
			double lower = distribution.cumulative(quantile, true, false);
			double upper = distribution.cumulative(quantile, false, false);
			maxTail = Math.max(maxTail, Math.abs((lower + upper) - 1.0));
			/* A discrete inverse may overshoot by the atom at the quantile. */
			double atom = distribution.density(quantile, false);
			maxRoundTrip = Math.max(maxRoundTrip,
					Math.max(0.0, Math.abs(lower - probability) - atom));
		}
		boolean momentsStable = Double.isFinite(mean)
				&& Double.isFinite(absoluteMean) && Double.isFinite(second);
		double variance = momentsStable ? Math.max(0.0, second - mean * mean)
				: Double.NaN;
		if (momentsStable) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.INFO,
					"FINITE_MOMENTS",
					"finite declared support guarantees finite mathematical moments"));
		} else {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"MOMENT_OVERFLOW",
					"finite-support moments overflowed double arithmetic"));
		}
		return new DistributionAnalysis(findings, relativeError, maxTail,
				maxRoundTrip, mean, variance, absoluteMean, second, momentsStable);
	}
}
