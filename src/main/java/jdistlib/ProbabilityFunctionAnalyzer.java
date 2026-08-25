/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.List;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationResult;
import jdistlib.math.IntegrationStabilityResult;
import jdistlib.math.UnivariateFunction;

/** Advisory probes for user-supplied probability kernels. */
public final class ProbabilityFunctionAnalyzer {
	private ProbabilityFunctionAnalyzer() {}

	public static FunctionAnalysis analyze(UnivariateFunction kernel, double lower,
			double upper) {
		return analyze(kernel, lower, upper, FunctionAnalysisOptions.defaults());
	}

	public static FunctionAnalysis analyze(UnivariateFunction kernel, double lower,
			double upper, FunctionAnalysisOptions options) {
		if (kernel == null || options == null) {
			throw new IllegalArgumentException("kernel and options must not be null");
		}
		if (Double.isNaN(lower) || Double.isNaN(upper) || !(lower < upper)) {
			throw new IllegalArgumentException("lower must be less than upper");
		}

		int count = options.getSampleCount();
		double[] x = new double[count];
		double[] y = new double[count];
		List<DiagnosticFinding> findings = new ArrayList<DiagnosticFinding>();
		List<Double> breaks = new ArrayList<Double>();
		double minimumPositive = Double.POSITIVE_INFINITY;
		double maximum = 0.0;
		int finiteCount = 0;
		int negativeCount = 0;
		int callbackFailureCount = 0;
		int nonFiniteCount = 0;
		for (int i = 0; i < count; i++) {
			double unit = (i + 0.5) / count;
			x[i] = mapUnit(unit, lower, upper);
			boolean callbackFailed = false;
			try {
				y[i] = kernel.eval(x[i]);
			} catch (RuntimeException exception) {
				y[i] = Double.NaN;
				callbackFailed = true;
				callbackFailureCount++;
				if (callbackFailureCount <= 8) {
					findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
							"CALLBACK_EXCEPTION", exception.getClass().getSimpleName()
									+ (exception.getMessage() == null ? ""
											: ": " + exception.getMessage()), x[i]));
				}
			}
			if (!Double.isFinite(y[i])) {
				if (!callbackFailed) {
					nonFiniteCount++;
					if (nonFiniteCount <= 8) {
						findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
								"NON_FINITE", "kernel returned " + y[i], x[i]));
					}
				}
				continue;
			}
			finiteCount++;
			if (y[i] < 0.0) {
				negativeCount++;
				if (negativeCount <= 8) {
					findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
							"NEGATIVE", "probability kernel is negative", x[i]));
				}
			} else if (y[i] > 0.0) {
				minimumPositive = Math.min(minimumPositive, y[i]);
				maximum = Math.max(maximum, y[i]);
			}
		}
		if (negativeCount > 8) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NEGATIVE_COUNT", (negativeCount - 8)
							+ " additional negative samples were omitted"));
		}
		if (callbackFailureCount > 8) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"CALLBACK_EXCEPTION_COUNT", (callbackFailureCount - 8)
							+ " additional callback exceptions were omitted"));
		}
		if (nonFiniteCount > 8) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NON_FINITE_COUNT", (nonFiniteCount - 8)
							+ " additional non-finite samples were omitted"));
		}
		if (finiteCount == 0 || !(maximum > 0.0)) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NO_POSITIVE_MASS", "no positive finite kernel value was observed"));
		}

		checkRepeatability(kernel, x, y, options.getRepeatabilityChecks(), findings);
		checkShape(x, y, options, findings, breaks);
		if (minimumPositive < Double.POSITIVE_INFINITY && maximum > 0.0) {
			double orders = Math.log10(maximum) - Math.log10(minimumPositive);
			if (orders > options.getDynamicRangeOrders()) {
				findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
						"DYNAMIC_RANGE", "observed dynamic range spans about "
								+ Math.round(orders) + " decimal orders; consider a log-kernel"));
			}
		}
		checkTails(y, lower, upper, findings);

		IntegrationStabilityResult stability = Integrate.assessStability(kernel,
				lower, upper, options.getIntegrationOptions());
		IntegrationResult normalization = stability.getTightened();
		if (!normalization.isSuccess()) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NORMALIZATION_FAILED", normalization.detailedMessage(),
					normalization.failureX));
		} else if (!(normalization.result > 0.0)
				|| !Double.isFinite(normalization.result)) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"INVALID_NORMALIZATION",
					"normalization must be finite and positive"));
		} else if (!stability.isStable()) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"UNSTABLE_NORMALIZATION", stability.message()));
		} else {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.INFO,
					"STABLE_NORMALIZATION", stability.message()));
		}

		double[] suggested = new double[breaks.size()];
		for (int i = 0; i < suggested.length; i++) suggested[i] = breaks.get(i);
		return new FunctionAnalysis(findings, count, minimumPositive, maximum,
				suggested, stability);
	}

	private static void checkRepeatability(UnivariateFunction kernel, double[] x,
			double[] y, int checks, List<DiagnosticFinding> findings) {
		for (int i = 0; i < checks; i++) {
			int index = checks == 1 ? x.length / 2
					: (int) Math.round(i * (x.length - 1.0) / Math.max(1, checks - 1));
			if (!Double.isFinite(y[index])) continue;
			try {
				double repeated = kernel.eval(x[index]);
				double tolerance = 64.0 * Math.ulp(Math.max(1.0, Math.abs(y[index])));
				if (!Double.isFinite(repeated)
						|| Math.abs(repeated - y[index]) > tolerance) {
					findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
							"NON_DETERMINISTIC",
							"repeated evaluation changed from " + y[index]
									+ " to " + repeated, x[index]));
				}
			} catch (RuntimeException exception) {
				findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
						"NON_DETERMINISTIC_EXCEPTION",
						"repeated evaluation threw "
								+ exception.getClass().getSimpleName(), x[index]));
			}
		}
	}

	private static void checkShape(double[] x, double[] y,
			FunctionAnalysisOptions options, List<DiagnosticFinding> findings,
			List<Double> breaks) {
		int changes = 0;
		double previousSlope = Double.NaN;
		for (int i = 1; i < y.length; i++) {
			if (!Double.isFinite(y[i - 1]) || !Double.isFinite(y[i])) continue;
			double small = Math.min(Math.abs(y[i - 1]), Math.abs(y[i]));
			double large = Math.max(Math.abs(y[i - 1]), Math.abs(y[i]));
			if (small > 0.0 && large / small > options.getDiscontinuityRatio()
					&& i > 1 && i + 1 < y.length) {
				double point = x[i - 1] * 0.5 + x[i] * 0.5;
				if (breaks.size() < 32) {
					findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
							"SHARP_CHANGE", "adjacent samples differ by a factor above "
									+ options.getDiscontinuityRatio(), point));
					breaks.add(point);
				}
			}
			double slope = y[i] - y[i - 1];
			if (slope != 0.0 && previousSlope != 0.0 && Double.isFinite(previousSlope)
					&& Math.copySign(1.0, slope) != Math.copySign(1.0, previousSlope)) {
				changes++;
			}
			if (slope != 0.0) previousSlope = slope;
		}
		if (changes > y.length / 3) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"OSCILLATORY", "frequent sampled slope reversals suggest oscillation"));
		}
	}

	private static void checkTails(double[] y, double lower, double upper,
			List<DiagnosticFinding> findings) {
		if (!Double.isFinite(lower) && y.length >= 4 && Double.isFinite(y[0])
				&& Double.isFinite(y[1]) && Math.abs(y[1]) > 0.0
				&& Math.abs(y[0]) >= 0.9 * Math.abs(y[1])) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"LEFT_TAIL", "sampled left tail does not show clear decay"));
		}
		int n = y.length;
		if (!Double.isFinite(upper) && n >= 4 && Double.isFinite(y[n - 1])
				&& Double.isFinite(y[n - 2])
				&& Math.abs(y[n - 2]) > 0.0
				&& Math.abs(y[n - 1]) >= 0.9 * Math.abs(y[n - 2])) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.WARNING,
					"RIGHT_TAIL", "sampled right tail does not show clear decay"));
		}
	}

	static double mapUnit(double unit, double lower, double upper) {
		if (Double.isFinite(lower) && Double.isFinite(upper)) {
			return lower * (1.0 - unit) + upper * unit;
		}
		if (Double.isFinite(lower)) {
			return lower + Math.tan(Math.PI * unit * 0.5);
		}
		if (Double.isFinite(upper)) {
			return upper - Math.tan(Math.PI * (1.0 - unit) * 0.5);
		}
		return Math.tan(Math.PI * (unit - 0.5));
	}
}
