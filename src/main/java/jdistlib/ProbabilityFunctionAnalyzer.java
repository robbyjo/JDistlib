/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

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
		List<DiagnosticFinding> findings = new ArrayList<DiagnosticFinding>();
		List<Double> breaks = new ArrayList<Double>();
		ProbeCounts counts = new ProbeCounts();
		List<Probe> probes = new ArrayList<Probe>(count
				+ options.getRandomizedProbeBudget());
		for (int i = 0; i < count; i++) {
			double unit = (i + 0.5) / count;
			Probe probe = new Probe(unit, mapUnit(unit, lower, upper));
			evaluate(kernel, probe, findings, counts);
			probes.add(probe);
		}

		int randomBudget = options.getRandomizedProbeBudget();
		if (randomBudget > 0) {
			Random random = new Random(options.getRandomSeed());
			int exploratory = Math.min(randomBudget, Math.max(1,
					(randomBudget + 1) / 2));
			for (int i = 0; i < exploratory; i++) {
				double unit = (i + random.nextDouble()) / exploratory;
				Probe probe = new Probe(unit, mapUnit(unit, lower, upper));
				evaluate(kernel, probe, findings, counts);
				probes.add(probe);
			}
			int remaining = randomBudget - exploratory;
			for (int round = 0; round < options.getAdaptiveProbeRounds()
					&& remaining > 0; round++) {
				Collections.sort(probes, Probe.BY_UNIT);
				int roundsLeft = options.getAdaptiveProbeRounds() - round;
				int inRound = Math.max(1, remaining / roundsLeft);
				int interval = selectAdaptiveInterval(probes, random);
				double focusLeft = probes.get(interval).unit;
				double focusRight = probes.get(interval + 1).unit;
				for (int i = 0; i < inRound && remaining > 0; i++, remaining--) {
					double unit;
					if ((i & 3) == 3) {
						unit = random.nextDouble();
					} else {
						unit = focusLeft + (focusRight - focusLeft)
								* random.nextDouble();
					}
					Probe probe = new Probe(unit, mapUnit(unit, lower, upper));
					evaluate(kernel, probe, findings, counts);
					probes.add(probe);
				}
			}
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.INFO,
					"RANDOMIZED_PROBES", randomBudget
							+ " seeded adaptive probes completed with seed "
							+ options.getRandomSeed()));
		}

		Collections.sort(probes, Probe.BY_UNIT);
		double[] x = new double[probes.size()];
		double[] y = new double[probes.size()];
		for (int i = 0; i < probes.size(); i++) {
			x[i] = probes.get(i).x;
			y[i] = probes.get(i).value;
		}
		if (counts.negative > 8) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NEGATIVE_COUNT", (counts.negative - 8)
							+ " additional negative samples were omitted"));
		}
		if (counts.callbackFailures > 8) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"CALLBACK_EXCEPTION_COUNT", (counts.callbackFailures - 8)
							+ " additional callback exceptions were omitted"));
		}
		if (counts.nonFinite > 8) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NON_FINITE_COUNT", (counts.nonFinite - 8)
							+ " additional non-finite samples were omitted"));
		}
		if (counts.finite == 0 || !(counts.maximum > 0.0)) {
			findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
					"NO_POSITIVE_MASS", "no positive finite kernel value was observed"));
		}

		checkRepeatability(kernel, x, y, options.getRepeatabilityChecks(), findings);
		checkShape(x, y, options, findings, breaks);
		if (counts.minimumPositive < Double.POSITIVE_INFINITY
				&& counts.maximum > 0.0) {
			double orders = Math.log10(counts.maximum)
					- Math.log10(counts.minimumPositive);
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
		return new FunctionAnalysis(findings, probes.size(), randomBudget,
				options.getRandomSeed(), counts.minimumPositive, counts.maximum,
				suggested, stability);
	}

	private static void evaluate(UnivariateFunction kernel, Probe probe,
			List<DiagnosticFinding> findings, ProbeCounts counts) {
		boolean callbackFailed = false;
		try {
			probe.value = kernel.eval(probe.x);
		} catch (RuntimeException exception) {
			probe.value = Double.NaN;
			callbackFailed = true;
			counts.callbackFailures++;
			if (counts.callbackFailures <= 8) {
				findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
						"CALLBACK_EXCEPTION", exception.getClass().getSimpleName()
								+ (exception.getMessage() == null ? ""
										: ": " + exception.getMessage()), probe.x));
			}
		}
		if (!Double.isFinite(probe.value)) {
			if (!callbackFailed) {
				counts.nonFinite++;
				if (counts.nonFinite <= 8) {
					findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
							"NON_FINITE", "kernel returned " + probe.value, probe.x));
				}
			}
			return;
		}
		counts.finite++;
		if (probe.value < 0.0) {
			counts.negative++;
			if (counts.negative <= 8) {
				findings.add(new DiagnosticFinding(DiagnosticFinding.Severity.ERROR,
						"NEGATIVE", "probability kernel is negative", probe.x));
			}
		} else if (probe.value > 0.0) {
			counts.minimumPositive = Math.min(counts.minimumPositive, probe.value);
			counts.maximum = Math.max(counts.maximum, probe.value);
		}
	}

	private static int selectAdaptiveInterval(List<Probe> probes, Random random) {
		double total = 0.0;
		double[] scores = new double[probes.size() - 1];
		for (int i = 0; i < scores.length; i++) {
			Probe left = probes.get(i);
			Probe right = probes.get(i + 1);
			double width = right.unit - left.unit;
			double shape = 0.0;
			if (left.value > 0.0 && right.value > 0.0
					&& Double.isFinite(left.value) && Double.isFinite(right.value)) {
				shape = Math.min(30.0,
						Math.abs(Math.log(left.value) - Math.log(right.value)));
			} else if (left.value != right.value) {
				shape = 8.0;
			}
			scores[i] = width * (1.0 + shape);
			total += scores[i];
		}
		if (!(total > 0.0)) return random.nextInt(scores.length);
		double target = random.nextDouble() * total;
		for (int i = 0; i < scores.length; i++) {
			target -= scores[i];
			if (target <= 0.0) return i;
		}
		return scores.length - 1;
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

	private static final class Probe {
		static final Comparator<Probe> BY_UNIT = new Comparator<Probe>() {
			@Override public int compare(Probe left, Probe right) {
				return Double.compare(left.unit, right.unit);
			}
		};
		final double unit;
		final double x;
		double value;

		Probe(double unit, double x) {
			this.unit = unit;
			this.x = x;
		}
	}

	private static final class ProbeCounts {
		double minimumPositive = Double.POSITIVE_INFINITY;
		double maximum;
		int finite;
		int negative;
		int callbackFailures;
		int nonFinite;
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
