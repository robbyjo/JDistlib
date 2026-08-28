/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;

/** Atom-aware tail-risk, partial-moment, and option-payoff functionals. */
public final class FinancialRisk {
	private static final int PANELS = 4096;
	private FinancialRisk() {}

	/** Loss VaR at confidence {@code level}; returns are converted to losses first. */
	public static double valueAtRisk(GenericDistribution distribution, double level,
			RiskConvention convention) {
		require(distribution, level, convention);
		return convention == RiskConvention.LOSS
				? distribution.quantile(level, true, false)
				: -distribution.quantile(1.0 - level, true, false);
	}

	/** Atom-aware expected loss beyond VaR, defined through the quantile integral. */
	public static NumericalEstimate expectedShortfall(GenericDistribution distribution,
			double level, RiskConvention convention) {
		require(distribution, level, convention);
		if (convention == RiskConvention.LOSS) {
			return quantileAverage(distribution, level, 1.0, 1.0);
		}
		return quantileAverage(distribution, 0.0, 1.0 - level, -1.0);
	}

	/** Lower/upper partial moment E[(threshold-X)+^order] or E[(X-threshold)+^order]. */
	public static NumericalEstimate partialMoment(GenericDistribution distribution,
			double threshold, double order, Tail tail) {
		if (distribution == null || tail == null) throw new IllegalArgumentException("distribution and tail are required");
		if (!(order >= 0.0) || !Double.isFinite(order) || Double.isNaN(threshold))
			throw new IllegalArgumentException("order must be finite and nonnegative and threshold must not be NaN");
		return integrateQuantiles(distribution, 0.0, 1.0, q -> {
			double gap = tail == Tail.UPPER ? q - threshold : threshold - q;
			return gap > 0.0 ? Math.pow(gap, order) : 0.0;
		}, "quantile-partial-moment");
	}

	public static NumericalEstimate stopLoss(GenericDistribution distribution, double retention) {
		return partialMoment(distribution, retention, 1.0, Tail.UPPER);
	}

	public static NumericalEstimate callPayoff(GenericDistribution terminalPrice, double strike) {
		return stopLoss(terminalPrice, strike);
	}

	public static NumericalEstimate putPayoff(GenericDistribution terminalPrice, double strike) {
		return partialMoment(terminalPrice, strike, 1.0, Tail.LOWER);
	}

	public static NumericalEstimate downsideDeviation(GenericDistribution returns,
			double minimumAcceptableReturn) {
		NumericalEstimate moment = partialMoment(returns, minimumAcceptableReturn, 2.0, Tail.LOWER);
		return new NumericalEstimate(Math.sqrt(moment.getValue()),
				moment.getAbsoluteError() / Math.max(2.0 * Math.sqrt(moment.getValue()), 1e-300),
				moment.isConverged(), moment.getEvaluations(), "downside-deviation", moment.getWarning());
	}

	public static double shortfallProbability(GenericDistribution distribution, double target) {
		if (distribution == null) throw new IllegalArgumentException("distribution is required");
		return distribution.cumulative(target, true, false);
	}

	public static NumericalEstimate expectedShortfallMagnitude(GenericDistribution distribution,
			double target) {
		double probability = shortfallProbability(distribution, target);
		NumericalEstimate partial = partialMoment(distribution, target, 1.0, Tail.LOWER);
		if (probability == 0.0) return new NumericalEstimate(0.0, partial.getAbsoluteError(), true,
				partial.getEvaluations(), "conditional-shortfall", "event has zero probability");
		return new NumericalEstimate(partial.getValue() / probability,
				partial.getAbsoluteError() / probability, partial.isConverged(), partial.getEvaluations(),
				"conditional-shortfall", partial.getWarning());
	}

	/** Asymmetric least-squares expectile with a bracket/convergence report. */
	public static NumericalEstimate expectile(GenericDistribution distribution, double probability) {
		if (distribution == null || !(probability > 0.0 && probability < 1.0))
			throw new IllegalArgumentException("distribution is required and probability must be in (0,1)");
		double low = distribution.quantile(1e-8, true, false);
		double high = distribution.quantile(1.0 - 1e-8, true, false);
		if (!Double.isFinite(low) || !Double.isFinite(high))
			return new NumericalEstimate(Double.NaN, Double.POSITIVE_INFINITY, false, 0,
					"expectile-bisection", "finite bracketing quantiles do not exist");
		int evaluations = 0;
		for (int iteration = 0; iteration < 60; iteration++) {
			double middle = low + (high - low) / 2.0;
			NumericalEstimate upper = partialMoment(distribution, middle, 1.0, Tail.UPPER);
			NumericalEstimate lower = partialMoment(distribution, middle, 1.0, Tail.LOWER);
			evaluations += upper.getEvaluations() + lower.getEvaluations();
			double score = probability * upper.getValue() - (1.0 - probability) * lower.getValue();
			if (!Double.isFinite(score)) return new NumericalEstimate(Double.NaN,
					Double.POSITIVE_INFINITY, false, evaluations, "expectile-bisection",
					"required first partial moment does not exist");
			if (score > 0.0) low = middle; else high = middle;
		}
		return new NumericalEstimate(low + (high - low) / 2.0, (high - low) / 2.0,
				true, evaluations, "expectile-bisection", "");
	}

	public static void valueAtRiskInto(GenericDistribution distribution, double[] levels,
			int inputOffset, double[] output, int outputOffset, int length,
			RiskConvention convention) {
		if (levels == null || output == null || inputOffset < 0 || outputOffset < 0 || length < 0
				|| inputOffset > levels.length - length || outputOffset > output.length - length)
			throw new IllegalArgumentException("invalid array range");
		for (int i = 0; i < length; i++) output[outputOffset + i] =
				valueAtRisk(distribution, levels[inputOffset + i], convention);
	}

	private interface QuantileFunction { double value(double quantile); }

	private static NumericalEstimate quantileAverage(GenericDistribution distribution,
			double from, double to, double sign) {
		NumericalEstimate integral = integrateQuantiles(distribution, from, to,
				q -> sign * q, "quantile-expected-shortfall");
		double width = to - from;
		return new NumericalEstimate(integral.getValue() / width,
				integral.getAbsoluteError() / width, integral.isConverged(),
				integral.getEvaluations(), integral.getStrategy(), integral.getWarning());
	}

	private static NumericalEstimate integrateQuantiles(GenericDistribution distribution,
			double from, double to, QuantileFunction function, String strategy) {
		double coarse = midpoint(distribution, from, to, PANELS / 2, function);
		double fine = midpoint(distribution, from, to, PANELS, function);
		boolean finite = Double.isFinite(fine);
		return new NumericalEstimate(fine, finite ? Math.abs(fine - coarse) : Double.POSITIVE_INFINITY,
				finite, PANELS + PANELS / 2, strategy,
				finite ? "" : "required moment does not exist or overflowed");
	}

	private static double midpoint(GenericDistribution distribution, double from, double to,
			int panels, QuantileFunction function) {
		double width = (to - from) / panels;
		double sum = 0.0;
		for (int i = 0; i < panels; i++) {
			double p = from + (i + 0.5) * width;
			sum += function.value(distribution.quantile(p, true, false));
		}
		return sum * width;
	}

	private static void require(GenericDistribution distribution, double level,
			RiskConvention convention) {
		if (distribution == null || convention == null || !(level > 0.0 && level < 1.0))
			throw new IllegalArgumentException("distribution/convention are required and level must be in (0,1)");
	}
}
