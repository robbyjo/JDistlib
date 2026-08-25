/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.UnivariateFunction;

/**
 * A finite discrete distribution obtained by normalizing a nonnegative weight
 * function over a declared set of numeric outcomes.
 *
 * <p>The weight function is evaluated once during construction. Outcomes are
 * sorted, must be unique and finite, and may be supplied explicitly or as an
 * inclusive integer range. Summation is scaled to avoid overflow when weights
 * have a large common magnitude.</p>
 */
public class NumericalDiscreteDistribution extends GenericDistribution
		implements SupportedDistribution {
	private static final int MAX_GENERATED_SUPPORT = 1000000;
	private final double[] support;
	private final double[] probabilities;
	private final double[] lowerCumulative;
	private final double[] upperCumulative;
	private final double normalization;
	private final double logNormalization;
	private final double[] aliasProbability;
	private final int[] aliasIndex;

	/** Returns a fluent builder for a finite custom discrete distribution. */
	public static Builder builder() { return new Builder(); }

	public static final class Builder {
		private UnivariateFunction weight;
		private UnivariateFunction logWeight;
		private double[] support;
		private Builder() {}
		public Builder weights(UnivariateFunction value) {
			weight = value;
			logWeight = null;
			return this;
		}
		public Builder logWeights(UnivariateFunction value) {
			logWeight = value;
			weight = null;
			return this;
		}
		public Builder support(double... values) {
			support = values == null ? null : values.clone();
			return this;
		}
		public Builder integerSupport(int lowerInclusive, int upperInclusive) {
			support = NumericalDiscreteDistribution.integerSupport(lowerInclusive,
					upperInclusive);
			return this;
		}
		public NumericalDiscreteDistribution build() {
			if ((weight == null) == (logWeight == null)) {
				throw new IllegalStateException("exactly one weights or logWeights function is required");
			}
			if (support == null || support.length == 0) {
				throw new IllegalStateException("nonempty support is required");
			}
			return logWeight == null
					? new NumericalDiscreteDistribution(weight, support)
					: fromLogWeights(logWeight, support);
		}
	}

	/**
	 * Constructs a distribution over an arbitrary finite set of outcomes.
	 *
	 * @param weight nonnegative unnormalized probability-mass function
	 * @param support finite, unique numeric outcomes
	 * @throws IllegalArgumentException if the support or a weight is invalid
	 */
	public NumericalDiscreteDistribution(UnivariateFunction weight,
			double[] support) {
		this(weight, support, 0.0);
	}

	private NumericalDiscreteDistribution(UnivariateFunction weight,
			double[] support, double logScale) {
		if (weight == null) throw new IllegalArgumentException("weight must not be null");
		if (support == null || support.length == 0) {
			throw new IllegalArgumentException("support must contain at least one outcome");
		}
		this.support = support.clone();
		Arrays.sort(this.support);
		for (int i = 0; i < this.support.length; i++) {
			if (!Double.isFinite(this.support[i])) {
				throw new IllegalArgumentException("support outcomes must be finite");
			}
			if (i > 0 && this.support[i] == this.support[i - 1]) {
				throw new IllegalArgumentException("support outcomes must be unique");
			}
		}

		double[] weights = new double[this.support.length];
		double maximum = 0.0;
		for (int i = 0; i < this.support.length; i++) {
			weights[i] = weight.eval(this.support[i]);
			if (!(weights[i] >= 0.0) || Double.isInfinite(weights[i])) {
				throw new IllegalArgumentException(
						"weight must be finite and nonnegative at x=" + this.support[i]);
			}
			maximum = Math.max(maximum, weights[i]);
		}
		if (!(maximum > 0.0)) {
			throw new IllegalArgumentException("at least one weight must be positive");
		}

		double scaledSum = compensatedScaledSum(weights, maximum);
		logNormalization = Math.log(maximum) + Math.log(scaledSum) + logScale;
		double directSum = maximum * scaledSum;
		normalization = logScale == 0.0
				? (Double.isFinite(directSum) ? directSum : Double.POSITIVE_INFINITY)
				: Math.exp(logNormalization);
		probabilities = new double[weights.length];
		for (int i = 0; i < weights.length; i++) {
			probabilities[i] = (weights[i] / maximum) / scaledSum;
		}

		lowerCumulative = new double[probabilities.length];
		double sum = 0.0;
		double correction = 0.0;
		for (int i = 0; i < probabilities.length; i++) {
			double adjusted = probabilities[i] - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
			lowerCumulative[i] = Math.min(1.0, sum);
		}
		lowerCumulative[lowerCumulative.length - 1] = 1.0;

		upperCumulative = new double[probabilities.length];
		sum = 0.0;
		correction = 0.0;
		for (int i = probabilities.length - 1; i >= 0; i--) {
			double adjusted = probabilities[i] - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
			upperCumulative[i] = Math.min(1.0, sum);
		}
		upperCumulative[0] = 1.0;
		aliasProbability = new double[probabilities.length];
		aliasIndex = new int[probabilities.length];
		buildAliasTable();
	}

	/**
	 * Constructs a distribution over every integer in the inclusive range.
	 */
	public NumericalDiscreteDistribution(UnivariateFunction weight,
			int lowerInclusive, int upperInclusive) {
		this(weight, integerSupport(lowerInclusive, upperInclusive));
	}

	/** Constructs over explicit outcomes from unnormalized log-weights. */
	public static NumericalDiscreteDistribution fromLogWeights(
			UnivariateFunction logWeight, double[] support) {
		if (logWeight == null || support == null || support.length == 0) {
			throw new IllegalArgumentException("logWeight and nonempty support are required");
		}
		double reference = Double.NEGATIVE_INFINITY;
		for (double outcome : support) {
			if (!Double.isFinite(outcome)) {
				throw new IllegalArgumentException("support outcomes must be finite");
			}
			double value = logWeight.eval(outcome);
			if (Double.isNaN(value) || value == Double.POSITIVE_INFINITY) {
				throw new IllegalArgumentException(
						"log-weight must not return NaN or positive infinity at x=" + outcome);
			}
			reference = Math.max(reference, value);
		}
		if (!Double.isFinite(reference)) {
			throw new IllegalArgumentException("at least one log-weight must be finite");
		}
		final double scale = reference;
		return new NumericalDiscreteDistribution(x -> {
			double value = logWeight.eval(x);
			return value == Double.NEGATIVE_INFINITY ? 0.0 : Math.exp(value - scale);
		}, support, scale);
	}

	/** Constructs over an inclusive integer range from unnormalized log-weights. */
	public static NumericalDiscreteDistribution fromLogWeights(
			UnivariateFunction logWeight, int lowerInclusive, int upperInclusive) {
		return fromLogWeights(logWeight,
				integerSupport(lowerInclusive, upperInclusive));
	}

	/** Returns the sorted declared support. */
	public double[] getSupport() { return support.clone(); }
	@Override public double getLowerBound() { return support[0]; }
	@Override public double getUpperBound() { return support[support.length - 1]; }

	/** Returns probabilities corresponding to {@link #getSupport()}. */
	public double[] getProbabilities() { return probabilities.clone(); }

	/**
	 * Returns the normalization constant, or positive infinity if its magnitude
	 * exceeds the representable {@code double} range.
	 */
	public double getNormalizationConstant() { return normalization; }

	/** Returns the log normalization constant without overflow. */
	public double getLogNormalizationConstant() { return logNormalization; }

	/** Runs mass, CDF, quantile, tail, and moment diagnostics. */
	public DistributionAnalysis analyzeDistribution() {
		return NumericalDistributionAnalyzer.analyze(this);
	}

	/** Runs diagnostics with user-selected absolute-moment orders and tail split. */
	public DistributionAnalysis analyzeDistribution(MomentAnalysisOptions settings) {
		return NumericalDistributionAnalyzer.analyze(this, settings);
	}

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		int index = Arrays.binarySearch(support, x);
		double probability = index >= 0 ? probabilities[index] : 0.0;
		return log ? Math.log(probability) : probability;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		int firstGreater = upperBound(x);
		double probability;
		if (lowerTail) {
			probability = firstGreater == 0 ? 0.0
					: lowerCumulative[firstGreater - 1];
		} else {
			probability = firstGreater == support.length ? 0.0
					: upperCumulative[firstGreater];
		}
		return logP ? Math.log(probability) : probability;
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		double probability = logP ? Math.exp(p) : p;
		double target = lowerTail ? probability : 1.0 - probability;
		if (target <= 0.0) return support[0];
		if (target >= 1.0) return support[support.length - 1];
		int low = 0;
		int high = lowerCumulative.length - 1;
		while (low < high) {
			int middle = (low + high) >>> 1;
			if (lowerCumulative[middle] >= target) high = middle;
			else low = middle + 1;
		}
		return support[low];
	}

	@Override public double random() {
		int column = (int) (random.nextDouble() * support.length);
		if (column == support.length) column--;
		return support[random.nextDouble() < aliasProbability[column]
				? column : aliasIndex[column]];
	}

	public SamplingStrategy getSamplingStrategy() { return SamplingStrategy.WALKER_ALIAS; }
	public String getSamplingStrategyExplanation() {
		return "Walker alias table built once from the normalized finite support";
	}

	/** Exactly evaluates E[g(X)] over the retained finite support. */
	public double expectation(UnivariateFunction function) {
		if (function == null) throw new IllegalArgumentException("function must not be null");
		double sum = 0.0;
		double correction = 0.0;
		for (int i = 0; i < support.length; i++) {
			double adjusted = function.eval(support[i]) * probabilities[i] - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
		}
		return sum;
	}

	public double rawMoment(double order) {
		validateMomentOrder(order, false);
		return expectation(x -> Math.pow(x, order));
	}

	public double centralMoment(double order) {
		validateMomentOrder(order, true);
		double mean = expectation(x -> x);
		return expectation(x -> Math.pow(x - mean, order));
	}

	/** Shannon entropy in nats. */
	public double entropy() {
		double result = 0.0;
		for (double probability : probabilities) {
			if (probability > 0.0) result -= probability * Math.log(probability);
		}
		return result;
	}

	/** Returns the smallest outcome having maximum mass. */
	public double mode() {
		int best = 0;
		for (int i = 1; i < probabilities.length; i++) {
			if (probabilities[i] > probabilities[best]) best = i;
		}
		return support[best];
	}

	/** Returns an equal-tail interval; its actual discrete mass may exceed the request. */
	public ProbabilityInterval probabilityInterval(double probability) {
		if (!(probability > 0.0 && probability < 1.0)) {
			throw new IllegalArgumentException("probability must lie between zero and one");
		}
		double tail = (1.0 - probability) * 0.5;
		return new ProbabilityInterval(quantile(tail), quantile(1.0 - tail),
				probability, "equal-tail-discrete");
	}

	private void validateMomentOrder(double order, boolean requireInteger) {
		if (!(order >= 0.0) || !Double.isFinite(order)
				|| (requireInteger && order != Math.rint(order))
				|| (support[0] < 0.0 && order != Math.rint(order))) {
			throw new IllegalArgumentException(
					"moment order must be finite and nonnegative; negative support and central moments require integer orders");
		}
	}

	private void buildAliasTable() {
		int n = probabilities.length;
		double[] scaled = new double[n];
		int[] small = new int[n];
		int[] large = new int[n];
		int smallCount = 0;
		int largeCount = 0;
		for (int i = 0; i < n; i++) {
			scaled[i] = probabilities[i] * n;
			if (scaled[i] < 1.0) small[smallCount++] = i;
			else large[largeCount++] = i;
		}
		while (smallCount > 0 && largeCount > 0) {
			int below = small[--smallCount];
			int above = large[--largeCount];
			aliasProbability[below] = scaled[below];
			aliasIndex[below] = above;
			scaled[above] = scaled[above] + scaled[below] - 1.0;
			if (scaled[above] < 1.0) small[smallCount++] = above;
			else large[largeCount++] = above;
		}
		while (largeCount > 0) {
			int index = large[--largeCount];
			aliasProbability[index] = 1.0;
			aliasIndex[index] = index;
		}
		while (smallCount > 0) {
			int index = small[--smallCount];
			aliasProbability[index] = 1.0;
			aliasIndex[index] = index;
		}
	}

	private int upperBound(double x) {
		int low = 0;
		int high = support.length;
		while (low < high) {
			int middle = (low + high) >>> 1;
			if (support[middle] <= x) low = middle + 1;
			else high = middle;
		}
		return low;
	}

	private static double compensatedScaledSum(double[] values, double scale) {
		double sum = 0.0;
		double correction = 0.0;
		for (double value : values) {
			double adjusted = value / scale - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
		}
		return sum;
	}

	private static double[] integerSupport(int lower, int upper) {
		if (lower > upper) {
			throw new IllegalArgumentException(
					"lower integer bound must not exceed upper bound");
		}
		long size = (long) upper - lower + 1L;
		if (size > MAX_GENERATED_SUPPORT) {
			throw new IllegalArgumentException("generated integer support exceeds "
					+ MAX_GENERATED_SUPPORT + " outcomes");
		}
		double[] result = new double[(int) size];
		for (int i = 0; i < result.length; i++) result[i] = (double) lower + i;
		return result;
	}
}
