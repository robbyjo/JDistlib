/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.UnivariateFunction;

/**
 * Finite approximation to an infinite integer-supported distribution, stopped
 * only when user-provided tail certificates bound the omitted probability.
 */
public final class CertifiedInfiniteDiscreteDistribution extends GenericDistribution
		implements SupportedDistribution {
	private static final long MAX_EXACT_INTEGER = 9007199254740992L;
	private final NumericalDiscreteDistribution truncated;
	private final double tailWeightUpperBound;
	private final double omittedProbabilityUpperBound;
	private final int includedTerms;
	private final String supportDescription;

	private CertifiedInfiniteDiscreteDistribution(Prepared prepared) {
		truncated = new NumericalDiscreteDistribution(prepared.function, prepared.support);
		tailWeightUpperBound = prepared.tailWeightUpperBound;
		omittedProbabilityUpperBound = prepared.omittedProbabilityUpperBound;
		includedTerms = prepared.support.length;
		supportDescription = prepared.description;
	}

	/** Constructs support {@code start, start+1, ...}. */
	public static CertifiedInfiniteDiscreteDistribution rightInfinite(
			UnivariateFunction weight, long start, DiscreteTailBound tailBound,
			CertifiedDiscreteOptions options) {
		return new CertifiedInfiniteDiscreteDistribution(
				oneSided(weight, start, 1, tailBound, options));
	}

	/** Constructs support {@code ..., start-1, start}. */
	public static CertifiedInfiniteDiscreteDistribution leftInfinite(
			UnivariateFunction weight, long start, DiscreteTailBound tailBound,
			CertifiedDiscreteOptions options) {
		return new CertifiedInfiniteDiscreteDistribution(
				oneSided(weight, start, -1, tailBound, options));
	}

	/** Constructs two-sided integer support around a finite center. */
	public static CertifiedInfiniteDiscreteDistribution twoSided(
			UnivariateFunction weight, long center, DiscreteTailBound leftTailBound,
			DiscreteTailBound rightTailBound, CertifiedDiscreteOptions options) {
		validate(weight, center, leftTailBound, options);
		if (rightTailBound == null) {
			throw new IllegalArgumentException("rightTailBound must not be null");
		}
		List<Double> outcomes = new ArrayList<Double>();
		List<Double> weights = new ArrayList<Double>();
		ScaledAccumulator sum = new ScaledAccumulator();
		add(weight, center, outcomes, weights, sum);
		double tail = Double.POSITIVE_INFINITY;
		double probabilityBound = 1.0;
		long radius = 0L;
		while (outcomes.size() < options.getMaximumTerms()) {
			radius++;
			if (center - radius < -MAX_EXACT_INTEGER
					|| center + radius > MAX_EXACT_INTEGER) {
				throw new IllegalArgumentException("integer support exceeded exact double range");
			}
			add(weight, center - radius, outcomes, weights, sum);
			if (outcomes.size() >= options.getMaximumTerms()) break;
			add(weight, center + radius, outcomes, weights, sum);
			if (outcomes.size() < options.getMinimumTerms()) continue;
			long firstLeft = center - radius - 1L;
			long firstRight = center + radius + 1L;
			double leftWeight = checkedWeight(weight, firstLeft);
			double rightWeight = checkedWeight(weight, firstRight);
			double left = certifiedBound(leftTailBound, firstLeft, leftWeight);
			double right = certifiedBound(rightTailBound, firstRight, rightWeight);
			tail = left + right;
			probabilityBound = sum.omittedProbabilityBound(tail);
			if (probabilityBound <= options.getOmittedProbabilityTolerance()) {
				return new CertifiedInfiniteDiscreteDistribution(prepared(
						outcomes, weights, tail, probabilityBound,
						"two-sided integer support around " + center));
			}
		}
		throw new IllegalArgumentException(
				"certified two-sided tail tolerance was not reached within maximumTerms");
	}

	public double getTailWeightUpperBound() { return tailWeightUpperBound; }
	public double getOmittedProbabilityUpperBound() {
		return omittedProbabilityUpperBound;
	}
	public int getIncludedTerms() { return includedTerms; }
	public String getSupportDescription() { return supportDescription; }
	public NumericalDiscreteDistribution getTruncatedDistribution() {
		return truncated;
	}
	@Override public double getLowerBound() { return truncated.getLowerBound(); }
	@Override public double getUpperBound() { return truncated.getUpperBound(); }

	@Override public double density(double x, boolean log) {
		return truncated.density(x, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return truncated.cumulative(x, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return truncated.quantile(p, lowerTail, logP);
	}
	@Override public double random() { return truncated.quantile(random.nextDouble()); }

	private static Prepared oneSided(UnivariateFunction weight, long start,
			int direction, DiscreteTailBound tailBound,
			CertifiedDiscreteOptions options) {
		validate(weight, start, tailBound, options);
		List<Double> outcomes = new ArrayList<Double>();
		List<Double> weights = new ArrayList<Double>();
		ScaledAccumulator sum = new ScaledAccumulator();
		for (int i = 0; i < options.getMaximumTerms(); i++) {
			long outcome;
			try {
				outcome = Math.addExact(start, (long) direction * i);
			} catch (ArithmeticException exception) {
				throw new IllegalArgumentException("integer support overflowed", exception);
			}
			if (Math.abs(outcome) > MAX_EXACT_INTEGER) {
				throw new IllegalArgumentException("integer support exceeded exact double range");
			}
			add(weight, outcome, outcomes, weights, sum);
			if (outcomes.size() < options.getMinimumTerms()) continue;
			long firstOmitted = outcome + direction;
			double firstWeight = checkedWeight(weight, firstOmitted);
			double bound = certifiedBound(tailBound, firstOmitted, firstWeight);
			double probabilityBound = sum.omittedProbabilityBound(bound);
			if (probabilityBound <= options.getOmittedProbabilityTolerance()) {
				return prepared(outcomes, weights, bound, probabilityBound,
						direction > 0 ? "right-infinite integer support from " + start
								: "left-infinite integer support through " + start);
			}
		}
		throw new IllegalArgumentException(
				"certified tail tolerance was not reached within maximumTerms");
	}

	private static void validate(UnivariateFunction weight, long start,
			DiscreteTailBound tailBound, CertifiedDiscreteOptions options) {
		if (weight == null || tailBound == null || options == null) {
			throw new IllegalArgumentException("weight, tail bound, and options are required");
		}
		if (Math.abs(start) > MAX_EXACT_INTEGER) {
			throw new IllegalArgumentException("start is outside exact double integer range");
		}
	}

	private static void add(UnivariateFunction weight, long outcome,
			List<Double> outcomes, List<Double> weights, ScaledAccumulator sum) {
		double value = checkedWeight(weight, outcome);
		outcomes.add((double) outcome);
		weights.add(value);
		sum.add(value);
	}

	private static double checkedWeight(UnivariateFunction weight, long outcome) {
		double value = weight.eval((double) outcome);
		if (!(value >= 0.0) || !Double.isFinite(value)) {
			throw new IllegalArgumentException("invalid weight at integer " + outcome);
		}
		return value;
	}

	private static double certifiedBound(DiscreteTailBound certificate,
			long firstOmitted, double firstWeight) {
		double bound = certificate.upperBound(firstOmitted, firstWeight);
		if (!(bound >= firstWeight) || !Double.isFinite(bound)) {
			throw new IllegalArgumentException("tail certificate at " + firstOmitted
					+ " must be finite and include the first omitted weight");
		}
		return bound;
	}

	private static Prepared prepared(List<Double> outcomes,
			List<Double> weights, double tail,
			double probabilityBound, String description) {
		double[] support = new double[outcomes.size()];
		double[] raw = new double[weights.size()];
		for (int i = 0; i < support.length; i++) {
			support[i] = outcomes.get(i);
			raw[i] = weights.get(i);
		}
		sortPairs(support, raw, 0, support.length - 1);
		UnivariateFunction retained = x -> {
			int index = Arrays.binarySearch(support, x);
			return index < 0 ? 0.0 : raw[index];
		};
		return new Prepared(retained, support, raw, tail, probabilityBound,
				description);
	}

	private static void sortPairs(double[] support, double[] weight, int low,
			int high) {
		int i = low;
		int j = high;
		double pivot = support[(low + high) >>> 1];
		while (i <= j) {
			while (support[i] < pivot) i++;
			while (support[j] > pivot) j--;
			if (i <= j) {
				double x = support[i];
				support[i] = support[j];
				support[j] = x;
				double w = weight[i];
				weight[i] = weight[j];
				weight[j] = w;
				i++;
				j--;
			}
		}
		if (low < j) sortPairs(support, weight, low, j);
		if (i < high) sortPairs(support, weight, i, high);
	}

	private static final class ScaledAccumulator {
		double scale;
		double scaledSum;
		void add(double value) {
			if (value == 0.0) return;
			if (scale == 0.0) {
				scale = value;
				scaledSum = 1.0;
			} else if (value > scale) {
				scaledSum = scaledSum * (scale / value) + 1.0;
				scale = value;
			} else {
				scaledSum += value / scale;
			}
		}
		double omittedProbabilityBound(double tail) {
			if (!(scale > 0.0)) return 1.0;
			double scaledTail = tail / scale;
			if (Double.isInfinite(scaledTail)) return 1.0;
			return scaledTail / (scaledSum + scaledTail);
		}
	}

	private static final class Prepared {
		final UnivariateFunction function;
		final double[] support;
		final double[] weight;
		final double tailWeightUpperBound;
		final double omittedProbabilityUpperBound;
		final String description;
		Prepared(UnivariateFunction function, double[] support, double[] weight,
				double tailWeightUpperBound, double omittedProbabilityUpperBound,
				String description) {
			this.function = function;
			this.support = support;
			this.weight = weight;
			this.tailWeightUpperBound = tailWeightUpperBound;
			this.omittedProbabilityUpperBound = omittedProbabilityUpperBound;
			this.description = description;
		}
	}
}
