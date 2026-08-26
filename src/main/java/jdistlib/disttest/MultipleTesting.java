/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.disttest;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

import jdistlib.math.VectorMath;
import jdistlib.math.spline.SmoothSpline;
import jdistlib.math.spline.SmoothSplineResult;

/**
 * Multiple-testing adjustments and Storey q-values.
 *
 * <p>The returned adjusted p-values retain the input order. {@code NaN} denotes
 * a missing p-value, is preserved in the result, and is not counted as a test
 * unless the overload with an explicit number of tests is used. All other
 * values must be finite probabilities in {@code [0, 1]}.</p>
 *
 * <p>This class is stateless and thread-safe.</p>
 */
public final class MultipleTesting {
	/** Result of the level-dependent two-stage BKY procedure. */
	public static final class AdaptiveFdrResult {
		private final boolean[] rejected;
		private final int rejectedCount;
		private final int stageOneRejections;
		private final int estimatedTrueNulls;
		private final double stageOneLevel;
		private final double finalLevel;
		private final double threshold;

		private AdaptiveFdrResult(boolean[] rejected, int rejectedCount,
				int stageOneRejections,
				int estimatedTrueNulls, double stageOneLevel,
				double finalLevel, double threshold) {
			this.rejected = rejected;
			this.rejectedCount = rejectedCount;
			this.stageOneRejections = stageOneRejections;
			this.estimatedTrueNulls = estimatedTrueNulls;
			this.stageOneLevel = stageOneLevel;
			this.finalLevel = finalLevel;
			this.threshold = threshold;
		}

		/** Returns rejection flags in input order; missing values are false. */
		public boolean[] getRejected() { return rejected.clone(); }
		/** Returns the final number of rejected hypotheses. */
		public int getRejectedCount() { return rejectedCount; }
		/** Returns the number rejected by the first-stage BH procedure. */
		public int getStageOneRejections() { return stageOneRejections; }
		/** Returns the first-stage estimate of the number of true nulls. */
		public int getEstimatedTrueNulls() { return estimatedTrueNulls; }
		/** Returns q/(1+q), the BH level used in stage one. */
		public double getStageOneLevel() { return stageOneLevel; }
		/** Returns the BH level used for the final stage. */
		public double getFinalLevel() { return finalLevel; }
		/** Returns the largest rejected raw p-value, or NaN. */
		public double getThreshold() { return threshold; }
	}

	/** Result for a family whose unrecorded p-values are known to exceed a limit. */
	public static final class CensoredTestResult {
		private final double[] adjustedPValues;
		private final boolean[] rejected;
		private final int unobservedCount;
		private final boolean decisionsExact;
		private final double threshold;

		private CensoredTestResult(double[] adjustedPValues,
				boolean[] rejected, int unobservedCount,
				boolean decisionsExact, double threshold) {
			this.adjustedPValues = adjustedPValues;
			this.rejected = rejected;
			this.unobservedCount = unobservedCount;
			this.decisionsExact = decisionsExact;
			this.threshold = threshold;
		}

		/**
		 * Returns adjusted values for recorded tests. Values are conservative
		 * when {@link #areDecisionsExact()} is false.
		 */
		public double[] getAdjustedPValues() { return adjustedPValues.clone(); }
		/** Returns rejection flags in input order; missing values are false. */
		public boolean[] getRejected() { return rejected.clone(); }
		/** Returns the number of recorded hypotheses rejected. */
		public int getRejectedCount() {
			int count = 0;
			for (boolean value : rejected) if (value) count++;
			return count;
		}
		/** Returns the number of unrecorded hypotheses. */
		public int getUnobservedCount() { return unobservedCount; }
		/**
		 * Returns whether the censoring limit rules out every unrecorded
		 * hypothesis from rejection at the requested level.
		 */
		public boolean areDecisionsExact() { return decisionsExact; }
		/** Returns the largest rejected recorded raw p-value, or NaN. */
		public double getThreshold() { return threshold; }
	}

	/** Result of a level-dependent step-down FDR procedure. */
	public static final class StepDownFdrResult {
		private final boolean[] rejected;
		private final int rejectedCount;
		private final double threshold;
		private final double criticalValue;

		private StepDownFdrResult(boolean[] rejected, int rejectedCount,
				double threshold, double criticalValue) {
			this.rejected = rejected;
			this.rejectedCount = rejectedCount;
			this.threshold = threshold;
			this.criticalValue = criticalValue;
		}

		/** Returns rejection flags in input order; missing values are false. */
		public boolean[] getRejected() { return rejected.clone(); }
		/** Returns the number of rejected hypotheses. */
		public int getRejectedCount() { return rejectedCount; }
		/** Returns the largest rejected raw p-value, or NaN. */
		public double getThreshold() { return threshold; }
		/** Returns the rank-specific cutoff at the last rejection, or NaN. */
		public double getCriticalValue() { return criticalValue; }
	}

	/** Result of the two-level Benjamini-Bogomolov grouped procedure. */
	public static final class GroupedFdrResult {
		private final boolean[] rejected;
		private final int[] groupLabels;
		private final double[] groupPValues;
		private final boolean[] selectedGroups;
		private final double withinGroupLevel;

		private GroupedFdrResult(boolean[] rejected, int[] groupLabels,
				double[] groupPValues, boolean[] selectedGroups,
				double withinGroupLevel) {
			this.rejected = rejected;
			this.groupLabels = groupLabels;
			this.groupPValues = groupPValues;
			this.selectedGroups = selectedGroups;
			this.withinGroupLevel = withinGroupLevel;
		}

		/** Returns hypothesis-level rejection flags in input order. */
		public boolean[] getRejected() { return rejected.clone(); }
		/** Returns the sorted group labels corresponding to the group arrays. */
		public int[] getGroupLabels() { return groupLabels.clone(); }
		/** Returns one Simes p-value per group. */
		public double[] getGroupPValues() { return groupPValues.clone(); }
		/** Returns group-selection flags in {@link #getGroupLabels()} order. */
		public boolean[] getSelectedGroups() { return selectedGroups.clone(); }
		/** Returns the number of selected groups. */
		public int getSelectedGroupCount() { return countTrue(selectedGroups); }
		/** Returns the BH level used inside every selected group. */
		public double getWithinGroupLevel() { return withinGroupLevel; }
	}

	/** Supported p-value adjustment procedures. */
	public enum Method {
		/** No adjustment. */
		NONE,
		/** Bonferroni family-wise error-rate control. */
		BONFERRONI,
		/** Holm step-down family-wise error-rate control. */
		HOLM,
		/** Hochberg step-up control under independence or positive dependence. */
		HOCHBERG,
		/** Hommel control under independence or positive dependence. */
		HOMMEL,
		/** Single-step Sidak control for independent tests. */
		SIDAK,
		/** Holm-Sidak step-down control for independent tests. */
		HOLM_SIDAK,
		/** Benjamini-Hochberg false-discovery-rate control. */
		BENJAMINI_HOCHBERG,
		/** Benjamini-Yekutieli FDR control under arbitrary dependence. */
		BENJAMINI_YEKUTIELI
	}

	private static final double[] DEFAULT_LAMBDAS = {
		0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
		0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
	};
	private static final double DEFAULT_SMOOTHING_DF = 3.0;
	private static final double DEFAULT_PI0_QUANTILE = 0.10;

	private MultipleTesting() {}

	/**
	 * Adjusts p-values, counting only non-missing inputs as tests.
	 *
	 * @param pValues raw p-values
	 * @param method adjustment procedure
	 * @return adjusted p-values in the original order
	 */
	public static double[] adjust(double[] pValues, Method method) {
		ValidatedPValues validated = validatePValues(pValues);
		return adjust(validated, method, validated.values.length);
	}

	/**
	 * Adjusts p-values for a declared total number of comparisons.
	 *
	 * <p>The declared count may exceed the number of observed non-missing
	 * p-values. Unobserved values are treated as larger than the observed values
	 * for Bonferroni and Holm and as one for the other stepwise procedures, in
	 * the same manner as R's {@code p.adjust(..., n=)} contract.</p>
	 *
	 * @param pValues raw p-values
	 * @param method adjustment procedure
	 * @param numberOfTests total number of comparisons
	 * @return adjusted p-values in the original order
	 */
	public static double[] adjust(double[] pValues, Method method,
			int numberOfTests) {
		return adjust(validatePValues(pValues), method, numberOfTests);
	}

	/**
	 * Adjusts natural-log p-values without exponentiating them.
	 *
	 * <p>This is the preferred entry point for p-values that may underflow in
	 * ordinary floating-point representation. Inputs must be {@code NaN},
	 * negative infinity, or finite values in {@code [-infinity, 0]}; outputs
	 * are natural logs of the adjusted p-values.</p>
	 */
	public static double[] adjustLog(double[] logPValues, Method method) {
		ValidatedPValues validated = validateLogPValues(logPValues);
		return adjustLog(validated, method, validated.values.length);
	}

	/** Adjusts natural-log p-values for a declared total family size. */
	public static double[] adjustLog(double[] logPValues, Method method,
			int numberOfTests) {
		return adjustLog(validateLogPValues(logPValues), method, numberOfTests);
	}

	/**
	 * Computes weighted Benjamini-Hochberg adjusted p-values.
	 *
	 * <p>Weights must be finite, strictly positive, and chosen independently of
	 * the p-values under their null hypotheses. They are rescaled over the
	 * non-missing family to have mean one, so multiplying every weight by the
	 * same positive constant does not change the result. Larger weights give a
	 * hypothesis greater priority.</p>
	 *
	 * @see <a href="https://doi.org/10.1093/biomet/93.3.509">Genovese,
	 * Roeder, and Wasserman (2006)</a>
	 */
	public static double[] adjustWeightedBenjaminiHochberg(
			double[] pValues, double[] weights) {
		ValidatedPValues validated = validatePValues(pValues);
		ValidatedPValues weightedLogs = weightedLogPValues(validated,
				weights, false);
		double[] logAdjusted = adjustLog(weightedLogs,
				Method.BENJAMINI_HOCHBERG, validated.values.length);
		double[] result = missingResult(pValues.length);
		for (int i = 0; i < result.length; i++) {
			if (!Double.isNaN(logAdjusted[i])) result[i] = Math.exp(logAdjusted[i]);
		}
		return result;
	}

	/**
	 * Computes weighted BH adjusted values directly from natural-log p-values.
	 * The output remains in natural-log form.
	 */
	public static double[] adjustLogWeightedBenjaminiHochberg(
			double[] logPValues, double[] weights) {
		ValidatedPValues validated = validateLogPValues(logPValues);
		return adjustLog(weightedLogPValues(validated, weights, true),
				Method.BENJAMINI_HOCHBERG, validated.values.length);
	}

	/**
	 * Computes weighted Benjamini-Yekutieli adjusted p-values.
	 *
	 * <p>This is weighted BH with the harmonic family-size correction and
	 * therefore retains FDR control under arbitrary dependence. Weight
	 * requirements and normalization are identical to weighted BH.</p>
	 */
	public static double[] adjustWeightedBenjaminiYekutieli(
			double[] pValues, double[] weights) {
		return adjustWeighted(pValues, weights,
				Method.BENJAMINI_YEKUTIELI);
	}

	/** Weighted BY adjustment for natural-log p-values. */
	public static double[] adjustLogWeightedBenjaminiYekutieli(
			double[] logPValues, double[] weights) {
		return adjustLogWeighted(logPValues, weights,
				Method.BENJAMINI_YEKUTIELI);
	}

	/** Computes weighted Bonferroni adjusted p-values. */
	public static double[] adjustWeightedBonferroni(
			double[] pValues, double[] weights) {
		return adjustWeighted(pValues, weights, Method.BONFERRONI);
	}

	/** Weighted Bonferroni adjustment for natural-log p-values. */
	public static double[] adjustLogWeightedBonferroni(
			double[] logPValues, double[] weights) {
		return adjustLogWeighted(logPValues, weights, Method.BONFERRONI);
	}

	/**
	 * Computes weighted Holm adjusted p-values.
	 *
	 * <p>Hypotheses are ordered by {@code p[i] / weight[i]}. At each step the
	 * multiplier is the sum of weights still under consideration. This strongly
	 * controls FWER and reduces exactly to ordinary Holm for equal weights.</p>
	 */
	public static double[] adjustWeightedHolm(
			double[] pValues, double[] weights) {
		return adjustWeightedHolm(validatePValues(pValues), weights, false);
	}

	/** Weighted Holm adjustment for natural-log p-values. */
	public static double[] adjustLogWeightedHolm(
			double[] logPValues, double[] weights) {
		return adjustWeightedHolm(validateLogPValues(logPValues), weights, true);
	}

	/** Returns weighted-BH rejection flags at the requested FDR level. */
	public static boolean[] rejectWeightedBenjaminiHochberg(
			double[] pValues, double[] weights, double level) {
		validateLevel(level);
		return rejectAdjusted(adjustWeightedBenjaminiHochberg(pValues, weights),
				level);
	}

	/** Returns weighted-BH rejection flags for natural-log p-values. */
	public static boolean[] rejectLogWeightedBenjaminiHochberg(
			double[] logPValues, double[] weights, double level) {
		validateLevel(level);
		double[] adjusted = adjustLogWeightedBenjaminiHochberg(logPValues,
				weights);
		double logLevel = level == 0.0 ? Double.NEGATIVE_INFINITY
				: Math.log(level);
		return rejectAdjusted(adjusted, logLevel);
	}

	/** Returns weighted-BY rejection flags. */
	public static boolean[] rejectWeightedBenjaminiYekutieli(
			double[] pValues, double[] weights, double level) {
		validateLevel(level);
		return rejectAdjusted(adjustWeightedBenjaminiYekutieli(pValues, weights),
				level);
	}

	/** Returns weighted-Bonferroni rejection flags. */
	public static boolean[] rejectWeightedBonferroni(
			double[] pValues, double[] weights, double level) {
		validateLevel(level);
		return rejectAdjusted(adjustWeightedBonferroni(pValues, weights), level);
	}

	/** Returns weighted-Holm rejection flags. */
	public static boolean[] rejectWeightedHolm(
			double[] pValues, double[] weights, double level) {
		validateLevel(level);
		return rejectAdjusted(adjustWeightedHolm(pValues, weights), level);
	}

	/** Returns log-scale weighted-BY rejection flags. */
	public static boolean[] rejectLogWeightedBenjaminiYekutieli(
			double[] logPValues, double[] weights, double level) {
		return rejectLogWeighted(adjustLogWeightedBenjaminiYekutieli(
				logPValues, weights), level);
	}

	/** Returns log-scale weighted-Bonferroni rejection flags. */
	public static boolean[] rejectLogWeightedBonferroni(
			double[] logPValues, double[] weights, double level) {
		return rejectLogWeighted(adjustLogWeightedBonferroni(logPValues,
				weights), level);
	}

	/** Returns log-scale weighted-Holm rejection flags. */
	public static boolean[] rejectLogWeightedHolm(
			double[] logPValues, double[] weights, double level) {
		return rejectLogWeighted(adjustLogWeightedHolm(logPValues, weights),
				level);
	}

	/** Returns rejection flags for natural-log p-values. */
	public static boolean[] rejectLog(double[] logPValues, double level,
			Method method) {
		return rejectLog(logPValues, level, method,
				validateLogPValues(logPValues).values.length);
	}

	/** Returns log-p rejection flags for a declared total family size. */
	public static boolean[] rejectLog(double[] logPValues, double level,
			Method method, int numberOfTests) {
		validateLevel(level);
		double[] adjusted = adjustLog(logPValues, method, numberOfTests);
		double logLevel = level == 0.0 ? Double.NEGATIVE_INFINITY
				: Math.log(level);
		boolean[] rejected = new boolean[logPValues.length];
		for (int i = 0; i < adjusted.length; i++)
			rejected[i] = !Double.isNaN(adjusted[i])
					&& adjusted[i] <= logLevel;
		return rejected;
	}

	/**
	 * Runs the Benjamini-Krieger-Yekutieli two-stage linear step-up test.
	 *
	 * <p>Unlike ordinary adjusted p-values, BKY depends on the requested FDR
	 * level, so the result reports the stage levels and rejection decisions
	 * rather than pretending to provide level-independent adjusted values. Its
	 * proven FDR guarantee is for independent test statistics.</p>
	 *
	 * @see <a href="https://doi.org/10.1093/biomet/93.3.491">Benjamini,
	 * Krieger, and Yekutieli (2006), Definition 6</a>
	 */
	public static AdaptiveFdrResult benjaminiKriegerYekutieli(
			double[] pValues, double level) {
		ValidatedPValues validated = validatePValues(pValues);
		return benjaminiKriegerYekutieli(validated, level,
				validated.values.length);
	}

	/**
	 * Runs the Gavrilov-Benjamini-Sarkar adaptive step-down FDR procedure.
	 *
	 * <p>Sorted hypotheses are rejected only while every p-value through rank
	 * {@code i} is at most {@code i*q/(m+1-i*(1-q))}. The proven finite-sample
	 * FDR guarantee is for independent test statistics.</p>
	 *
	 * @see <a href="https://doi.org/10.1214/07-AOS586">Gavrilov,
	 * Benjamini, and Sarkar (2009)</a>
	 */
	public static StepDownFdrResult gavrilovBenjaminiSarkar(
			double[] pValues, double level) {
		ValidatedPValues validated = validatePValues(pValues);
		return gavrilovBenjaminiSarkar(validated, level,
				validated.values.length);
	}

	/** Runs GBS for a declared total family size. */
	public static StepDownFdrResult gavrilovBenjaminiSarkar(
			double[] pValues, double level, int numberOfTests) {
		return gavrilovBenjaminiSarkar(validatePValues(pValues), level,
				numberOfTests);
	}

	/** Runs two-stage BKY for a declared total family size. */
	public static AdaptiveFdrResult benjaminiKriegerYekutieli(
			double[] pValues, double level, int numberOfTests) {
		return benjaminiKriegerYekutieli(validatePValues(pValues), level,
				numberOfTests);
	}

	/**
	 * Tests a right-censored family, where every unrecorded p-value is known
	 * to be greater than {@code censoringThreshold}.
	 *
	 * <p>The total family size must be known. Unrecorded values are completed
	 * conservatively with one. {@code areDecisionsExact()} is true when the
	 * censoring threshold is at least the largest possible rejection boundary;
	 * otherwise the reported rejections remain conservative, but additional
	 * discoveries cannot be ruled out without the censored values.</p>
	 */
	public static CensoredTestResult testRightCensored(
			double[] observedPValues, double censoringThreshold,
			int numberOfTests, double level, Method method) {
		validateLevel(censoringThreshold);
		validateLevel(level);
		ValidatedPValues validated = validatePValues(observedPValues);
		if (numberOfTests < validated.values.length)
			throw new IllegalArgumentException(
					"number of tests is smaller than observed p-values");
		for (double value : validated.values) {
			if (value > censoringThreshold)
				throw new IllegalArgumentException(
						"recorded p-values must not exceed the censoring threshold");
		}
		double[] adjusted = adjust(validated, method, numberOfTests);
		boolean[] rejected = new boolean[observedPValues.length];
		double threshold = Double.NaN;
		for (int i = 0; i < adjusted.length; i++) {
			rejected[i] = !Double.isNaN(adjusted[i]) && adjusted[i] <= level;
			if (rejected[i] && (Double.isNaN(threshold)
					|| observedPValues[i] > threshold))
				threshold = observedPValues[i];
		}
		boolean exact = censoringThreshold >= largestRejectionBoundary(
				level, method, numberOfTests);
		return new CensoredTestResult(adjusted, rejected,
				numberOfTests - validated.values.length, exact, threshold);
	}

	/** Returns one rejection flag per input, with missing values marked false. */
	public static boolean[] reject(double[] pValues, double level, Method method) {
		return reject(pValues, level, method,
				validatePValues(pValues).values.length);
	}

	/** Returns rejection flags for a declared total family size. */
	public static boolean[] reject(double[] pValues, double level, Method method,
			int numberOfTests) {
		validateLevel(level);
		double[] adjusted = adjust(pValues, method, numberOfTests);
		boolean[] rejected = new boolean[pValues.length];
		for (int i = 0; i < adjusted.length; i++) {
			rejected[i] = !Double.isNaN(adjusted[i]) && adjusted[i] <= level;
		}
		return rejected;
	}

	/** Returns the number of hypotheses rejected at the requested level. */
	public static int countRejected(double[] pValues, double level,
			Method method) {
		boolean[] rejected = reject(pValues, level, method);
		int count = 0;
		for (boolean value : rejected) if (value) count++;
		return count;
	}

	/** Returns the rejection count for a declared total family size. */
	public static int countRejected(double[] pValues, double level,
			Method method, int numberOfTests) {
		return countTrue(reject(pValues, level, method, numberOfTests));
	}

	/** Returns the rejection count for natural-log p-values. */
	public static int countRejectedLog(double[] logPValues, double level,
			Method method) {
		return countTrue(rejectLog(logPValues, level, method));
	}

	/** Returns the log-p rejection count for a declared total family size. */
	public static int countRejectedLog(double[] logPValues, double level,
			Method method, int numberOfTests) {
		return countTrue(rejectLog(logPValues, level, method, numberOfTests));
	}

	/**
	 * Returns the largest observed raw p-value rejected at the requested level.
	 *
	 * @return the raw cutoff, or {@code NaN} when no hypothesis is rejected
	 */
	public static double threshold(double[] pValues, double level, Method method) {
		return threshold(pValues, level, method,
				validatePValues(pValues).values.length);
	}

	/** Returns the raw rejection threshold for a declared total family size. */
	public static double threshold(double[] pValues, double level, Method method,
			int numberOfTests) {
		validateLevel(level);
		double[] adjusted = adjust(pValues, method, numberOfTests);
		double threshold = Double.NaN;
		for (int i = 0; i < pValues.length; i++) {
			if (!Double.isNaN(adjusted[i]) && adjusted[i] <= level
					&& (Double.isNaN(threshold) || pValues[i] > threshold)) {
				threshold = pValues[i];
			}
		}
		return threshold;
	}

	/** Returns the largest rejected natural-log p-value, or {@code NaN}. */
	public static double thresholdLog(double[] logPValues, double level,
			Method method) {
		return thresholdLog(logPValues, level, method,
				validateLogPValues(logPValues).values.length);
	}

	/** Returns the log rejection threshold for a declared total family size. */
	public static double thresholdLog(double[] logPValues, double level,
			Method method, int numberOfTests) {
		boolean[] rejected = rejectLog(logPValues, level, method, numberOfTests);
		double threshold = Double.NaN;
		for (int i = 0; i < logPValues.length; i++) {
			if (rejected[i] && (Double.isNaN(threshold)
					|| logPValues[i] > threshold)) threshold = logPValues[i];
		}
		return threshold;
	}

	/**
	 * Selects groups with BH on Simes p-values and tests selected groups with
	 * selection-adjusted BH.
	 *
	 * <p>Group labels may be any integers and need not be contiguous. All
	 * p-values must be observed. If {@code R} of {@code G} groups are selected,
	 * each selected group is tested at {@code withinGroupLevel * R / G}. The
	 * guarantee concerns the expected average FDR over selected families under
	 * the assumptions of Benjamini and Bogomolov (2014), rather than pooled FDR
	 * across every hypothesis.</p>
	 *
	 * @see <a href="https://doi.org/10.1111/rssb.12028">Benjamini and
	 * Bogomolov (2014)</a>
	 */
	public static GroupedFdrResult selectiveGroupedBenjaminiHochberg(
			double[] pValues, int[] groups, double groupLevel,
			double withinGroupLevel) {
		validateLevel(groupLevel);
		validateLevel(withinGroupLevel);
		ValidatedPValues validated = validatePValues(pValues);
		if (validated.values.length != pValues.length)
			throw new IllegalArgumentException(
					"grouped testing does not accept missing p-values");
		if (groups == null || groups.length != pValues.length)
			throw new IllegalArgumentException(
					"groups must have the same length as the p-values");
		TreeMap<Integer, Integer> counts = new TreeMap<Integer, Integer>();
		for (int group : groups) counts.put(group,
				counts.containsKey(group) ? counts.get(group) + 1 : 1);
		int groupCount = counts.size();
		int[] labels = new int[groupCount];
		double[] groupP = new double[groupCount];
		int next = 0;
		for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
			labels[next] = entry.getKey();
			double[] family = new double[entry.getValue()];
			int position = 0;
			for (int i = 0; i < groups.length; i++)
				if (groups[i] == entry.getKey()) family[position++] = pValues[i];
			Arrays.sort(family);
			double simes = 1.0;
			for (int rank = 0; rank < family.length; rank++)
				simes = Math.min(simes,
						family.length * family[rank] / (rank + 1.0));
			groupP[next++] = probability(simes);
		}
		boolean[] selected = reject(groupP, groupLevel,
				Method.BENJAMINI_HOCHBERG);
		int selectedCount = countTrue(selected);
		double familyLevel = groupCount == 0 ? 0.0
				: withinGroupLevel * selectedCount / groupCount;
		boolean[] rejected = new boolean[pValues.length];
		for (int groupIndex = 0; groupIndex < labels.length; groupIndex++) {
			if (!selected[groupIndex]) continue;
			int size = counts.get(labels[groupIndex]);
			double[] family = new double[size];
			int[] indices = new int[size];
			int position = 0;
			for (int i = 0; i < groups.length; i++) {
				if (groups[i] == labels[groupIndex]) {
					family[position] = pValues[i];
					indices[position++] = i;
				}
			}
			boolean[] familyRejected = reject(family, familyLevel,
					Method.BENJAMINI_HOCHBERG);
			for (int i = 0; i < size; i++)
				rejected[indices[i]] = familyRejected[i];
		}
		return new GroupedFdrResult(rejected, labels, groupP, selected,
				familyLevel);
	}

	/**
	 * Computes Storey q-values using the default smoothing-spline estimate of
	 * the true-null proportion.
	 */
	public static double[] qValues(double[] pValues) {
		ValidatedPValues validated = validatePValues(pValues);
		if (validated.values.length == 0) return missingResult(pValues.length);
		return qValues(pValues, estimateNullProportion(pValues));
	}

	/** Computes Storey q-values for a caller-supplied true-null proportion. */
	public static double[] qValues(double[] pValues, double nullProportion) {
		if (!Double.isFinite(nullProportion) || nullProportion < 0.0
				|| nullProportion > 1.0)
			throw new IllegalArgumentException("null proportion must be in [0, 1]");
		ValidatedPValues validated = validatePValues(pValues);
		double[] result = missingResult(pValues.length);
		if (validated.values.length == 0) return result;
		SortedPValues sorted = sort(validated);
		double last = 1.0;
		int count = sorted.values.length;
		for (int rank = count - 1; rank >= 0; rank--) {
			double candidate = nullProportion * count * sorted.values[rank]
					/ (rank + 1.0);
			last = Math.min(last, candidate);
			result[sorted.originalIndices[rank]] = probability(last);
		}
		return result;
	}

	/** Computes Storey q-values with caller-controlled spline settings. */
	public static double[] qValues(double[] pValues, double[] lambdas,
			double smoothingDegreesOfFreedom) {
		ValidatedPValues validated = validatePValues(pValues);
		if (validated.values.length == 0) return missingResult(pValues.length);
		return qValues(pValues, estimateNullProportion(pValues, lambdas,
				smoothingDegreesOfFreedom));
	}

	/** Computes q-values using the quantile-based estimate of pi-zero. */
	public static double[] qValuesQuantile(double[] pValues, double quantile) {
		ValidatedPValues validated = validatePValues(pValues);
		if (validated.values.length == 0) return missingResult(pValues.length);
		return qValues(pValues,
				estimateNullProportionQuantile(pValues, DEFAULT_LAMBDAS, quantile));
	}

	/** Computes quantile-estimated q-values with a caller-supplied lambda grid. */
	public static double[] qValuesQuantile(double[] pValues, double[] lambdas,
			double quantile) {
		ValidatedPValues validated = validatePValues(pValues);
		if (validated.values.length == 0) return missingResult(pValues.length);
		return qValues(pValues,
				estimateNullProportionQuantile(pValues, lambdas, quantile));
	}

	/** Computes q-values using the default 0.10 quantile pi-zero estimate. */
	public static double[] qValuesQuantile(double[] pValues) {
		return qValuesQuantile(pValues, DEFAULT_PI0_QUANTILE);
	}

	/** Estimates pi-zero with the default lambda grid and three spline degrees of freedom. */
	public static double estimateNullProportion(double[] pValues) {
		return estimateNullProportion(pValues, DEFAULT_LAMBDAS,
				DEFAULT_SMOOTHING_DF);
	}

	/**
	 * Estimates the true-null proportion by smoothing the Storey estimates over
	 * lambda and predicting at the largest lambda.
	 */
	public static double estimateNullProportion(double[] pValues,
			double[] lambdas, double smoothingDegreesOfFreedom) {
		ValidatedPValues validated = validatePValues(pValues);
		validateLambdas(lambdas, 4);
		if (validated.values.length == 0)
			throw new IllegalArgumentException("at least one p-value is required");
		if (!Double.isFinite(smoothingDegreesOfFreedom)
				|| smoothingDegreesOfFreedom <= 1.0
				|| smoothingDegreesOfFreedom > lambdas.length)
			throw new IllegalArgumentException("invalid smoothing degrees of freedom");
		double[] estimates = nullProportions(validated.values, lambdas);
		SmoothSplineResult fit = SmoothSpline.fitDFMatch(lambdas.clone(),
				estimates, smoothingDegreesOfFreedom);
		if (fit.mHasFactorizationProblems)
			throw new ArithmeticException("null-proportion spline factorization failed");
		double estimate = SmoothSpline.predict(fit,
				lambdas[lambdas.length - 1], 0);
		if (!Double.isFinite(estimate))
			throw new ArithmeticException("null-proportion spline returned a non-finite value");
		estimate = probability(estimate);
		if (!(estimate > 0.0))
			throw new ArithmeticException(
					"estimated null proportion is zero; inspect the p-value distribution or lambda grid");
		return estimate;
	}

	/** Estimates pi-zero as a quantile of the estimates across lambda values. */
	public static double estimateNullProportionQuantile(double[] pValues,
			double[] lambdas, double quantile) {
		ValidatedPValues validated = validatePValues(pValues);
		validateLambdas(lambdas, 1);
		validateLevel(quantile);
		if (validated.values.length == 0)
			throw new IllegalArgumentException("at least one p-value is required");
		double[] estimates = nullProportions(validated.values, lambdas);
		Arrays.sort(estimates);
		double estimate = probability(VectorMath.quantile(estimates, quantile));
		if (!(estimate > 0.0))
			throw new ArithmeticException(
					"estimated null proportion is zero; inspect the p-value distribution or lambda grid");
		return estimate;
	}

	/** Uses the default lambda grid for the quantile pi-zero estimator. */
	public static double estimateNullProportionQuantile(double[] pValues,
			double quantile) {
		return estimateNullProportionQuantile(pValues, DEFAULT_LAMBDAS, quantile);
	}

	/**
	 * Estimates FDR at a raw p-value cutoff using the default spline pi-zero estimate.
	 */
	public static double estimatedFalseDiscoveryRate(double[] pValues,
			double rawThreshold) {
		return estimatedFalseDiscoveryRate(pValues, rawThreshold,
				estimateNullProportion(pValues));
	}

	/** Estimates FDR at a raw cutoff for a caller-supplied pi-zero. */
	public static double estimatedFalseDiscoveryRate(double[] pValues,
			double rawThreshold, double nullProportion) {
		validateLevel(rawThreshold);
		if (!Double.isFinite(nullProportion) || nullProportion < 0.0
				|| nullProportion > 1.0)
			throw new IllegalArgumentException("null proportion must be in [0, 1]");
		ValidatedPValues validated = validatePValues(pValues);
		int discoveries = 0;
		for (double value : validated.values) if (value <= rawThreshold) discoveries++;
		if (discoveries == 0) return 0.0;
		return probability(nullProportion * validated.values.length * rawThreshold
				/ discoveries);
	}

	/** Returns a defensive copy of the default q-value lambda grid. */
	public static double[] defaultLambdas() { return DEFAULT_LAMBDAS.clone(); }

	private static double[] adjust(ValidatedPValues validated, Method method,
			int numberOfTests) {
		if (method == null) throw new IllegalArgumentException("method must not be null");
		if (numberOfTests < validated.values.length)
			throw new IllegalArgumentException("number of tests is smaller than observed p-values");
		double[] result = missingResult(validated.originalLength);
		if (validated.values.length == 0) return result;
		if (method == Method.NONE) {
			for (int i = 0; i < validated.values.length; i++)
				result[validated.originalIndices[i]] = validated.values[i];
			return result;
		}
		if (numberOfTests == 0)
			throw new IllegalArgumentException("number of tests must be positive");
		SortedPValues sorted = sort(validated);
		switch (method) {
			case BONFERRONI:
				for (int rank = 0; rank < sorted.values.length; rank++)
					setSorted(result, sorted, rank,
							numberOfTests * sorted.values[rank]);
				break;
			case HOLM:
				adjustHolm(result, sorted, numberOfTests, false);
				break;
			case HOLM_SIDAK:
				adjustHolm(result, sorted, numberOfTests, true);
				break;
			case HOCHBERG:
				adjustReverse(result, sorted, numberOfTests, 1.0);
				break;
			case BENJAMINI_HOCHBERG:
				adjustReverse(result, sorted, numberOfTests, 0.0);
				break;
			case BENJAMINI_YEKUTIELI:
				adjustReverse(result, sorted, numberOfTests,
						harmonic(numberOfTests));
				break;
			case SIDAK:
				for (int rank = 0; rank < sorted.values.length; rank++)
					setSorted(result, sorted, rank,
							sidak(sorted.values[rank], numberOfTests));
				break;
			case HOMMEL:
				adjustHommel(result, sorted, numberOfTests);
				break;
			default:
				throw new AssertionError(method);
		}
		return result;
	}

	private static double[] adjustLog(ValidatedPValues validated, Method method,
			int numberOfTests) {
		if (method == null) throw new IllegalArgumentException("method must not be null");
		if (numberOfTests < validated.values.length)
			throw new IllegalArgumentException("number of tests is smaller than observed p-values");
		double[] result = missingResult(validated.originalLength);
		if (validated.values.length == 0) return result;
		if (method == Method.NONE) {
			for (int i = 0; i < validated.values.length; i++)
				result[validated.originalIndices[i]] = validated.values[i];
			return result;
		}
		if (numberOfTests == 0)
			throw new IllegalArgumentException("number of tests must be positive");
		SortedPValues sorted = sort(validated);
		switch (method) {
			case BONFERRONI:
				for (int rank = 0; rank < sorted.values.length; rank++)
					setSortedLog(result, sorted, rank,
							sorted.values[rank] + Math.log(numberOfTests));
				break;
			case HOLM:
				adjustHolmLog(result, sorted, numberOfTests, false);
				break;
			case HOLM_SIDAK:
				adjustHolmLog(result, sorted, numberOfTests, true);
				break;
			case HOCHBERG:
				adjustReverseLog(result, sorted, numberOfTests, 1.0);
				break;
			case BENJAMINI_HOCHBERG:
				adjustReverseLog(result, sorted, numberOfTests, 0.0);
				break;
			case BENJAMINI_YEKUTIELI:
				adjustReverseLog(result, sorted, numberOfTests,
						harmonic(numberOfTests));
				break;
			case SIDAK:
				for (int rank = 0; rank < sorted.values.length; rank++)
					setSortedLog(result, sorted, rank,
							logSidak(sorted.values[rank], numberOfTests));
				break;
			case HOMMEL:
				adjustHommelLog(result, sorted, numberOfTests);
				break;
			default:
				throw new AssertionError(method);
		}
		return result;
	}

	private static AdaptiveFdrResult benjaminiKriegerYekutieli(
			ValidatedPValues validated, double level, int numberOfTests) {
		validateLevel(level);
		if (numberOfTests < validated.values.length)
			throw new IllegalArgumentException(
					"number of tests is smaller than observed p-values");
		if (numberOfTests == 0) {
			double stageOneLevel = level / (1.0 + level);
			return new AdaptiveFdrResult(new boolean[validated.originalLength],
					0, 0, 0, stageOneLevel, stageOneLevel, Double.NaN);
		}
		double stageOneLevel = level / (1.0 + level);
		boolean[] first = rejectAdjusted(adjust(validated,
				Method.BENJAMINI_HOCHBERG, numberOfTests), stageOneLevel);
		int stageOneRejections = countTrue(first);
		if (stageOneRejections == 0 || stageOneRejections == numberOfTests) {
			return new AdaptiveFdrResult(first, stageOneRejections,
					stageOneRejections,
					numberOfTests - stageOneRejections, stageOneLevel,
					stageOneLevel, rawThreshold(validated, first));
		}
		int estimatedTrueNulls = numberOfTests - stageOneRejections;
		double finalLevel = stageOneLevel * numberOfTests / estimatedTrueNulls;
		boolean[] finalRejected = rejectAdjusted(adjust(validated,
				Method.BENJAMINI_HOCHBERG, numberOfTests), finalLevel);
		int finalRejections = finalLevel >= 1.0 ? numberOfTests
				: countTrue(finalRejected);
		return new AdaptiveFdrResult(finalRejected, finalRejections,
				stageOneRejections,
				estimatedTrueNulls, stageOneLevel, finalLevel,
				rawThreshold(validated, finalRejected));
	}

	private static StepDownFdrResult gavrilovBenjaminiSarkar(
			ValidatedPValues validated, double level, int numberOfTests) {
		validateLevel(level);
		if (numberOfTests < validated.values.length)
			throw new IllegalArgumentException(
					"number of tests is smaller than observed p-values");
		boolean[] rejected = new boolean[validated.originalLength];
		if (numberOfTests == 0 || validated.values.length == 0)
			return new StepDownFdrResult(rejected, 0, Double.NaN, Double.NaN);
		SortedPValues sorted = sort(validated);
		int rejectedCount = 0;
		double criticalValue = Double.NaN;
		for (int rank = 0; rank < sorted.values.length; rank++) {
			double candidate = gbsCriticalValue(rank + 1, numberOfTests, level);
			if (sorted.values[rank] > candidate) break;
			rejected[sorted.originalIndices[rank]] = true;
			rejectedCount++;
			criticalValue = candidate;
		}
		double threshold = rejectedCount == 0 ? Double.NaN
				: sorted.values[rejectedCount - 1];
		return new StepDownFdrResult(rejected, rejectedCount, threshold,
				criticalValue);
	}

	private static double gbsCriticalValue(int rank, int numberOfTests,
			double level) {
		return rank * level
				/ (numberOfTests + 1.0 - rank * (1.0 - level));
	}

	private static boolean[] rejectAdjusted(double[] adjusted, double level) {
		boolean[] rejected = new boolean[adjusted.length];
		for (int i = 0; i < adjusted.length; i++)
			rejected[i] = !Double.isNaN(adjusted[i]) && adjusted[i] <= level;
		return rejected;
	}

	private static boolean[] rejectLogWeighted(double[] adjusted,
			double level) {
		validateLevel(level);
		double logLevel = level == 0.0 ? Double.NEGATIVE_INFINITY
				: Math.log(level);
		return rejectAdjusted(adjusted, logLevel);
	}

	private static double[] adjustWeighted(double[] pValues, double[] weights,
			Method method) {
		ValidatedPValues validated = validatePValues(pValues);
		double[] logAdjusted = adjustLog(weightedLogPValues(validated, weights,
				false), method, validated.values.length);
		double[] result = missingResult(pValues.length);
		for (int i = 0; i < result.length; i++)
			if (!Double.isNaN(logAdjusted[i])) result[i] = Math.exp(logAdjusted[i]);
		return result;
	}

	private static double[] adjustLogWeighted(double[] logPValues,
			double[] weights, Method method) {
		ValidatedPValues validated = validateLogPValues(logPValues);
		return adjustLog(weightedLogPValues(validated, weights, true), method,
				validated.values.length);
	}

	private static double[] adjustWeightedHolm(ValidatedPValues validated,
			double[] weights, boolean valuesAreLogs) {
		double[] normalized = normalizedWeights(validated, weights);
		ValidatedPValues weighted = weightedLogPValues(validated, weights,
				valuesAreLogs);
		SortedPValues sorted = sort(weighted);
		double[] result = missingResult(validated.originalLength);
		if (sorted.values.length == 0) return result;
		double remainingWeight = validated.values.length;
		double last = Double.NEGATIVE_INFINITY;
		for (int rank = 0; rank < sorted.values.length; rank++) {
			double candidate = sorted.values[rank] + Math.log(remainingWeight);
			last = Math.max(last, candidate);
			setSortedLog(result, sorted, rank, last);
			remainingWeight -= normalized[sorted.originalIndices[rank]];
		}
		if (!valuesAreLogs) {
			for (int i = 0; i < result.length; i++)
				if (!Double.isNaN(result[i])) result[i] = Math.exp(result[i]);
		}
		return result;
	}

	private static double[] normalizedWeights(ValidatedPValues validated,
			double[] weights) {
		if (weights == null || weights.length != validated.originalLength)
			throw new IllegalArgumentException(
					"weights must have the same length as the p-values");
		for (double weight : weights) {
			if (!Double.isFinite(weight) || weight <= 0.0)
				throw new IllegalArgumentException(
						"weights must be finite and strictly positive");
		}
		double[] normalized = new double[weights.length];
		if (validated.values.length == 0) return normalized;
		double maximum = 0.0;
		for (int index : validated.originalIndices)
			maximum = Math.max(maximum, weights[index]);
		double scaledSum = 0.0;
		for (int index : validated.originalIndices)
			scaledSum += weights[index] / maximum;
		double scaledMean = scaledSum / validated.values.length;
		for (int index : validated.originalIndices)
			normalized[index] = (weights[index] / maximum) / scaledMean;
		return normalized;
	}

	private static ValidatedPValues weightedLogPValues(
			ValidatedPValues validated, double[] weights,
			boolean valuesAreLogs) {
		double[] normalized = normalizedWeights(validated, weights);
		if (validated.values.length == 0)
			return new ValidatedPValues(validated.originalLength,
					new double[0], new int[0]);
		double[] values = new double[validated.values.length];
		for (int i = 0; i < values.length; i++) {
			double logPValue = valuesAreLogs ? validated.values[i]
					: validated.values[i] == 0.0
							? Double.NEGATIVE_INFINITY
							: Math.log(validated.values[i]);
			double logNormalizedWeight = Math.log(
					normalized[validated.originalIndices[i]]);
			values[i] = logPValue - logNormalizedWeight;
		}
		return new ValidatedPValues(validated.originalLength, values,
				validated.originalIndices.clone());
	}

	private static int countTrue(boolean[] values) {
		int count = 0;
		for (boolean value : values) if (value) count++;
		return count;
	}

	private static double rawThreshold(ValidatedPValues validated,
			boolean[] rejected) {
		double threshold = Double.NaN;
		for (int i = 0; i < validated.values.length; i++) {
			int original = validated.originalIndices[i];
			if (rejected[original] && (Double.isNaN(threshold)
					|| validated.values[i] > threshold))
				threshold = validated.values[i];
		}
		return threshold;
	}

	private static double largestRejectionBoundary(double level,
			Method method, int numberOfTests) {
		if (method == null) throw new IllegalArgumentException("method must not be null");
		if (numberOfTests <= 0) return 0.0;
		switch (method) {
			case BONFERRONI:
				return level / numberOfTests;
			case SIDAK:
				return -Math.expm1(Math.log1p(-level) / numberOfTests);
			case BENJAMINI_YEKUTIELI:
				return level / harmonic(numberOfTests);
			default:
				return level;
		}
	}

	private static void adjustHolm(double[] result, SortedPValues sorted,
			int numberOfTests, boolean sidak) {
		double last = 0.0;
		for (int rank = 0; rank < sorted.values.length; rank++) {
			int remaining = numberOfTests - rank;
			double candidate = sidak
					? sidak(sorted.values[rank], remaining)
					: remaining * sorted.values[rank];
			last = Math.max(last, candidate);
			setSorted(result, sorted, rank, last);
		}
	}

	private static void adjustHolmLog(double[] result, SortedPValues sorted,
			int numberOfTests, boolean sidak) {
		double last = Double.NEGATIVE_INFINITY;
		for (int rank = 0; rank < sorted.values.length; rank++) {
			int remaining = numberOfTests - rank;
			double candidate = sidak
					? logSidak(sorted.values[rank], remaining)
					: sorted.values[rank] + Math.log(remaining);
			last = Math.max(last, candidate);
			setSortedLog(result, sorted, rank, last);
		}
	}

	private static void adjustReverse(double[] result, SortedPValues sorted,
			int numberOfTests, double multiplier) {
		double last = 1.0;
		for (int rank = sorted.values.length - 1; rank >= 0; rank--) {
			double candidate;
			if (multiplier == 1.0) {
				candidate = (numberOfTests - rank) * sorted.values[rank];
			} else {
				double correction = multiplier == 0.0 ? 1.0 : multiplier;
				candidate = correction * numberOfTests * sorted.values[rank]
						/ (rank + 1.0);
			}
			last = Math.min(last, candidate);
			setSorted(result, sorted, rank, last);
		}
	}

	private static void adjustReverseLog(double[] result,
			SortedPValues sorted, int numberOfTests, double multiplier) {
		double last = 0.0;
		for (int rank = sorted.values.length - 1; rank >= 0; rank--) {
			double candidate;
			if (multiplier == 1.0) {
				candidate = sorted.values[rank]
						+ Math.log(numberOfTests - rank);
			} else {
				double correction = multiplier == 0.0 ? 1.0 : multiplier;
				candidate = sorted.values[rank] + Math.log(correction)
						+ Math.log(numberOfTests) - Math.log(rank + 1.0);
			}
			last = Math.min(last, candidate);
			setSortedLog(result, sorted, rank, last);
		}
	}

	private static void adjustHommel(double[] result, SortedPValues sorted,
			int numberOfTests) {
		if (numberOfTests <= 2) {
			adjustReverse(result, sorted, numberOfTests, 1.0);
			return;
		}
		double[] p = new double[numberOfTests];
		System.arraycopy(sorted.values, 0, p, 0, sorted.values.length);
		Arrays.fill(p, sorted.values.length, numberOfTests, 1.0);
		double initial = Double.POSITIVE_INFINITY;
		for (int i = 0; i < numberOfTests; i++)
			initial = Math.min(initial, numberOfTests * p[i] / (i + 1.0));
		double[] q = new double[numberOfTests];
		double[] adjusted = new double[numberOfTests];
		Arrays.fill(q, initial);
		Arrays.fill(adjusted, initial);
		for (int j = numberOfTests - 1; j >= 2; j--) {
			int headLength = numberOfTests - j + 1;
			double q1 = Double.POSITIVE_INFINITY;
			for (int denominator = 2; denominator <= j; denominator++) {
				int index = headLength + denominator - 2;
				q1 = Math.min(q1, j * p[index] / denominator);
			}
			for (int i = 0; i < headLength; i++)
				q[i] = Math.min(j * p[i], q1);
			double tailValue = q[headLength - 1];
			for (int i = headLength; i < numberOfTests; i++) q[i] = tailValue;
			for (int i = 0; i < numberOfTests; i++)
				adjusted[i] = Math.max(adjusted[i], q[i]);
		}
		for (int rank = 0; rank < sorted.values.length; rank++)
			setSorted(result, sorted, rank,
					Math.max(adjusted[rank], sorted.values[rank]));
	}

	private static void adjustHommelLog(double[] result,
			SortedPValues sorted, int numberOfTests) {
		if (numberOfTests <= 2) {
			adjustReverseLog(result, sorted, numberOfTests, 1.0);
			return;
		}
		double[] p = new double[numberOfTests];
		System.arraycopy(sorted.values, 0, p, 0, sorted.values.length);
		Arrays.fill(p, sorted.values.length, numberOfTests, 0.0);
		double initial = Double.POSITIVE_INFINITY;
		for (int i = 0; i < numberOfTests; i++)
			initial = Math.min(initial, p[i] + Math.log(numberOfTests)
					- Math.log(i + 1.0));
		double[] q = new double[numberOfTests];
		double[] adjusted = new double[numberOfTests];
		Arrays.fill(q, initial);
		Arrays.fill(adjusted, initial);
		for (int j = numberOfTests - 1; j >= 2; j--) {
			int headLength = numberOfTests - j + 1;
			double q1 = Double.POSITIVE_INFINITY;
			for (int denominator = 2; denominator <= j; denominator++) {
				int index = headLength + denominator - 2;
				q1 = Math.min(q1, p[index] + Math.log(j)
						- Math.log(denominator));
			}
			for (int i = 0; i < headLength; i++)
				q[i] = Math.min(p[i] + Math.log(j), q1);
			double tailValue = q[headLength - 1];
			for (int i = headLength; i < numberOfTests; i++) q[i] = tailValue;
			for (int i = 0; i < numberOfTests; i++)
				adjusted[i] = Math.max(adjusted[i], q[i]);
		}
		for (int rank = 0; rank < sorted.values.length; rank++)
			setSortedLog(result, sorted, rank,
					Math.max(adjusted[rank], sorted.values[rank]));
	}

	private static void setSorted(double[] result, SortedPValues sorted,
			int rank, double value) {
		result[sorted.originalIndices[rank]] = probability(value);
	}

	private static void setSortedLog(double[] result, SortedPValues sorted,
			int rank, double value) {
		result[sorted.originalIndices[rank]] = Math.min(0.0, value);
	}

	private static double sidak(double pValue, int tests) {
		return -Math.expm1(tests * Math.log1p(-pValue));
	}

	private static double logSidak(double logPValue, int tests) {
		return logOneMinusExp(tests * logOneMinusExp(logPValue));
	}

	/** Computes log(1-exp(value)) accurately for value at most zero. */
	private static double logOneMinusExp(double value) {
		if (value == Double.NEGATIVE_INFINITY) return 0.0;
		if (value == 0.0) return Double.NEGATIVE_INFINITY;
		if (value < -Math.log(2.0)) return Math.log1p(-Math.exp(value));
		return Math.log(-Math.expm1(value));
	}

	private static double harmonic(int count) {
		double result = 0.0;
		for (int i = 1; i <= count; i++) result += 1.0 / i;
		return result;
	}

	private static double[] nullProportions(double[] pValues, double[] lambdas) {
		double[] estimates = new double[lambdas.length];
		for (int i = 0; i < lambdas.length; i++) {
			int above = 0;
			for (double pValue : pValues) if (pValue >= lambdas[i]) above++;
			estimates[i] = probability(above
					/ (pValues.length * (1.0 - lambdas[i])));
		}
		return estimates;
	}

	private static void validateLambdas(double[] lambdas, int minimum) {
		if (lambdas == null || lambdas.length < minimum)
			throw new IllegalArgumentException("too few lambda values");
		double previous = -1.0;
		for (double lambda : lambdas) {
			if (!Double.isFinite(lambda) || lambda < 0.0 || lambda >= 1.0
					|| lambda <= previous)
				throw new IllegalArgumentException(
						"lambdas must be finite, strictly increasing, and in [0, 1)");
			previous = lambda;
		}
	}

	private static void validateLevel(double level) {
		if (!Double.isFinite(level) || level < 0.0 || level > 1.0)
			throw new IllegalArgumentException("level must be in [0, 1]");
	}

	private static ValidatedPValues validatePValues(double[] pValues) {
		if (pValues == null) throw new IllegalArgumentException("p-values must not be null");
		int count = 0;
		for (double value : pValues) {
			if (Double.isNaN(value)) continue;
			if (!Double.isFinite(value) || value < 0.0 || value > 1.0)
				throw new IllegalArgumentException(
						"p-values must be NaN or finite values in [0, 1]");
			count++;
		}
		double[] values = new double[count];
		int[] indices = new int[count];
		int next = 0;
		for (int i = 0; i < pValues.length; i++) {
			if (!Double.isNaN(pValues[i])) {
				values[next] = pValues[i];
				indices[next] = i;
				next++;
			}
		}
		return new ValidatedPValues(pValues.length, values, indices);
	}

	private static ValidatedPValues validateLogPValues(double[] logPValues) {
		if (logPValues == null)
			throw new IllegalArgumentException("log p-values must not be null");
		int count = 0;
		for (double value : logPValues) {
			if (Double.isNaN(value)) continue;
			if (value == Double.POSITIVE_INFINITY || value > 0.0)
				throw new IllegalArgumentException(
						"log p-values must be NaN or values in [-infinity, 0]");
			count++;
		}
		double[] values = new double[count];
		int[] indices = new int[count];
		int next = 0;
		for (int i = 0; i < logPValues.length; i++) {
			if (!Double.isNaN(logPValues[i])) {
				values[next] = logPValues[i];
				indices[next] = i;
				next++;
			}
		}
		return new ValidatedPValues(logPValues.length, values, indices);
	}

	private static SortedPValues sort(ValidatedPValues validated) {
		Integer[] order = new Integer[validated.values.length];
		for (int i = 0; i < order.length; i++) order[i] = i;
		Arrays.sort(order, new Comparator<Integer>() {
			@Override public int compare(Integer first, Integer second) {
				return Double.compare(validated.values[first], validated.values[second]);
			}
		});
		double[] values = new double[order.length];
		int[] originalIndices = new int[order.length];
		for (int rank = 0; rank < order.length; rank++) {
			int source = order[rank];
			values[rank] = validated.values[source];
			originalIndices[rank] = validated.originalIndices[source];
		}
		return new SortedPValues(values, originalIndices);
	}

	private static double[] missingResult(int length) {
		double[] result = new double[length];
		Arrays.fill(result, Double.NaN);
		return result;
	}

	private static double probability(double value) {
		return Math.max(0.0, Math.min(1.0, value));
	}

	private static final class ValidatedPValues {
		final int originalLength;
		final double[] values;
		final int[] originalIndices;
		ValidatedPValues(int originalLength, double[] values,
				int[] originalIndices) {
			this.originalLength = originalLength;
			this.values = values;
			this.originalIndices = originalIndices;
		}
	}

	private static final class SortedPValues {
		final double[] values;
		final int[] originalIndices;
		SortedPValues(double[] values, int[] originalIndices) {
			this.values = values;
			this.originalIndices = originalIndices;
		}
	}
}
