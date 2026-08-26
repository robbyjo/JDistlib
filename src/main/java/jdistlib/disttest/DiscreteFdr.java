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
import java.util.TreeSet;

/** FDR procedures that exploit known heterogeneous discrete null CDFs. */
public final class DiscreteFdr {
	/** Rejection decisions and level-dependent DBH critical values. */
	public static final class Result {
		private final boolean[] rejected;
		private final double[] criticalValues;
		private final double threshold;

		private Result(boolean[] rejected, double[] criticalValues,
				double threshold) {
			this.rejected = rejected;
			this.criticalValues = criticalValues;
			this.threshold = threshold;
		}

		public boolean[] getRejected() { return rejected.clone(); }
		public int getRejectedCount() {
			int count = 0;
			for (boolean value : rejected) if (value) count++;
			return count;
		}
		/** Returns tau_1 through tau_m. */
		public double[] getCriticalValues() { return criticalValues.clone(); }
		/** Returns the largest rejected observed p-value, or NaN. */
		public double getThreshold() { return threshold; }
	}

	private DiscreteFdr() {}

	/**
	 * Runs the DBH step-down procedure of Döhler, Durand, and Roquain.
	 * Its finite-sample FDR guarantee requires independent p-values.
	 */
	public static Result dbhStepDown(double[] pValues,
			DiscretePValueDistribution[] nullDistributions, double level) {
		Inputs inputs = validate(pValues, nullDistributions, level);
		double[] critical = new double[inputs.count];
		for (int rank = 1; rank <= inputs.count; rank++) {
			double boundary = level * rank / inputs.count;
			for (double point : inputs.jointSupport) {
				if (sdTransform(point, nullDistributions) <= boundary)
					critical[rank - 1] = point;
			}
		}
		return decide(pValues, critical, false);
	}

	/**
	 * Runs the DBH step-up procedure of Döhler, Durand, and Roquain.
	 * Its finite-sample FDR guarantee requires independent p-values.
	 */
	public static Result dbhStepUp(double[] pValues,
			DiscretePValueDistribution[] nullDistributions, double level) {
		Inputs inputs = validate(pValues, nullDistributions, level);
		double tauM = 0.0;
		for (double point : inputs.jointSupport)
			if (sdTransform(point, nullDistributions) <= level) tauM = point;
		double[] denominators = new double[inputs.count];
		for (int i = 0; i < inputs.count; i++)
			denominators[i] = 1.0 - nullDistributions[i].cdf(tauM);
		double[] critical = new double[inputs.count];
		critical[inputs.count - 1] = tauM;
		for (int rank = 1; rank < inputs.count; rank++) {
			double boundary = level * rank / inputs.count;
			for (double point : inputs.jointSupport) {
				if (point > tauM) break;
				if (suTransform(point, nullDistributions, denominators) <= boundary)
					critical[rank - 1] = point;
			}
		}
		return decide(pValues, critical, true);
	}

	private static double sdTransform(double point,
			DiscretePValueDistribution[] distributions) {
		double sum = 0.0;
		for (DiscretePValueDistribution distribution : distributions) {
			double value = distribution.cdf(point);
			if (value >= 1.0) return Double.POSITIVE_INFINITY;
			sum += value / (1.0 - value);
		}
		return sum / distributions.length;
	}

	private static double suTransform(double point,
			DiscretePValueDistribution[] distributions, double[] denominators) {
		double sum = 0.0;
		for (int i = 0; i < distributions.length; i++) {
			double numerator = distributions[i].cdf(point);
			if (denominators[i] == 0.0) {
				if (numerator > 0.0) return Double.POSITIVE_INFINITY;
			} else sum += numerator / denominators[i];
		}
		return sum / distributions.length;
	}

	private static Result decide(double[] pValues, double[] critical,
			boolean stepUp) {
		Integer[] order = new Integer[pValues.length];
		for (int i = 0; i < order.length; i++) order[i] = i;
		Arrays.sort(order, new Comparator<Integer>() {
			@Override public int compare(Integer left, Integer right) {
				int comparison = Double.compare(pValues[left], pValues[right]);
				return comparison != 0 ? comparison : Integer.compare(left, right);
			}
		});
		int count = 0;
		if (stepUp) {
			for (int rank = 0; rank < order.length; rank++)
				if (pValues[order[rank]] <= critical[rank]) count = rank + 1;
		} else {
			while (count < order.length
					&& pValues[order[count]] <= critical[count]) count++;
		}
		boolean[] rejected = new boolean[pValues.length];
		for (int rank = 0; rank < count; rank++) rejected[order[rank]] = true;
		double threshold = count == 0 ? Double.NaN : pValues[order[count - 1]];
		return new Result(rejected, critical, threshold);
	}

	private static Inputs validate(double[] pValues,
			DiscretePValueDistribution[] distributions, double level) {
		if (pValues == null || distributions == null || pValues.length == 0
				|| pValues.length != distributions.length)
			throw new IllegalArgumentException(
					"p-values and null distributions must be nonempty and have equal length");
		if (!Double.isFinite(level) || level <= 0.0 || level > 1.0)
			throw new IllegalArgumentException("level must be in (0, 1]");
		TreeSet<Double> support = new TreeSet<Double>();
		support.add(0.0);
		for (int i = 0; i < pValues.length; i++) {
			if (!Double.isFinite(pValues[i]) || pValues[i] < 0.0
					|| pValues[i] > 1.0)
				throw new IllegalArgumentException(
						"p-values must be finite probabilities");
			if (distributions[i] == null)
				throw new IllegalArgumentException(
						"null distributions must not contain null");
			for (double point : distributions[i].getSupport()) support.add(point);
		}
		double[] joint = new double[support.size()];
		int next = 0;
		for (double point : support) joint[next++] = point;
		return new Inputs(pValues.length, joint);
	}

	private static final class Inputs {
		private final int count;
		private final double[] jointSupport;
		private Inputs(int count, double[] jointSupport) {
			this.count = count;
			this.jointSupport = jointSupport;
		}
	}
}
