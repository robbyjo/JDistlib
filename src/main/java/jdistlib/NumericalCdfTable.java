/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.List;

/**
 * Reusable monotone CDF approximation built from directly integrated values.
 * It uses a transformed domain and shape-preserving cubic interpolation.
 */
public final class NumericalCdfTable {
	private final double lower;
	private final double upper;
	private final double[] unit;
	private final double[] x;
	private final double[] probability;
	private final double[] slope;
	private final double maximumValidationError;
	private final boolean saturated;

	private NumericalCdfTable(double lower, double upper, List<Node> nodes,
			double maximumValidationError, boolean saturated) {
		this.lower = lower;
		this.upper = upper;
		this.maximumValidationError = maximumValidationError;
		this.saturated = saturated;
		unit = new double[nodes.size()];
		x = new double[nodes.size()];
		probability = new double[nodes.size()];
		for (int i = 0; i < nodes.size(); i++) {
			unit[i] = nodes.get(i).unit;
			x[i] = nodes.get(i).x;
			probability[i] = Math.max(i == 0 ? 0.0 : probability[i - 1],
					Math.min(1.0, nodes.get(i).probability));
		}
		probability[0] = 0.0;
		probability[probability.length - 1] = 1.0;
		slope = monotoneSlopes(unit, probability);
	}

	static NumericalCdfTable build(NumericalContinuousDistribution distribution,
			CdfTableOptions options) {
		List<Node> nodes = new ArrayList<Node>();
		int intervals = options.getInitialIntervals();
		for (int i = 0; i <= intervals; i++) {
			double u = i / (double) intervals;
			double x = map(u, distribution.getLowerBound(),
					distribution.getUpperBound());
			double p = i == 0 ? 0.0 : (i == intervals ? 1.0
					: distribution.cumulativeDirect(x, true, false));
			nodes.add(new Node(u, x, p));
		}

		double maxError = 0.0;
		boolean saturated = false;
		for (int pass = 0; pass < options.getRefinementPasses(); pass++) {
			double passMaximumError = 0.0;
			double[] u = values(nodes, 0);
			double[] p = values(nodes, 1);
			double[] slopes = monotoneSlopes(u, p);
			List<Node> refined = new ArrayList<Node>(nodes.size() * 2);
			refined.add(nodes.get(0));
			boolean added = false;
			for (int i = 0; i + 1 < nodes.size(); i++) {
				Node left = nodes.get(i);
				Node right = nodes.get(i + 1);
				double middleUnit = (left.unit + right.unit) * 0.5;
				double middleX = map(middleUnit, distribution.getLowerBound(),
						distribution.getUpperBound());
				double actual = distribution.cumulativeDirect(middleX, true, false);
				double predicted = interpolate(left.unit, right.unit,
						left.probability, right.probability, slopes[i], slopes[i + 1],
						middleUnit);
				double error = Math.abs(actual - predicted);
				passMaximumError = Math.max(passMaximumError, error);
				if (error > options.getTolerance()) {
					if (refined.size() + (nodes.size() - i) < options.getMaximumNodes()) {
						refined.add(new Node(middleUnit, middleX, actual));
						added = true;
					} else {
						saturated = true;
					}
				}
				refined.add(right);
			}
			nodes = refined;
			maxError = passMaximumError;
			if (!added) break;
		}
		return new NumericalCdfTable(distribution.getLowerBound(),
				distribution.getUpperBound(), nodes, maxError, saturated);
	}

	public int size() { return unit.length; }
	public double getMaximumValidationError() { return maximumValidationError; }
	public boolean isSaturated() { return saturated; }

	public double cumulative(double value) {
		if (Double.isNaN(value)) return Double.NaN;
		if (value <= lower) return 0.0;
		if (value >= upper) return 1.0;
		double u = inverseMap(value, lower, upper);
		int index = upperBound(unit, u) - 1;
		index = Math.max(0, Math.min(index, unit.length - 2));
		return clamp(interpolate(unit[index], unit[index + 1],
				probability[index], probability[index + 1], slope[index],
				slope[index + 1], u));
	}

	public double quantile(double target) {
		if (Double.isNaN(target) || target < 0.0 || target > 1.0) return Double.NaN;
		if (target == 0.0) return lower;
		if (target == 1.0) return upper;
		int high = lowerBound(probability, target);
		if (high <= 0) return x[0];
		if (high >= probability.length) return x[x.length - 1];
		int low = high - 1;
		if (probability[high] == probability[low]) return x[high];
		double lowUnit = unit[low];
		double highUnit = unit[high];
		for (int i = 0; i < 64; i++) {
			double middle = (lowUnit + highUnit) * 0.5;
			double value = interpolate(unit[low], unit[high], probability[low],
					probability[high], slope[low], slope[high], middle);
			if (value >= target) highUnit = middle;
			else lowUnit = middle;
		}
		return map(highUnit, lower, upper);
	}

	private static double[] values(List<Node> nodes, int kind) {
		double[] result = new double[nodes.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = kind == 0 ? nodes.get(i).unit : nodes.get(i).probability;
		}
		return result;
	}

	private static double[] monotoneSlopes(double[] x, double[] y) {
		int n = x.length;
		double[] slope = new double[n];
		double[] secant = new double[n - 1];
		for (int i = 0; i + 1 < n; i++) {
			secant[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
		}
		slope[0] = secant[0];
		slope[n - 1] = secant[n - 2];
		for (int i = 1; i + 1 < n; i++) {
			if (secant[i - 1] <= 0.0 || secant[i] <= 0.0) slope[i] = 0.0;
			else slope[i] = 2.0 / (1.0 / secant[i - 1] + 1.0 / secant[i]);
		}
		return slope;
	}

	private static double interpolate(double x0, double x1, double y0, double y1,
			double m0, double m1, double value) {
		double width = x1 - x0;
		double t = (value - x0) / width;
		double t2 = t * t;
		double t3 = t2 * t;
		return (2.0 * t3 - 3.0 * t2 + 1.0) * y0
				+ (t3 - 2.0 * t2 + t) * width * m0
				+ (-2.0 * t3 + 3.0 * t2) * y1
				+ (t3 - t2) * width * m1;
	}

	private static double map(double unit, double lower, double upper) {
		if (unit <= 0.0) return lower;
		if (unit >= 1.0) return upper;
		return ProbabilityFunctionAnalyzer.mapUnit(unit, lower, upper);
	}

	private static double inverseMap(double value, double lower, double upper) {
		if (Double.isFinite(lower) && Double.isFinite(upper)) {
			double width = upper * 0.5 - lower * 0.5;
			return (value * 0.5 - lower * 0.5) / width;
		}
		if (Double.isFinite(lower)) {
			return Math.atan(Math.max(0.0, value - lower)) * 2.0 / Math.PI;
		}
		if (Double.isFinite(upper)) {
			return 1.0 - Math.atan(Math.max(0.0, upper - value)) * 2.0 / Math.PI;
		}
		return Math.atan(value) / Math.PI + 0.5;
	}

	private static int upperBound(double[] values, double target) {
		int low = 0;
		int high = values.length;
		while (low < high) {
			int middle = (low + high) >>> 1;
			if (values[middle] <= target) low = middle + 1;
			else high = middle;
		}
		return low;
	}

	private static int lowerBound(double[] values, double target) {
		int low = 0;
		int high = values.length;
		while (low < high) {
			int middle = (low + high) >>> 1;
			if (values[middle] < target) low = middle + 1;
			else high = middle;
		}
		return low;
	}

	private static double clamp(double value) {
		return Math.max(0.0, Math.min(1.0, value));
	}

	private static final class Node {
		final double unit;
		final double x;
		final double probability;

		Node(double unit, double x, double probability) {
			this.unit = unit;
			this.x = x;
			this.probability = probability;
		}
	}
}
