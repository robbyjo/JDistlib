/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

final class PredictiveMath {
	private PredictiveMath() {}
	static double logSumExp(double[] values) {
		double maximum = Double.NEGATIVE_INFINITY;
		for (double value : values) maximum = Math.max(maximum, value);
		if (maximum == Double.NEGATIVE_INFINITY) return maximum;
		double sum = 0.0;
		for (double value : values) sum += Math.exp(value - maximum);
		return maximum + Math.log(sum);
	}
	static double logMeanExp(double[] values) { return logSumExp(values) - Math.log(values.length); }
	static double sampleVariance(double[] values) {
		if (values.length < 2) return 0.0;
		double mean = 0.0;
		for (double value : values) mean += value;
		mean /= values.length;
		double sum = 0.0;
		for (double value : values) { double difference = value - mean; sum += difference * difference; }
		return sum / (values.length - 1.0);
	}
	static void requireFiniteMatrix(double[][] values, int minimumRows, int minimumColumns) {
		if (values == null || values.length < minimumRows) throw new IllegalArgumentException("too few rows");
		int columns = values[0].length;
		if (columns < minimumColumns) throw new IllegalArgumentException("too few columns");
		for (double[] row : values) {
			if (row == null || row.length != columns) throw new IllegalArgumentException("matrix must be rectangular");
			for (double value : row) if (!Double.isFinite(value)) throw new IllegalArgumentException("matrix must be finite");
		}
	}
}
