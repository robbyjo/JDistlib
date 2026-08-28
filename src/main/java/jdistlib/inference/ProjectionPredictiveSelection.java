/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Forward projection-predictive variable selection for Gaussian linear reference models. */
public final class ProjectionPredictiveSelection {
	private ProjectionPredictiveSelection() {}
	public static Result select(double[][] design, String[] variableNames, double[][] coefficientDraws,
			double[] interceptDraws, double[] residualScaleDraws, double maximumMeanKl) {
		PredictiveMath.requireFiniteMatrix(design, 1, 1);
		PredictiveMath.requireFiniteMatrix(coefficientDraws, 1, design[0].length);
		int observations = design.length, variables = design[0].length, draws = coefficientDraws.length;
		if (coefficientDraws[0].length != variables || variableNames == null || variableNames.length != variables
				|| residualScaleDraws == null || residualScaleDraws.length != draws
				|| (interceptDraws != null && interceptDraws.length != draws)
				|| maximumMeanKl < 0.0 || !Double.isFinite(maximumMeanKl))
			throw new IllegalArgumentException("projection inputs do not match");
		for (String name : variableNames) if (name == null || name.trim().isEmpty()) throw new IllegalArgumentException("variable names required");
		for (double scale : residualScaleDraws) if (!(scale > 0.0) || !Double.isFinite(scale)) throw new IllegalArgumentException("residual scales must be positive");
		if (interceptDraws != null) for (double intercept : interceptDraws)
			if (!Double.isFinite(intercept)) throw new IllegalArgumentException("intercepts must be finite");
		double[][] reference = new double[draws][observations];
		for (int draw = 0; draw < draws; draw++) for (int row = 0; row < observations; row++) {
			double value = interceptDraws == null ? 0.0 : interceptDraws[draw];
			for (int variable = 0; variable < variables; variable++) value += design[row][variable] * coefficientDraws[draw][variable];
			reference[draw][row] = value;
		}
		List<Step> path = new ArrayList<Step>(); boolean[] included = new boolean[variables];
		int[] subset = new int[0];
		path.add(step(subset, variableNames, meanKl(design, reference, residualScaleDraws, subset)));
		for (int size = 1; size <= variables; size++) {
			int best = -1; double bestLoss = Double.POSITIVE_INFINITY;
			for (int candidate = 0; candidate < variables; candidate++) if (!included[candidate]) {
				int[] trial = append(subset, candidate);
				double loss = meanKl(design, reference, residualScaleDraws, trial);
				if (loss < bestLoss) { bestLoss = loss; best = candidate; }
			}
			included[best] = true; subset = append(subset, best);
			path.add(step(subset, variableNames, bestLoss));
		}
		int selectedStep = path.size() - 1;
		for (int i = 0; i < path.size(); i++) if (path.get(i).meanKl() <= maximumMeanKl) { selectedStep = i; break; }
		return new Result(variableNames.clone(), path, selectedStep, maximumMeanKl);
	}
	private static Step step(int[] subset, String[] names, double loss) {
		String[] selectedNames = new String[subset.length];
		for (int i = 0; i < subset.length; i++) selectedNames[i] = names[subset[i]];
		return new Step(subset.clone(), selectedNames, loss);
	}
	private static int[] append(int[] values, int value) {
		int[] result = new int[values.length + 1]; System.arraycopy(values, 0, result, 0, values.length); result[values.length] = value; return result;
	}
	private static double meanKl(double[][] design, double[][] reference, double[] scales, int[] subset) {
		double total = 0.0;
		for (int draw = 0; draw < reference.length; draw++) {
			double[] projected = leastSquares(design, reference[draw], subset);
			double squaredError = 0.0;
			for (int row = 0; row < design.length; row++) {
				double fitted = projected[0];
				for (int column = 0; column < subset.length; column++) fitted += design[row][subset[column]] * projected[column + 1];
				double difference = reference[draw][row] - fitted; squaredError += difference * difference;
			}
			double variance = scales[draw] * scales[draw];
			total += 0.5 * Math.log1p(squaredError / design.length / variance);
		}
		return total / reference.length;
	}
	private static double[] leastSquares(double[][] design, double[] response, int[] subset) {
		int columns = subset.length + 1; double[][] system = new double[columns][columns]; double[] rhs = new double[columns];
		for (int row = 0; row < design.length; row++) {
			double[] values = new double[columns]; values[0] = 1.0;
			for (int column = 1; column < columns; column++) values[column] = design[row][subset[column - 1]];
			for (int first = 0; first < columns; first++) {
				rhs[first] += values[first] * response[row];
				for (int second = 0; second < columns; second++) system[first][second] += values[first] * values[second];
			}
		}
		double ridge = Math.max(1e-12, design.length * 1e-12);
		for (int column = 0; column < columns; column++) system[column][column] += ridge;
		return solve(system, rhs);
	}
	private static double[] solve(double[][] matrix, double[] rhs) {
		int size = rhs.length; double[][] a = new double[size][size]; double[] b = rhs.clone();
		for (int row = 0; row < size; row++) a[row] = matrix[row].clone();
		for (int pivot = 0; pivot < size; pivot++) {
			int best = pivot;
			for (int row = pivot + 1; row < size; row++) if (Math.abs(a[row][pivot]) > Math.abs(a[best][pivot])) best = row;
			double[] swap = a[pivot]; a[pivot] = a[best]; a[best] = swap;
			double scalar = b[pivot]; b[pivot] = b[best]; b[best] = scalar;
			if (Math.abs(a[pivot][pivot]) < 1e-15) throw new IllegalArgumentException("singular projection design");
			for (int row = pivot + 1; row < size; row++) {
				double factor = a[row][pivot] / a[pivot][pivot]; b[row] -= factor * b[pivot];
				for (int column = pivot; column < size; column++) a[row][column] -= factor * a[pivot][column];
			}
		}
		double[] result = new double[size];
		for (int row = size - 1; row >= 0; row--) {
			double value = b[row]; for (int column = row + 1; column < size; column++) value -= a[row][column] * result[column];
			result[row] = value / a[row][row];
		}
		return result;
	}
	public static final class Step {
		private final int[] variableIndices; private final String[] variableNames; private final double meanKl;
		private Step(int[] variableIndices, String[] variableNames, double meanKl) { this.variableIndices = variableIndices; this.variableNames = variableNames; this.meanKl = meanKl; }
		public int size() { return variableIndices.length; }
		public int[] variableIndices() { return variableIndices.clone(); }
		public String[] variableNames() { return variableNames.clone(); }
		public double meanKl() { return meanKl; }
	}
	public static final class Result {
		private final String[] allVariables; private final List<Step> path; private final int selectedStep; private final double tolerance;
		private Result(String[] allVariables, List<Step> path, int selectedStep, double tolerance) {
			this.allVariables = allVariables; this.path = Collections.unmodifiableList(new ArrayList<Step>(path)); this.selectedStep = selectedStep; this.tolerance = tolerance;
		}
		public String[] allVariables() { return allVariables.clone(); }
		public List<Step> path() { return path; }
		public Step selectedModel() { return path.get(selectedStep); }
		public double maximumMeanKl() { return tolerance; }
	}
}
