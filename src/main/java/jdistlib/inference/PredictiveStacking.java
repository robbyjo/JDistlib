/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Optimizes simplex weights for stacking pointwise out-of-sample predictions. */
public final class PredictiveStacking {
	private PredictiveStacking() {}
	public static Result fit(String[] modelNames, double[][] pointwiseLogPredictiveDensity) {
		PredictiveMath.requireFiniteMatrix(pointwiseLogPredictiveDensity, 2, 1);
		int modelCount = pointwiseLogPredictiveDensity.length;
		if (modelNames == null || modelNames.length != modelCount) throw new IllegalArgumentException("model names must match rows");
		String[] names = modelNames.clone();
		for (String name : names) if (name == null || name.trim().isEmpty()) throw new IllegalArgumentException("model names required");
		double[] weights = new double[modelCount];
		for (int model = 0; model < modelCount; model++) weights[model] = 1.0 / modelCount;
		double objective = objective(weights, pointwiseLogPredictiveDensity), step = 0.1;
		boolean converged = false; int iteration;
		for (iteration = 0; iteration < 10000; iteration++) {
			double[] gradient = gradient(weights, pointwiseLogPredictiveDensity);
			double weightedGradient = 0.0;
			for (int model = 0; model < modelCount; model++) weightedGradient += weights[model] * gradient[model];
			double[] candidate = new double[modelCount]; double sum = 0.0;
			for (int model = 0; model < modelCount; model++) {
				candidate[model] = weights[model] * Math.exp(Math.max(-50.0, Math.min(50.0,
						step * (gradient[model] - weightedGradient))));
				sum += candidate[model];
			}
			for (int model = 0; model < modelCount; model++) candidate[model] /= sum;
			double candidateObjective = objective(candidate, pointwiseLogPredictiveDensity);
			if (candidateObjective + 1e-12 < objective) { step *= 0.5; continue; }
			double change = 0.0;
			for (int model = 0; model < modelCount; model++) change = Math.max(change, Math.abs(candidate[model] - weights[model]));
			weights = candidate; objective = candidateObjective; step = Math.min(1.0, step * 1.02);
			if (change < 1e-10) { converged = true; break; }
		}
		return new Result(names, weights, objective, converged, Math.min(iteration + 1, 10000));
	}
	public static Result fit(String[] names, PsisLoo.Result... results) {
		if (results == null || results.length != names.length) throw new IllegalArgumentException("names and results must match");
		double[][] pointwise = new double[results.length][];
		ObservationMetadata metadata = results[0].metadata();
		for (int model = 0; model < results.length; model++) {
			if (!metadata.equals(results[model].metadata())) throw new IllegalArgumentException("observation metadata must match");
			pointwise[model] = results[model].pointwiseElpd();
		}
		return fit(names, pointwise);
	}
	private static double objective(double[] weights, double[][] values) {
		double result = 0.0;
		for (int observation = 0; observation < values[0].length; observation++) {
			double maximum = Double.NEGATIVE_INFINITY;
			for (int model = 0; model < values.length; model++) maximum = Math.max(maximum, values[model][observation]);
			double mixture = 0.0;
			for (int model = 0; model < values.length; model++) mixture += weights[model] * Math.exp(values[model][observation] - maximum);
			result += maximum + Math.log(mixture);
		}
		return result;
	}
	private static double[] gradient(double[] weights, double[][] values) {
		double[] result = new double[weights.length];
		for (int observation = 0; observation < values[0].length; observation++) {
			double maximum = Double.NEGATIVE_INFINITY;
			for (int model = 0; model < values.length; model++) maximum = Math.max(maximum, values[model][observation]);
			double denominator = 0.0;
			for (int model = 0; model < values.length; model++) denominator += weights[model] * Math.exp(values[model][observation] - maximum);
			for (int model = 0; model < values.length; model++) result[model] += Math.exp(values[model][observation] - maximum) / denominator;
		}
		return result;
	}
	public static final class Result {
		private final String[] modelNames; private final double[] weights;
		private final double objective; private final boolean converged; private final int iterations;
		private Result(String[] modelNames, double[] weights, double objective, boolean converged, int iterations) {
			this.modelNames = modelNames; this.weights = weights; this.objective = objective;
			this.converged = converged; this.iterations = iterations;
		}
		public String[] modelNames() { return modelNames.clone(); }
		public double[] weights() { return weights.clone(); }
		public double objective() { return objective; }
		public boolean converged() { return converged; }
		public int iterations() { return iterations; }
	}
}
