/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Multi-path quasi-Newton Gaussian approximation with mixture scoring and PSIS resampling. */
public final class Pathfinder {
	private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);
	private Pathfinder() {}
	public static PathfinderFit fit(DifferentiableLogDensity target, double[][] initialStates,
			PathfinderOptions options, RandomEngine random) {
		if (target == null || initialStates == null || initialStates.length == 0 || options == null || random == null)
			throw new IllegalArgumentException("target, states, options, and random are required");
		int dimension = initialStates[0].length, paths = options.paths();
		if (dimension == 0) throw new IllegalArgumentException("state dimension must be positive");
		OptimizationResult[] optimizations = new OptimizationResult[paths];
		Gaussian[] approximations = new Gaussian[paths];
		int[] selectedIterations = new int[paths]; double[] selectedElbos = new double[paths];
		for (int path = 0; path < paths; path++) {
			double[] initial = initialStates[path % initialStates.length].clone();
			if (initial.length != dimension) throw new IllegalArgumentException("state dimensions must match");
			if (path >= initialStates.length) for (int d = 0; d < dimension; d++) initial[d] += 0.1 * random.nextGaussian();
			OptimizationTrace trace = LbfgsOptimizer.trace(target, initial, options.maximumIterations(), options.historySize(), options.tolerance());
			optimizations[path] = trace.result(); Selection selection = selectApproximation(target, trace, random);
			approximations[path] = selection.gaussian; selectedIterations[path] = selection.iteration; selectedElbos[path] = selection.elbo;
		}
		int count = paths * options.drawsPerPath(); double[][] candidates = new double[count][dimension];
		double[] logRatios = new double[count]; int index = 0;
		for (int path = 0; path < paths; path++) for (int draw = 0; draw < options.drawsPerPath(); draw++) {
			candidates[index] = approximations[path].sample(random);
			double logTarget = target.logDensity(candidates[index]);
			double[] mixtureTerms = new double[paths];
			for (int component = 0; component < paths; component++) mixtureTerms[component] = approximations[component].logDensity(candidates[index]);
			logRatios[index] = logTarget - logSumExp(mixtureTerms) + Math.log(paths); index++;
		}
		ParetoSmoothedImportanceSampling.Result psis = ParetoSmoothedImportanceSampling.smooth(logRatios);
		double[] logWeights = psis.logWeights(); double[][] draws = systematicResample(candidates, logWeights, options.resampledDraws(), random);
		return new PathfinderFit(draws, candidates, logWeights, optimizations, selectedIterations, selectedElbos, psis.paretoK());
	}
	private static Selection selectApproximation(DifferentiableLogDensity target, OptimizationTrace trace, RandomEngine random) {
		double[][] points = trace.points(); double[][][] inverseHessians = trace.inverseHessians();
		int candidates = Math.min(8, points.length); Gaussian best = null; double bestElbo = Double.NEGATIVE_INFINITY; int bestIteration = points.length - 1;
		for (int candidate = 0; candidate < candidates; candidate++) { int iteration = candidates == 1 ? points.length - 1
				: (int) Math.round(candidate * (points.length - 1.0) / (candidates - 1.0)); Gaussian approximation = gaussianApproximation(target, points[iteration], inverseHessians[iteration]);
			double elbo = 0.0; int finite = 0; for (int draw = 0; draw < 5; draw++) { double[] value = approximation.sample(random); double term = target.logDensity(value) - approximation.logDensity(value);
				if (Double.isFinite(term)) { elbo += term; finite++; } } elbo = finite == 0 ? Double.NEGATIVE_INFINITY : elbo / finite;
			if (elbo > bestElbo) { bestElbo = elbo; best = approximation; bestIteration = iteration; }
		}
		if (best == null) { best = gaussianApproximation(target, points[points.length - 1]); bestElbo = Double.NEGATIVE_INFINITY; }
		return new Selection(best, bestIteration, bestElbo);
	}
	private static Gaussian gaussianApproximation(DifferentiableLogDensity target, double[] mean, double[][] covariance) {
		double[][] regularized = copy(covariance); double jitter = 1e-10; double[][] covarianceCholesky = null;
		for (int attempt = 0; attempt < 12; attempt++) { double[][] candidate = copy(regularized); for (int i = 0; i < candidate.length; i++) candidate[i][i] += jitter;
			covarianceCholesky = cholesky(candidate); if (covarianceCholesky != null) { regularized = candidate; break; } jitter *= 10.0; }
		if (covarianceCholesky == null) return gaussianApproximation(target, mean); double[][] precision = inverseFromCholesky(covarianceCholesky);
		return new Gaussian(mean, precision, covarianceCholesky);
	}
	private static final class Selection { final Gaussian gaussian; final int iteration; final double elbo;
		Selection(Gaussian gaussian, int iteration, double elbo) { this.gaussian = gaussian; this.iteration = iteration; this.elbo = elbo; } }
	private static Gaussian gaussianApproximation(DifferentiableLogDensity target, double[] mode) {
		int n = mode.length; double[][] precision = new double[n][n];
		double[] plusGradient = new double[n], minusGradient = new double[n];
		for (int column = 0; column < n; column++) {
			double h = Math.cbrt(Math.ulp(1.0)) * Math.max(1.0, Math.abs(mode[column]));
			double[] plus = mode.clone(), minus = mode.clone(); plus[column] += h; minus[column] -= h;
			target.logDensityAndGradient(plus, plusGradient); target.logDensityAndGradient(minus, minusGradient);
			for (int row = 0; row < n; row++) precision[row][column] = -(plusGradient[row] - minusGradient[row]) / (2.0 * h);
		}
		for (int row = 0; row < n; row++) for (int column = row + 1; column < n; column++) {
			double symmetric = 0.5 * (precision[row][column] + precision[column][row]);
			precision[row][column] = symmetric; precision[column][row] = symmetric;
		}
		double jitter = 1e-8; double[][] precisionCholesky = null;
		for (int attempt = 0; attempt < 12; attempt++) {
			double[][] candidate = copy(precision); for (int i = 0; i < n; i++) candidate[i][i] += jitter;
			precisionCholesky = cholesky(candidate); if (precisionCholesky != null) { precision = candidate; break; }
			jitter *= 10.0;
		}
		if (precisionCholesky == null) { precision = new double[n][n]; for (int i = 0; i < n; i++) precision[i][i] = 1.0; precisionCholesky = cholesky(precision); }
		double[][] covariance = inverseFromCholesky(precisionCholesky), covarianceCholesky = cholesky(covariance);
		return new Gaussian(mode, precision, covarianceCholesky);
	}
	private static final class Gaussian {
		private final double[] mean; private final double[][] precision, covarianceCholesky; private final double logNormalizer;
		Gaussian(double[] mean, double[][] precision, double[][] covarianceCholesky) {
			this.mean = mean.clone(); this.precision = precision; this.covarianceCholesky = covarianceCholesky;
			double logDetPrecision = 0.0; double[][] precisionCholesky = cholesky(precision);
			for (int i = 0; i < mean.length; i++) logDetPrecision += 2.0 * Math.log(precisionCholesky[i][i]);
			logNormalizer = 0.5 * logDetPrecision - 0.5 * mean.length * LOG_TWO_PI;
		}
		double[] sample(RandomEngine random) { double[] result = mean.clone(), z = new double[mean.length];
			for (int i = 0; i < z.length; i++) z[i] = random.nextGaussian();
			for (int row = 0; row < mean.length; row++) for (int column = 0; column <= row; column++) result[row] += covarianceCholesky[row][column] * z[column]; return result; }
		double logDensity(double[] value) { double quadratic = 0.0;
			for (int i = 0; i < mean.length; i++) for (int j = 0; j < mean.length; j++) quadratic += (value[i] - mean[i]) * precision[i][j] * (value[j] - mean[j]);
			return logNormalizer - 0.5 * quadratic; }
	}
	private static double[][] systematicResample(double[][] candidates, double[] logWeights, int count, RandomEngine random) {
		double[][] result = new double[count][]; double step = 1.0 / count, position = random.nextDouble() * step, cumulative = Math.exp(logWeights[0]); int source = 0;
		for (int i = 0; i < count; i++, position += step) { while (position > cumulative && source + 1 < candidates.length) cumulative += Math.exp(logWeights[++source]); result[i] = candidates[source].clone(); }
		return result;
	}
	private static double logSumExp(double[] values) { double maximum = Double.NEGATIVE_INFINITY; for (double value : values) maximum = Math.max(maximum, value);
		double sum = 0.0; for (double value : values) sum += Math.exp(value - maximum); return maximum + Math.log(sum); }
	private static double[][] cholesky(double[][] matrix) { int n = matrix.length; double[][] result = new double[n][n];
		for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) { double sum = matrix[i][j]; for (int k = 0; k < j; k++) sum -= result[i][k] * result[j][k];
			if (i == j) { if (!(sum > 0.0) || !Double.isFinite(sum)) return null; result[i][j] = Math.sqrt(sum); } else result[i][j] = sum / result[j][j]; } return result; }
	private static double[][] inverseFromCholesky(double[][] lower) { int n = lower.length; double[][] inverse = new double[n][n];
		for (int column = 0; column < n; column++) { double[] y = new double[n], x = new double[n];
			for (int i = 0; i < n; i++) { double sum = i == column ? 1.0 : 0.0; for (int k = 0; k < i; k++) sum -= lower[i][k] * y[k]; y[i] = sum / lower[i][i]; }
			for (int i = n - 1; i >= 0; i--) { double sum = y[i]; for (int k = i + 1; k < n; k++) sum -= lower[k][i] * x[k]; x[i] = sum / lower[i][i]; }
			for (int i = 0; i < n; i++) inverse[i][column] = x[i]; } return inverse; }
	private static double[][] copy(double[][] matrix) { double[][] result = new double[matrix.length][]; for (int i = 0; i < matrix.length; i++) result[i] = matrix[i].clone(); return result; }
}
