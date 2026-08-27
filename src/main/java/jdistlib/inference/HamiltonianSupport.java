/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Arrays;

import jdistlib.rng.RandomEngine;

/** Package-private numerical support shared by HMC and NUTS. */
final class HamiltonianSupport {
	private HamiltonianSupport() {}

	static DifferentiableLogDensity gradientTarget(LogDensity target,
			SamplingOptions options) {
		if (target instanceof BayesianModel) target = ((BayesianModel) target).evaluator();
		if (target instanceof DifferentiableLogDensity) {
			if (target instanceof GradientProvider
					&& !((GradientProvider) target).hasAnalyticGradient()
					&& !options.allowFiniteDifferences()) {
				throw new IllegalArgumentException("target lacks analytic gradients; explicitly enable finite differences to proceed");
			}
			return (DifferentiableLogDensity) target;
		}
		if (!options.allowFiniteDifferences())
			throw new IllegalArgumentException("HMC/NUTS require DifferentiableLogDensity unless finite differences are explicitly enabled");
		return Gradients.finiteDifference(target);
	}

	static final class Point {
		final double[] q;
		final double[] p;
		final double[] gradient;
		final double logDensity;
		Point(double[] q, double[] p, double[] gradient, double logDensity) {
			this.q = q; this.p = p; this.gradient = gradient; this.logDensity = logDensity;
		}
		Point copy() { return new Point(q.clone(), p.clone(), gradient.clone(), logDensity); }
	}

	static Point at(DifferentiableLogDensity target, double[] q) {
		double[] gradient = new double[q.length];
		double value = target.logDensityAndGradient(q, gradient);
		return new Point(q.clone(), new double[q.length], gradient, value);
	}

	static Point leapfrog(DifferentiableLogDensity target, Point point,
			double step, MassMatrix mass) {
		double[] p = point.p.clone();
		for (int i = 0; i < p.length; i++) p[i] += 0.5 * step * point.gradient[i];
		double[] q = point.q.clone();
		mass.addScaledVelocity(p, step, q);
		double[] gradient = new double[q.length];
		double value = target.logDensityAndGradient(q, gradient);
		for (int i = 0; i < p.length; i++) p[i] += 0.5 * step * gradient[i];
		return new Point(q, p, gradient, value);
	}

	static final class MassMatrix {
		private final int dimension;
		private double[][] inverse;
		private double[][] cholesky;
		MassMatrix(int dimension) {
			this.dimension = dimension;
			inverse = identity(dimension);
			cholesky = identity(dimension);
		}
		double[] momentum(RandomEngine random) {
			double[] z = new double[dimension];
			for (int i = 0; i < dimension; i++) z[i] = random.nextGaussian();
			double[] result = new double[dimension];
			for (int i = 0; i < dimension; i++)
				for (int j = 0; j <= i; j++) result[i] += cholesky[i][j] * z[j];
			return result;
		}
		double[] velocity(double[] momentum) {
			double[] result = new double[dimension];
			velocityInto(momentum, result);
			return result;
		}
		void velocityInto(double[] momentum, double[] result) {
			Arrays.fill(result, 0.0);
			for (int i = 0; i < dimension; i++)
				for (int j = 0; j < dimension; j++) result[i] += inverse[i][j] * momentum[j];
		}
		void addScaledVelocity(double[] momentum, double scale, double[] position) {
			for (int i = 0; i < dimension; i++) {
				double value = 0.0;
				for (int j = 0; j < dimension; j++) value += inverse[i][j] * momentum[j];
				position[i] += scale * value;
			}
		}
		double velocityDot(double[] momentum, double[] vector) {
			double result = 0.0;
			for (int i = 0; i < dimension; i++) {
				double velocity = 0.0;
				for (int j = 0; j < dimension; j++) velocity += inverse[i][j] * momentum[j];
				result += velocity * vector[i];
			}
			return result;
		}
		double kinetic(double[] momentum) {
			return 0.5 * velocityDot(momentum, momentum);
		}
		double[] inverseDiagonal() {
			double[] result = new double[dimension];
			for (int i = 0; i < dimension; i++) result[i] = inverse[i][i];
			return result;
		}
		double[][] inverseMatrix() {
			double[][] result = new double[dimension][dimension];
			for (int i = 0; i < dimension; i++)
				System.arraycopy(inverse[i], 0, result[i], 0, dimension);
			return result;
		}
		void update(double[][] covariance, boolean dense) {
			double[][] regularized = new double[dimension][dimension];
			for (int i = 0; i < dimension; i++) {
				for (int j = 0; j < dimension; j++)
					regularized[i][j] = dense ? covariance[i][j] : 0.0;
				regularized[i][i] = Math.max(1e-3, covariance[i][i] + 1e-3);
			}
			double[][] factor = cholesky(regularized);
			if (factor != null) {
				cholesky = factor;
				inverse = inverseFromCholesky(factor);
			}
		}
	}

	static final class RunningCovariance {
		private final int dimension;
		private int count;
		private final double[] mean;
		private final double[] delta;
		private final double[][] sumProducts;
		RunningCovariance(int dimension) {
			this.dimension = dimension;
			mean = new double[dimension];
			delta = new double[dimension];
			sumProducts = new double[dimension][dimension];
		}
		void add(double[] value) {
			count++;
			for (int i = 0; i < dimension; i++) {
				delta[i] = value[i] - mean[i];
				mean[i] += delta[i] / count;
			}
			for (int i = 0; i < dimension; i++)
				for (int j = 0; j <= i; j++) {
					sumProducts[i][j] += delta[i] * (value[j] - mean[j]);
					sumProducts[j][i] = sumProducts[i][j];
				}
		}
		double[][] covariance() {
			double[][] result = new double[dimension][dimension];
			if (count < 2) {
				for (int i = 0; i < dimension; i++) result[i][i] = 1.0;
				return result;
			}
			for (int i = 0; i < dimension; i++)
				for (int j = 0; j < dimension; j++)
					result[i][j] = sumProducts[i][j] / (count - 1.0);
			return result;
		}
		int count() { return count; }
	}

	static final class DualAveraging {
		private final double target;
		private final double mu;
		private double hbar;
		private double logAveraged;
		private int iteration;
		DualAveraging(double initial, double target) {
			this.target = target; mu = Math.log(10.0 * initial);
			logAveraged = Math.log(initial);
		}
		double update(double acceptance) {
			iteration++;
			double weight = 1.0 / (iteration + 10.0);
			hbar = (1.0 - weight) * hbar + weight * (target - acceptance);
			double logStep = mu - Math.sqrt(iteration) / 0.05 * hbar;
			double averageWeight = Math.pow(iteration, -0.75);
			logAveraged = averageWeight * logStep + (1.0 - averageWeight) * logAveraged;
			return Math.exp(Math.max(-20.0, Math.min(5.0, logStep)));
		}
		double averaged() { return Math.exp(logAveraged); }
	}

	static double findReasonableStep(DifferentiableLogDensity target, Point state,
			MassMatrix mass, RandomEngine random, double initial) {
		double step = initial;
		double[] momentum = mass.momentum(random);
		Point start = new Point(state.q, momentum, state.gradient, state.logDensity);
		double initialHamiltonian = -start.logDensity + mass.kinetic(momentum);
		Point proposal = leapfrog(target, start, step, mass);
		double logAcceptance = proposal.logDensity - mass.kinetic(proposal.p)
				+ initialHamiltonian;
		int direction = Double.isFinite(logAcceptance) && logAcceptance > Math.log(0.5) ? 1 : -1;
		for (int i = 0; i < 20; i++) {
			boolean acceptable = Double.isFinite(logAcceptance)
					&& logAcceptance > Math.log(0.5);
			if ((direction > 0) != acceptable) break;
			step = direction > 0 ? step * 2.0 : step * 0.5;
			proposal = leapfrog(target, start, step, mass);
			logAcceptance = proposal.logDensity - mass.kinetic(proposal.p)
					+ initialHamiltonian;
		}
		return Math.max(1e-8, Math.min(100.0, step));
	}

	private static double[][] identity(int dimension) {
		double[][] result = new double[dimension][dimension];
		for (int i = 0; i < dimension; i++) result[i][i] = 1.0;
		return result;
	}
	private static double[][] cholesky(double[][] matrix) {
		int n = matrix.length;
		double[][] result = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				double sum = matrix[i][j];
				for (int k = 0; k < j; k++) sum -= result[i][k] * result[j][k];
				if (i == j) {
					if (!(sum > 0.0) || !Double.isFinite(sum)) return null;
					result[i][j] = Math.sqrt(sum);
				} else result[i][j] = sum / result[j][j];
			}
		}
		return result;
	}
	private static double[][] inverseFromCholesky(double[][] lower) {
		int n = lower.length;
		double[][] inverse = new double[n][n];
		for (int column = 0; column < n; column++) {
			double[] y = new double[n];
			for (int i = 0; i < n; i++) {
				double value = i == column ? 1.0 : 0.0;
				for (int j = 0; j < i; j++) value -= lower[i][j] * y[j];
				y[i] = value / lower[i][i];
			}
			double[] x = new double[n];
			for (int i = n - 1; i >= 0; i--) {
				double value = y[i];
				for (int j = i + 1; j < n; j++) value -= lower[j][i] * x[j];
				x[i] = value / lower[i][i];
			}
			for (int row = 0; row < n; row++) inverse[row][column] = x[row];
		}
		return inverse;
	}
}
