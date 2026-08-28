/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

import java.util.Arrays;

/** Finite-difference Newton solver for small and medium dense algebraic systems. */
public final class AlgebraicSolver {
	private AlgebraicSolver() {}

	/** Solver controls. */
	public static final class Options {
		public final double tolerance;
		public final double differenceStep;
		public final int maximumIterations;
		public Options(double tolerance, double differenceStep, int maximumIterations) {
			if (!(tolerance > 0) || !(differenceStep > 0) || maximumIterations < 1)
				throw new IllegalArgumentException("positive tolerances and iteration limit required");
			this.tolerance = tolerance; this.differenceStep = differenceStep;
			this.maximumIterations = maximumIterations;
		}
		public static Options defaults() { return new Options(1e-10, 1e-6, 100); }
	}

	/** Converged root and diagnostic counts. */
	public static final class Result {
		private final double[] solution;
		private final int iterations;
		private final int evaluations;
		private final double residualNorm;
		Result(double[] solution, int iterations, int evaluations, double residualNorm) {
			this.solution = solution; this.iterations = iterations;
			this.evaluations = evaluations; this.residualNorm = residualNorm;
		}
		public double[] solution() { return solution.clone(); }
		public int iterations() { return iterations; }
		public int evaluations() { return evaluations; }
		public double residualNorm() { return residualNorm; }
	}

	/** Solves from {@code initial}; the caller's arrays are never modified. */
	public static Result solve(AlgebraicSystem system, double[] initial,
			double[] parameters, double[] data, Options options) {
		if (system == null || initial == null || options == null)
			throw new NullPointerException("system, initial, and options are required");
		if (initial.length == 0) throw new IllegalArgumentException("non-empty state required");
		double[] state = initial.clone();
		double[] parameterCopy = parameters == null ? new double[0] : parameters.clone();
		double[] dataCopy = data == null ? new double[0] : data.clone();
		double[] residual = new double[state.length]; int evaluations = 0;
		for (int iteration = 0; iteration <= options.maximumIterations; iteration++) {
			system.evaluate(state, parameterCopy, dataCopy, residual); evaluations++;
			double norm = infinityNorm(residual);
			if (!Double.isFinite(norm)) throw new ArithmeticException("non-finite algebraic residual");
			if (norm <= options.tolerance)
				return new Result(state.clone(), iteration, evaluations, norm);
			if (iteration == options.maximumIterations) break;
			double[][] jacobian = new double[state.length][state.length];
			for (int column = 0; column < state.length; column++) {
				double saved = state[column];
				double step = options.differenceStep * Math.max(1.0, Math.abs(saved));
				state[column] = saved + step;
				double[] shifted = new double[state.length];
				system.evaluate(state, parameterCopy, dataCopy, shifted); evaluations++;
				state[column] = saved;
				for (int row = 0; row < state.length; row++)
					jacobian[row][column] = (shifted[row] - residual[row]) / step;
			}
			double[] step = solveDense(jacobian, negate(residual));
			double scale = 1.0;
			while (scale > 1.0 / 1024.0) {
				double[] candidate = state.clone();
				for (int i = 0; i < candidate.length; i++) candidate[i] += scale * step[i];
				double[] candidateResidual = new double[state.length];
				system.evaluate(candidate, parameterCopy, dataCopy, candidateResidual); evaluations++;
				if (infinityNorm(candidateResidual) < norm) { state = candidate; break; }
				scale *= 0.5;
			}
			if (scale <= 1.0 / 1024.0)
				throw new ArithmeticException("algebraic Newton line search failed");
		}
		throw new ArithmeticException("algebraic solver did not converge in "
				+ options.maximumIterations + " iterations");
	}

	/**
	 * Solves the system and differentiates the root with respect to parameters
	 * using the implicit-function identity {@code dx/dp = -Jx^-1 Jp}.
	 */
	public static SensitivityResult solveWithSensitivities(AlgebraicSystem system,
			double[] initial, double[] parameters, double[] data, Options options) {
		if (parameters == null) throw new IllegalArgumentException("parameters are required for sensitivities");
		double[] parameterCopy = parameters.clone();
		double[] dataCopy = data == null ? new double[0] : data.clone();
		Result solved = solve(system, initial, parameterCopy, dataCopy, options);
		double[] root = solved.solution(); int states = root.length, count = parameterCopy.length;
		double[][] stateJacobian = new double[states][states];
		double[] baseline = new double[states]; system.evaluate(root, parameterCopy, dataCopy, baseline);
		for (int column = 0; column < states; column++) {
			double[] shifted = root.clone();
			double step = options.differenceStep * Math.max(1.0, Math.abs(root[column]));
			shifted[column] += step; double[] residual = new double[states];
			system.evaluate(shifted, parameterCopy, dataCopy, residual);
			for (int row = 0; row < states; row++) stateJacobian[row][column] = (residual[row]-baseline[row])/step;
		}
		double[][] sensitivity = new double[states][count];
		for (int parameter = 0; parameter < count; parameter++) {
			double[] shifted = parameterCopy.clone();
			double step = options.differenceStep * Math.max(1.0, Math.abs(parameterCopy[parameter]));
			shifted[parameter] += step; double[] residual = new double[states];
			system.evaluate(root, shifted, dataCopy, residual);
			double[] right = new double[states];
			for (int row = 0; row < states; row++) right[row] = -(residual[row]-baseline[row])/step;
			double[] derivative = solveDense(stateJacobian, right);
			for (int state = 0; state < states; state++) sensitivity[state][parameter] = derivative[state];
		}
		return new SensitivityResult(new double[][] {root}, new double[][][] {sensitivity});
	}

	static double[] solveDense(double[][] matrix, double[] right) {
		int size = right.length; double[][] a = new double[size][size];
		for (int row = 0; row < size; row++) a[row] = Arrays.copyOf(matrix[row], size);
		double[] b = right.clone();
		for (int pivot = 0; pivot < size; pivot++) {
			int best = pivot;
			for (int row = pivot + 1; row < size; row++)
				if (Math.abs(a[row][pivot]) > Math.abs(a[best][pivot])) best = row;
			if (!(Math.abs(a[best][pivot]) > 1e-15)) throw new ArithmeticException("singular Jacobian");
			double[] swap = a[pivot]; a[pivot] = a[best]; a[best] = swap;
			double bSwap = b[pivot]; b[pivot] = b[best]; b[best] = bSwap;
			for (int row = pivot + 1; row < size; row++) {
				double factor = a[row][pivot] / a[pivot][pivot];
				for (int column = pivot; column < size; column++) a[row][column] -= factor * a[pivot][column];
				b[row] -= factor * b[pivot];
			}
		}
		double[] result = new double[size];
		for (int row = size - 1; row >= 0; row--) {
			double value = b[row];
			for (int column = row + 1; column < size; column++) value -= a[row][column] * result[column];
			result[row] = value / a[row][row];
		}
		return result;
	}
	private static double[] negate(double[] values) {
		double[] result = new double[values.length];
		for (int i = 0; i < values.length; i++) result[i] = -values[i];
		return result;
	}
	public static double infinityNorm(double[] values) {
		double result = 0;
		for (double value : values) result = Math.max(result, Math.abs(value));
		return result;
	}
}
