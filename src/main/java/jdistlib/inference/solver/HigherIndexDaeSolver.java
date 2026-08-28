/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Projected velocity-Verlet solver for holonomic index-3 mechanical DAEs. */
public final class HigherIndexDaeSolver {
	private HigherIndexDaeSolver() {}

	/** Position/velocity trajectory and maximum observed constraint residual. */
	public static final class Result {
		private final double[][] positions, velocities; private final double maximumConstraintError;
		Result(double[][] positions, double[][] velocities, double maximumConstraintError) {
			this.positions = copy(positions); this.velocities = copy(velocities);
			this.maximumConstraintError = maximumConstraintError;
		}
		public double[][] positions() { return copy(positions); }
		public double[][] velocities() { return copy(velocities); }
		public double maximumConstraintError() { return maximumConstraintError; }
		private static double[][] copy(double[][] source) {
			double[][] result = new double[source.length][];
			for (int i = 0; i < source.length; i++) result[i] = source[i].clone(); return result;
		}
	}

	/** Integrates with fixed maximum step and Newton position/velocity projection. */
	public static Result integrate(HolonomicDaeSystem system, double[] initialPosition,
			double[] initialVelocity, double initialTime, double[] times, double maximumStep,
			double tolerance, int maximumProjectionIterations, double[] parameters, double[] data) {
		if (system == null || initialPosition == null || initialVelocity == null || times == null
				|| initialPosition.length != initialVelocity.length || !(maximumStep > 0)
				|| !(tolerance > 0) || maximumProjectionIterations < 1)
			throw new IllegalArgumentException("valid index-3 DAE state and controls required");
		int dimension = initialPosition.length, constraints = system.constraintCount();
		if (constraints < 1 || constraints > dimension)
			throw new IllegalArgumentException("constraint count must be in 1..state dimension");
		double[] p = parameters == null ? new double[0] : parameters.clone();
		double[] d = data == null ? new double[0] : data.clone();
		double[] q = initialPosition.clone(), v = initialVelocity.clone(); double time = initialTime;
		double[][] positions = new double[times.length][dimension], velocities = new double[times.length][dimension];
		double maximumError = project(system, time, q, v, p, d, tolerance, maximumProjectionIterations);
		for (int output = 0; output < times.length; output++) {
			if (!(times[output] > time)) throw new IllegalArgumentException("DAE times must be strictly increasing");
			while (time < times[output]) {
				double step = Math.min(maximumStep, times[output]-time);
				double[] acceleration = new double[dimension];
				system.acceleration(time, q, v, p, d, acceleration);
				for (int i = 0; i < dimension; i++) {
					v[i] += .5*step*acceleration[i]; q[i] += step*v[i];
				}
				time += step;
				maximumError = Math.max(maximumError,
						project(system, time, q, v, p, d, tolerance, maximumProjectionIterations));
				system.acceleration(time, q, v, p, d, acceleration);
				for (int i = 0; i < dimension; i++) v[i] += .5*step*acceleration[i];
				maximumError = Math.max(maximumError,
						projectVelocity(system, time, q, v, p, d, tolerance));
			}
			positions[output] = q.clone(); velocities[output] = v.clone();
		}
		return new Result(positions, velocities, maximumError);
	}

	private static double project(HolonomicDaeSystem system, double time, double[] q, double[] v,
			double[] p, double[] data, double tolerance, int maximumIterations) {
		double maximum = 0;
		for (int iteration = 0; iteration < maximumIterations; iteration++) {
			double[] residual = residual(system, time, q, p, data); double norm = AlgebraicSolver.infinityNorm(residual);
			maximum = Math.max(maximum, norm); if (norm <= tolerance) return maximum;
			double[][] jacobian = jacobian(system, time, q, p, data, residual);
			double[][] gram = gram(jacobian); double[] lambda = AlgebraicSolver.solveDense(gram, negate(residual));
			for (int state = 0; state < q.length; state++) for (int constraint = 0; constraint < residual.length; constraint++)
				q[state] += jacobian[constraint][state]*lambda[constraint];
		}
		throw new ArithmeticException("higher-index DAE position projection did not converge");
	}
	private static double projectVelocity(HolonomicDaeSystem system, double time, double[] q,
			double[] v, double[] p, double[] data, double tolerance) {
		double[] zero = residual(system, time, q, p, data); double[][] jacobian = jacobian(system, time, q, p, data, zero);
		double[] tangentResidual = new double[jacobian.length];
		for (int row = 0; row < jacobian.length; row++) for (int state = 0; state < v.length; state++)
			tangentResidual[row] += jacobian[row][state]*v[state];
		double norm = AlgebraicSolver.infinityNorm(tangentResidual); if (norm <= tolerance) return norm;
		double[] lambda = AlgebraicSolver.solveDense(gram(jacobian), negate(tangentResidual));
		for (int state = 0; state < v.length; state++) for (int constraint = 0; constraint < lambda.length; constraint++)
			v[state] += jacobian[constraint][state]*lambda[constraint];
		return norm;
	}
	private static double[] residual(HolonomicDaeSystem system, double time, double[] q, double[] p, double[] data) {
		double[] value = new double[system.constraintCount()]; system.constraints(time, q, p, data, value); return value;
	}
	private static double[][] jacobian(HolonomicDaeSystem system, double time, double[] q,
			double[] p, double[] data, double[] baseline) {
		double[][] result = new double[baseline.length][q.length];
		for (int column = 0; column < q.length; column++) {
			double[] shifted = q.clone(); double step = 1e-7*Math.max(1.0, Math.abs(q[column])); shifted[column] += step;
			double[] value = residual(system, time, shifted, p, data);
			for (int row = 0; row < baseline.length; row++) result[row][column] = (value[row]-baseline[row])/step;
		}
		return result;
	}
	private static double[][] gram(double[][] jacobian) {
		double[][] result = new double[jacobian.length][jacobian.length];
		for (int row = 0; row < result.length; row++) for (int column = 0; column < result.length; column++)
			for (int state = 0; state < jacobian[row].length; state++) result[row][column] += jacobian[row][state]*jacobian[column][state];
		return result;
	}
	private static double[] negate(double[] values) {
		double[] result = new double[values.length]; for (int i = 0; i < result.length; i++) result[i] = -values[i]; return result;
	}
}
