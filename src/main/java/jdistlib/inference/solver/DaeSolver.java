/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Implicit-Euler solver for index-1 differential-algebraic systems. */
public final class DaeSolver {
	private DaeSolver() {}

	/** Integrates at requested increasing times using implicit Euler and Newton iterations. */
	public static double[][] integrate(DaeSystem system, double[] initial, double initialTime,
			double[] times, double[] parameters, double[] data, AlgebraicSolver.Options options) {
		if (system == null || initial == null || times == null)
			throw new NullPointerException("system, state, and times are required");
		double[] previous = initial.clone(); double previousTime = initialTime;
		double[] p = parameters == null ? new double[0] : parameters.clone();
		double[] d = data == null ? new double[0] : data.clone();
		double[][] result = new double[times.length][initial.length];
		for (int output = 0; output < times.length; output++) {
			final double time = times[output], step = time - previousTime;
			if (!(step > 0)) throw new IllegalArgumentException("DAE times must be strictly increasing");
			final double[] base = previous.clone();
			AlgebraicSystem equation = (state, ignoredParameters, ignoredData, residual) -> {
				double[] derivative = new double[state.length];
				for (int i = 0; i < state.length; i++) derivative[i] = (state[i] - base[i]) / step;
				system.residual(time, state, derivative, p, d, residual);
			};
			previous = AlgebraicSolver.solve(equation, previous, null, null,
					options == null ? AlgebraicSolver.Options.defaults() : options).solution();
			result[output] = previous.clone(); previousTime = time;
		}
		return result;
	}
}
