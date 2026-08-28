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

	/** Differentiates a DAE trajectory by consistently perturbing each parameter. */
	public static SensitivityResult integrateWithSensitivities(DaeSystem system,
			double[] initial, double initialTime, double[] times, double[] parameters,
			double[] data, AlgebraicSolver.Options options) {
		if (parameters == null) throw new IllegalArgumentException("parameters are required for sensitivities");
		double[][] values = integrate(system, initial, initialTime, times, parameters, data, options);
		double[][][] sensitivity = new double[times.length][initial.length][parameters.length];
		for (int parameter = 0; parameter < parameters.length; parameter++) {
			double[] shifted = parameters.clone();
			double step = 1e-6 * Math.max(1.0, Math.abs(parameters[parameter])); shifted[parameter] += step;
			double[][] perturbed = integrate(system, initial, initialTime, times, shifted, data, options);
			for (int output = 0; output < times.length; output++) for (int state = 0; state < initial.length; state++)
				sensitivity[output][state][parameter] = (perturbed[output][state]-values[output][state])/step;
		}
		return new SensitivityResult(values, sensitivity);
	}
}
