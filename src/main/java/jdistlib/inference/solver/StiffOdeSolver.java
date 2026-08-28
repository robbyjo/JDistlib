/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Adaptive A-stable BDF1 integrator for stiff first-order ODE systems. */
public final class StiffOdeSolver {
	private StiffOdeSolver() {}

	/** Stiff integration controls. */
	public static final class Options {
		public final double relativeTolerance, absoluteTolerance, initialStep;
		public final int maximumSteps;
		public final AlgebraicSolver.Options nonlinearOptions;
		public Options(double relativeTolerance, double absoluteTolerance, double initialStep,
				int maximumSteps, AlgebraicSolver.Options nonlinearOptions) {
			if (!(relativeTolerance > 0) || !(absoluteTolerance > 0) || !(initialStep > 0)
					|| maximumSteps < 1 || nonlinearOptions == null)
				throw new IllegalArgumentException("positive stiff ODE controls required");
			this.relativeTolerance = relativeTolerance; this.absoluteTolerance = absoluteTolerance;
			this.initialStep = initialStep; this.maximumSteps = maximumSteps;
			this.nonlinearOptions = nonlinearOptions;
		}
		public static Options defaults() {
			return new Options(1e-6, 1e-9, 1e-3, 200000,
					new AlgebraicSolver.Options(1e-11, 1e-6, 50));
		}
	}

	/** Integrates to each strictly increasing output time with step-doubling error control. */
	public static double[][] integrate(OdeSystem system, double[] initial, double initialTime,
			double[] times, double[] parameters, double[] data, Options options) {
		if (system == null || initial == null || times == null || options == null)
			throw new NullPointerException("system, state, times, and options are required");
		double[] state = initial.clone(); double time = initialTime, step = options.initialStep;
		double[] p = parameters == null ? new double[0] : parameters.clone();
		double[] d = data == null ? new double[0] : data.clone();
		double[][] result = new double[times.length][state.length]; int steps = 0;
		for (int output = 0; output < times.length; output++) {
			double target = times[output];
			if (!(target > time)) throw new IllegalArgumentException("stiff ODE times must be strictly increasing");
			while (time < target) {
				if (++steps > options.maximumSteps) throw new ArithmeticException("stiff ODE maximum steps exceeded");
				step = Math.min(step, target-time);
				double[] full = implicitStep(system, state, time, step, p, d, options.nonlinearOptions);
				double[] half = implicitStep(system, state, time, step/2, p, d, options.nonlinearOptions);
				half = implicitStep(system, half, time+step/2, step/2, p, d, options.nonlinearOptions);
				double error = 0;
				for (int i = 0; i < state.length; i++) {
					double scale = options.absoluteTolerance + options.relativeTolerance
							* Math.max(Math.abs(state[i]), Math.abs(half[i]));
					error = Math.max(error, Math.abs(half[i]-full[i])/scale);
				}
				if (error <= 1.0) { state = half; time += step; }
				double factor = error == 0 ? 3.0 : Math.max(.2, Math.min(3.0, .9*Math.pow(error, -.5)));
				step *= factor;
				if (!(step > Math.ulp(Math.max(1.0, Math.abs(time)))))
					throw new ArithmeticException("stiff ODE step size underflow");
			}
			result[output] = state.clone();
		}
		return result;
	}

	private static double[] implicitStep(OdeSystem system, double[] previous, double time,
			double step, double[] parameters, double[] data, AlgebraicSolver.Options options) {
		AlgebraicSystem equation = (candidate, ignoredParameters, ignoredData, residual) -> {
			double[] derivative = new double[candidate.length];
			system.derivatives(time+step, candidate, parameters, data, derivative);
			for (int i = 0; i < candidate.length; i++)
				residual[i] = candidate[i]-previous[i]-step*derivative[i];
		};
		return AlgebraicSolver.solve(equation, previous, null, null, options).solution();
	}
}
