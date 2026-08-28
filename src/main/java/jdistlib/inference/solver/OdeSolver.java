/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.solver;

/** Adaptive Dormand-Prince 5(4) ODE integration at requested output times. */
public final class OdeSolver {
	private OdeSolver() {}

	/** Integration controls. */
	public static final class Options {
		public final double relativeTolerance, absoluteTolerance, initialStep;
		public final int maximumSteps;
		public Options(double relativeTolerance, double absoluteTolerance,
				double initialStep, int maximumSteps) {
			if (!(relativeTolerance > 0) || !(absoluteTolerance > 0)
					|| !(initialStep > 0) || maximumSteps < 1)
				throw new IllegalArgumentException("positive ODE controls required");
			this.relativeTolerance = relativeTolerance; this.absoluteTolerance = absoluteTolerance;
			this.initialStep = initialStep; this.maximumSteps = maximumSteps;
		}
		public static Options defaults() { return new Options(1e-8, 1e-10, 1e-2, 100000); }
	}

	/** Integrates from {@code initialTime}; output row {@code i} corresponds to {@code times[i]}. */
	public static double[][] integrate(OdeSystem system, double[] initial, double initialTime,
			double[] times, double[] parameters, double[] data, Options options) {
		if (system == null || initial == null || times == null || options == null)
			throw new NullPointerException("system, state, times, and options are required");
		if (initial.length == 0) throw new IllegalArgumentException("non-empty ODE state required");
		double[] state = initial.clone(); double time = initialTime;
		double[] p = parameters == null ? new double[0] : parameters.clone();
		double[] d = data == null ? new double[0] : data.clone();
		double[][] result = new double[times.length][initial.length];
		double step = options.initialStep; int steps = 0;
		for (int output = 0; output < times.length; output++) {
			double target = times[output];
			if (!(target > time)) throw new IllegalArgumentException("ODE times must be strictly increasing");
			while (time < target) {
				if (++steps > options.maximumSteps) throw new ArithmeticException("ODE maximum steps exceeded");
				step = Math.min(step, target - time);
				Step trial = dormandPrince(system, time, state, step, p, d,
						options.absoluteTolerance, options.relativeTolerance);
				if (trial.error <= 1.0) { time += step; state = trial.state; }
				double factor = trial.error == 0 ? 5.0
						: Math.max(0.2, Math.min(5.0, 0.9 * Math.pow(trial.error, -0.2)));
				step *= factor;
				if (!(step > Math.ulp(Math.max(1.0, Math.abs(time)))))
					throw new ArithmeticException("ODE step size underflow");
			}
			result[output] = state.clone();
		}
		return result;
	}

	/**
	 * Integrates the state and forward parameter-sensitivity equations.
	 * Jacobians of the caller's system are estimated with scaled finite
	 * differences, while the augmented sensitivity system uses the same adaptive
	 * Dormand-Prince error control as the state.
	 */
	public static SensitivityResult integrateWithSensitivities(OdeSystem system,
			double[] initial, double initialTime, double[] times, double[] parameters,
			double[] data, Options options) {
		if (parameters == null) throw new IllegalArgumentException("parameters are required for sensitivities");
		final int states = initial.length, parameterCount = parameters.length;
		double[] augmented = new double[states + states * parameterCount];
		System.arraycopy(initial, 0, augmented, 0, states);
		OdeSystem sensitivitySystem = (time, combined, ignored, observed, derivative) -> {
			double[] state = new double[states]; System.arraycopy(combined, 0, state, 0, states);
			double[] base = new double[states]; system.derivatives(time, state, parameters, observed, base);
			System.arraycopy(base, 0, derivative, 0, states);
			double[][] stateJacobian = new double[states][states];
			for (int column = 0; column < states; column++) {
				double[] shifted = state.clone(); double step = 1e-6 * Math.max(1.0, Math.abs(state[column]));
				shifted[column] += step; double[] value = new double[states];
				system.derivatives(time, shifted, parameters, observed, value);
				for (int row = 0; row < states; row++) stateJacobian[row][column] = (value[row]-base[row])/step;
			}
			double[][] parameterJacobian = new double[states][parameterCount];
			for (int column = 0; column < parameterCount; column++) {
				double[] shifted = parameters.clone(); double step = 1e-6 * Math.max(1.0, Math.abs(parameters[column]));
				shifted[column] += step; double[] value = new double[states];
				system.derivatives(time, state, shifted, observed, value);
				for (int row = 0; row < states; row++) parameterJacobian[row][column] = (value[row]-base[row])/step;
			}
			for (int row = 0; row < states; row++) for (int parameter = 0; parameter < parameterCount; parameter++) {
				double value = parameterJacobian[row][parameter];
				for (int column = 0; column < states; column++)
					value += stateJacobian[row][column] * combined[states + column*parameterCount + parameter];
				derivative[states + row*parameterCount + parameter] = value;
			}
		};
		double[][] combined = integrate(sensitivitySystem, augmented, initialTime, times,
				new double[0], data, options);
		double[][] values = new double[times.length][states];
		double[][][] sensitivities = new double[times.length][states][parameterCount];
		for (int output = 0; output < times.length; output++) for (int state = 0; state < states; state++) {
			values[output][state] = combined[output][state];
			for (int parameter = 0; parameter < parameterCount; parameter++)
				sensitivities[output][state][parameter] = combined[output][states + state*parameterCount + parameter];
		}
		return new SensitivityResult(values, sensitivities);
	}

	private static final class Step {
		final double[] state; final double error;
		Step(double[] state, double error) { this.state = state; this.error = error; }
	}
	private static Step dormandPrince(OdeSystem f, double t, double[] y, double h,
			double[] p, double[] d, double absoluteTolerance, double relativeTolerance) {
		int n = y.length; double[][] k = new double[7][n]; double[] work = new double[n];
		f.derivatives(t, y, p, d, k[0]);
		stage(f, t + h/5, y, h, p, d, k, work, 1, new double[] {1.0/5});
		stage(f, t + 3*h/10, y, h, p, d, k, work, 2, new double[] {3.0/40, 9.0/40});
		stage(f, t + 4*h/5, y, h, p, d, k, work, 3, new double[] {44.0/45, -56.0/15, 32.0/9});
		stage(f, t + 8*h/9, y, h, p, d, k, work, 4,
				new double[] {19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729});
		stage(f, t + h, y, h, p, d, k, work, 5,
				new double[] {9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656});
		stage(f, t + h, y, h, p, d, k, work, 6,
				new double[] {35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84});
		double[] high = new double[n]; double error = 0;
		double[] b5 = {35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0};
		double[] b4 = {5179.0/57600, 0, 7571.0/16695, 393.0/640,
				-92097.0/339200, 187.0/2100, 1.0/40};
		for (int i = 0; i < n; i++) {
			high[i] = y[i]; double low = y[i];
			for (int j = 0; j < 7; j++) { high[i] += h*b5[j]*k[j][i]; low += h*b4[j]*k[j][i]; }
			double scale = absoluteTolerance + relativeTolerance*Math.max(Math.abs(y[i]), Math.abs(high[i]));
			double normalized = (high[i] - low) / scale; error += normalized * normalized;
		}
		return new Step(high, Math.sqrt(error/n));
	}
	private static void stage(OdeSystem f, double t, double[] y, double h, double[] p,
			double[] d, double[][] k, double[] work, int output, double[] coefficients) {
		for (int i = 0; i < y.length; i++) {
			work[i] = y[i];
			for (int j = 0; j < coefficients.length; j++) work[i] += h*coefficients[j]*k[j][i];
		}
		f.derivatives(t, work, p, d, k[output]);
	}
}
