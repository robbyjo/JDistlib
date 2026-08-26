/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Controls rank-based initialization and optional likelihood refinement. */
public final class CopulaFitOptions {
	public enum Method { KENDALL_TAU, MAXIMUM_LIKELIHOOD }

	private final Method method;
	private final double minimumDegreesOfFreedom;
	private final double maximumDegreesOfFreedom;
	private final int optimizationIterations;

	public CopulaFitOptions() { this(Method.MAXIMUM_LIKELIHOOD, 0.5, 100.0, 56); }

	private CopulaFitOptions(Method method, double minimumDegreesOfFreedom,
			double maximumDegreesOfFreedom, int optimizationIterations) {
		if (method == null || !(minimumDegreesOfFreedom > 0.0)
				|| !(maximumDegreesOfFreedom > minimumDegreesOfFreedom)
				|| !Double.isFinite(minimumDegreesOfFreedom)
				|| !Double.isFinite(maximumDegreesOfFreedom)
				|| optimizationIterations < 8)
			throw new IllegalArgumentException("invalid copula fit options");
		this.method = method;
		this.minimumDegreesOfFreedom = minimumDegreesOfFreedom;
		this.maximumDegreesOfFreedom = maximumDegreesOfFreedom;
		this.optimizationIterations = optimizationIterations;
	}

	public Method getMethod() { return method; }
	public double getMinimumDegreesOfFreedom() { return minimumDegreesOfFreedom; }
	public double getMaximumDegreesOfFreedom() { return maximumDegreesOfFreedom; }
	public int getOptimizationIterations() { return optimizationIterations; }

	public CopulaFitOptions withMethod(Method value) {
		return new CopulaFitOptions(value, minimumDegreesOfFreedom,
				maximumDegreesOfFreedom, optimizationIterations);
	}

	public CopulaFitOptions withDegreesOfFreedomRange(double minimum, double maximum) {
		return new CopulaFitOptions(method, minimum, maximum, optimizationIterations);
	}

	public CopulaFitOptions withOptimizationIterations(int iterations) {
		return new CopulaFitOptions(method, minimumDegreesOfFreedom,
				maximumDegreesOfFreedom, iterations);
	}
}
