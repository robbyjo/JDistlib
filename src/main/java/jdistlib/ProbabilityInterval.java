/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Immutable interval containing a requested probability under a stated rule. */
public final class ProbabilityInterval {
	private final double lower;
	private final double upper;
	private final double probability;
	private final String method;

	public ProbabilityInterval(double lower, double upper, double probability,
			String method) {
		if (Double.isNaN(lower) || Double.isNaN(upper) || lower > upper) {
			throw new IllegalArgumentException("interval bounds are invalid");
		}
		if (!(probability > 0.0 && probability < 1.0)) {
			throw new IllegalArgumentException("probability must lie between zero and one");
		}
		if (method == null) throw new IllegalArgumentException("method must not be null");
		this.lower = lower;
		this.upper = upper;
		this.probability = probability;
		this.method = method;
	}

	public double getLower() { return lower; }
	public double getUpper() { return upper; }
	public double getProbability() { return probability; }
	public String getMethod() { return method; }
}
