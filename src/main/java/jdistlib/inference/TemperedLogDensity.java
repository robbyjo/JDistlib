/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Log density split into an untempered base (usually the prior) and likelihood. */
public interface TemperedLogDensity extends LogDensity {
	double baseLogDensity(double[] state);
	double temperedLogDensity(double[] state);
	@Override default double logDensity(double[] state) {
		return baseLogDensity(state) + temperedLogDensity(state);
	}
}
