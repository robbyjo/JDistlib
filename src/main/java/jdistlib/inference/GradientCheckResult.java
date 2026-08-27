/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable comparison between supplied and finite-difference gradients. */
public final class GradientCheckResult {
	private final boolean passed;
	private final double maximumAbsoluteError;
	private final double maximumRelativeError;
	private final int worstCoordinate;
	private final String message;

	GradientCheckResult(boolean passed, double maximumAbsoluteError,
			double maximumRelativeError, int worstCoordinate, String message) {
		this.passed = passed;
		this.maximumAbsoluteError = maximumAbsoluteError;
		this.maximumRelativeError = maximumRelativeError;
		this.worstCoordinate = worstCoordinate;
		this.message = message;
	}

	public boolean passed() { return passed; }
	public double maximumAbsoluteError() { return maximumAbsoluteError; }
	public double maximumRelativeError() { return maximumRelativeError; }
	public int worstCoordinate() { return worstCoordinate; }
	public String message() { return message; }
	@Override public String toString() {
		return "GradientCheckResult{passed=" + passed + ", worstCoordinate="
				+ worstCoordinate + ", maximumAbsoluteError=" + maximumAbsoluteError
				+ ", maximumRelativeError=" + maximumRelativeError + ", message='"
				+ message + "'}";
	}
}
