/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

/** Stable, typed interpretation of legacy QUADPACK-compatible status codes. */
public enum IntegrationStatus {
	SUCCESS(0, "OK"),
	MAXIMUM_SUBDIVISIONS(1, "maximum number of subdivisions reached"),
	ROUNDOFF(2, "roundoff error was detected"),
	BAD_INTEGRAND_BEHAVIOUR(3, "extremely bad integrand behaviour"),
	EXTRAPOLATION_ROUNDOFF(4,
			"roundoff error is detected in the extrapolation table"),
	PROBABLY_DIVERGENT(5, "the integral is probably divergent"),
	INVALID_INPUT(6, "the input is invalid"),
	CALLBACK_FAILED(7, "the integrand callback failed"),
	CANCELLED(8, "integration was cancelled"),
	EVALUATION_BUDGET_EXHAUSTED(9,
			"the function evaluation budget was exhausted"),
	NON_FINITE_VALUE(10, "the integrand returned a non-finite value"),
	CALLBACK_TIME_LIMIT_EXCEEDED(11, "the callback time limit was exceeded"),
	UNKNOWN(Integer.MIN_VALUE, "unknown integration status");

	private final int code;
	private final String message;

	IntegrationStatus(int code, String message) {
		this.code = code;
		this.message = message;
	}

	public int getCode() { return code; }
	public String getMessage() { return message; }

	/** Returns the typed status corresponding to a legacy integer code. */
	public static IntegrationStatus fromCode(int code) {
		for (IntegrationStatus status : values()) {
			if (status.code == code) return status;
		}
		return UNKNOWN;
	}
}
