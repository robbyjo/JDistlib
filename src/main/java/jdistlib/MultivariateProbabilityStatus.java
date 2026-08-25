/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Terminal status of a numerical multivariate probability calculation. */
public enum MultivariateProbabilityStatus {
	/** The requested tolerance was met, or the answer was obtained exactly. */
	SUCCESS(0, "OK"),
	/** A usable estimate was produced, but its error estimate exceeds tolerance. */
	MAX_EVALUATIONS_REACHED(1, "maximum number of evaluations reached"),
	/** Parameters, bounds, options, or the random stream were invalid. */
	INVALID_INPUT(2, "invalid parameters or bounds");

	private final int code;
	private final String message;

	MultivariateProbabilityStatus(int code, String message) {
		this.code = code;
		this.message = message;
	}

	/** Stable legacy integer code. */
	public int code() {
		return code;
	}

	/** Human-readable status text. */
	public String message() {
		return message;
	}

	static MultivariateProbabilityStatus fromCode(int code) {
		switch (code) {
		case 0: return SUCCESS;
		case 1: return MAX_EVALUATIONS_REACHED;
		case 2: return INVALID_INPUT;
		default: throw new IllegalArgumentException("unknown status code: " + code);
		}
	}
}
