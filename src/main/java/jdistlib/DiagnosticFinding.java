/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** One evidence-based finding produced by a numerical distribution analyzer. */
public final class DiagnosticFinding {
	public enum Severity { INFO, WARNING, ERROR }

	private final Severity severity;
	private final String code;
	private final String message;
	private final double x;

	public DiagnosticFinding(Severity severity, String code, String message) {
		this(severity, code, message, Double.NaN);
	}

	public DiagnosticFinding(Severity severity, String code, String message,
			double x) {
		if (severity == null || code == null || message == null) {
			throw new IllegalArgumentException("finding fields must not be null");
		}
		this.severity = severity;
		this.code = code;
		this.message = message;
		this.x = x;
	}

	public Severity getSeverity() { return severity; }
	public String getCode() { return code; }
	public String getMessage() { return message; }
	public double getX() { return x; }

	@Override public String toString() {
		return severity + "[" + code + "] " + message
				+ (Double.isNaN(x) ? "" : " (x=" + x + ")");
	}

	/** Returns this finding as an RFC 8259 JSON object. */
	public String toJson() { return DiagnosticJson.toJson(this); }
}
