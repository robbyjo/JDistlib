/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Diagnostic code, quantitative evidence, and an actionable remediation. */
public final class HealthIssue {
	private final String code, evidence, remediation; private final HealthSeverity severity;
	HealthIssue(String code, HealthSeverity severity, String evidence, String remediation) { this.code = code; this.severity = severity; this.evidence = evidence; this.remediation = remediation; }
	public String code() { return code; } public HealthSeverity severity() { return severity; }
	public String evidence() { return evidence; } public String remediation() { return remediation; }
}
