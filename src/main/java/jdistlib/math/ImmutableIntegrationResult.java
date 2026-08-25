/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

/**
 * Immutable modern integration result. Unlike {@link IntegrationResult}, this
 * snapshot does not retain the user callback or a mutable exception object.
 */
public final class ImmutableIntegrationResult {
	private final double value;
	private final double absoluteError;
	private final int evaluationCount;
	private final int subdivisions;
	private final IntegrationStatus status;
	private final int legacyStatusCode;
	private final double failureX;
	private final String causeType;
	private final String causeMessage;
	private final String detail;
	private final CallbackProfile callbackProfile;

	ImmutableIntegrationResult(IntegrationResult result) {
		value = result.result;
		absoluteError = result.abserr;
		evaluationCount = result.neval;
		subdivisions = result.last;
		status = IntegrationStatus.fromCode(result.ier);
		legacyStatusCode = result.ier;
		failureX = result.failureX;
		causeType = result.cause == null ? null : result.cause.getClass().getName();
		causeMessage = result.cause == null ? null : result.cause.getMessage();
		detail = result.detail;
		callbackProfile = result.getCallbackProfile();
	}

	public double getValue() { return value; }
	public double getAbsoluteError() { return absoluteError; }
	public int getEvaluationCount() { return evaluationCount; }
	public int getSubdivisions() { return subdivisions; }
	public IntegrationStatus getStatus() { return status; }
	public int getLegacyStatusCode() { return legacyStatusCode; }
	public double getFailureX() { return failureX; }
	public String getCauseType() { return causeType; }
	public String getCauseMessage() { return causeMessage; }
	public String getDetail() { return detail; }
	public CallbackProfile getCallbackProfile() { return callbackProfile; }
	public boolean isSuccess() { return legacyStatusCode == 0; }
	public String message() { return status.getMessage(); }
	public String detailedMessage() {
		return detail == null || detail.length() == 0
				? message() : message() + ": " + detail;
	}

	/** Returns an RFC 8259 JSON diagnostic record. */
	public String toJson() { return IntegrationJson.toJson(this); }
}
