/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

/** Dependency-free JSON serialization for integration diagnostics. */
public final class IntegrationJson {
	private IntegrationJson() {}

	public static String toJson(IntegrationResult result) {
		if (result == null) throw new IllegalArgumentException("result must not be null");
		return toJson(result.toImmutable());
	}

	public static String toJson(ImmutableIntegrationResult result) {
		if (result == null) throw new IllegalArgumentException("result must not be null");
		StringBuilder json = new StringBuilder(512);
		append(json, result);
		return json.toString();
	}

	public static String toJson(IntegrationStabilityResult result) {
		if (result == null) throw new IllegalArgumentException("result must not be null");
		StringBuilder json = new StringBuilder(1600);
		append(json, result);
		return json.toString();
	}

	static void append(StringBuilder json, ImmutableIntegrationResult result) {
		json.append('{');
		field(json, "schemaVersion", 1).append(',');
		field(json, "type", "integrationResult").append(',');
		field(json, "status", result.getStatus().name()).append(',');
		field(json, "statusCode", result.getLegacyStatusCode()).append(',');
		field(json, "success", result.isSuccess()).append(',');
		field(json, "value", result.getValue()).append(',');
		field(json, "absoluteError", result.getAbsoluteError()).append(',');
		field(json, "evaluationCount", result.getEvaluationCount()).append(',');
		field(json, "subdivisions", result.getSubdivisions()).append(',');
		field(json, "failureX", result.getFailureX()).append(',');
		field(json, "detail", result.getDetail()).append(',');
		field(json, "causeType", result.getCauseType()).append(',');
		field(json, "causeMessage", result.getCauseMessage()).append(',');
		quote(json, "callbackProfile").append(':');
		append(json, result.getCallbackProfile());
		json.append('}');
	}

	static void append(StringBuilder json, IntegrationStabilityResult result) {
		json.append('{');
		field(json, "schemaVersion", 1).append(',');
		field(json, "type", "integrationStability").append(',');
		field(json, "stable", result.isStable()).append(',');
		field(json, "maximumDiscrepancy", result.getMaximumDiscrepancy()).append(',');
		field(json, "allowedDiscrepancy", result.getAllowedDiscrepancy()).append(',');
		quote(json, "baseline").append(':');
		append(json, result.getBaseline().toImmutable());
		json.append(',');
		quote(json, "tightened").append(':');
		append(json, result.getTightened().toImmutable());
		json.append(',');
		quote(json, "split").append(':');
		append(json, result.getSplit().toImmutable());
		json.append('}');
	}

	private static void append(StringBuilder json, CallbackProfile profile) {
		json.append('{');
		field(json, "attemptedEvaluations", profile.getAttemptedEvaluations()).append(',');
		field(json, "completedEvaluations", profile.getCompletedEvaluations()).append(',');
		field(json, "totalCallbackNanos", profile.getTotalCallbackNanos()).append(',');
		field(json, "maximumCallbackNanos", profile.getMaximumCallbackNanos()).append(',');
		field(json, "averageCallbackNanos", profile.getAverageCallbackNanos()).append(',');
		field(json, "integrationWallNanos", profile.getIntegrationWallNanos());
		json.append('}');
	}

	static StringBuilder field(StringBuilder json, String name, String value) {
		quote(json, name).append(':');
		return value == null ? json.append("null") : quote(json, value);
	}

	static StringBuilder field(StringBuilder json, String name, double value) {
		quote(json, name).append(':');
		return Double.isFinite(value) ? json.append(Double.toString(value))
				: json.append("null");
	}

	static StringBuilder field(StringBuilder json, String name, long value) {
		quote(json, name).append(':').append(value);
		return json;
	}

	static StringBuilder field(StringBuilder json, String name, boolean value) {
		quote(json, name).append(':').append(value);
		return json;
	}

	static StringBuilder quote(StringBuilder json, String value) {
		json.append('"');
		for (int i = 0; i < value.length(); i++) {
			char c = value.charAt(i);
			switch (c) {
			case '"': json.append("\\\""); break;
			case '\\': json.append("\\\\"); break;
			case '\b': json.append("\\b"); break;
			case '\f': json.append("\\f"); break;
			case '\n': json.append("\\n"); break;
			case '\r': json.append("\\r"); break;
			case '\t': json.append("\\t"); break;
			default:
				if (c < 0x20) {
					String hex = Integer.toHexString(c);
					json.append("\\u0000", 0, 6 - hex.length()).append(hex);
				} else {
					json.append(c);
				}
			}
		}
		return json.append('"');
	}
}
