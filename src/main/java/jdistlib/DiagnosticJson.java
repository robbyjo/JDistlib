/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.List;

import jdistlib.math.IntegrationStabilityResult;

/** Dependency-free RFC 8259 serialization for numerical diagnostic reports. */
public final class DiagnosticJson {
	private DiagnosticJson() {}

	public static String toJson(DiagnosticFinding finding) {
		if (finding == null) throw new IllegalArgumentException("finding must not be null");
		StringBuilder json = new StringBuilder(192);
		append(json, finding);
		return json.toString();
	}

	public static String toJson(FunctionAnalysis analysis) {
		if (analysis == null) throw new IllegalArgumentException("analysis must not be null");
		StringBuilder json = new StringBuilder(4096);
		json.append('{');
		field(json, "schemaVersion", 1).append(',');
		field(json, "type", "functionAnalysis").append(',');
		field(json, "suitableForConstruction",
				analysis.isSuitableForConstruction()).append(',');
		field(json, "hasErrors", analysis.hasErrors()).append(',');
		field(json, "hasWarnings", analysis.hasWarnings()).append(',');
		field(json, "sampledPoints", analysis.getSampledPoints()).append(',');
		field(json, "randomizedSampledPoints",
				analysis.getRandomizedSampledPoints()).append(',');
		field(json, "randomSeed", analysis.getRandomSeed()).append(',');
		field(json, "minimumPositiveValue",
				analysis.getMinimumPositiveValue()).append(',');
		field(json, "maximumValue", analysis.getMaximumValue()).append(',');
		quote(json, "suggestedBreakpoints").append(':');
		append(json, analysis.getSuggestedBreakpoints());
		json.append(',');
		quote(json, "findings").append(':');
		appendFindings(json, analysis.getFindings());
		json.append(',');
		quote(json, "normalizationStability").append(':');
		append(json, analysis.getNormalizationStability());
		json.append('}');
		return json.toString();
	}

	public static String toJson(DistributionAnalysis analysis) {
		if (analysis == null) throw new IllegalArgumentException("analysis must not be null");
		StringBuilder json = new StringBuilder(8192);
		json.append('{');
		field(json, "schemaVersion", 1).append(',');
		field(json, "type", "distributionAnalysis").append(',');
		field(json, "hasErrors", analysis.hasErrors()).append(',');
		field(json, "hasWarnings", analysis.hasWarnings()).append(',');
		field(json, "normalizationRelativeError",
				analysis.getNormalizationRelativeError()).append(',');
		field(json, "maximumTailDisagreement",
				analysis.getMaximumTailDisagreement()).append(',');
		field(json, "maximumQuantileRoundTripError",
				analysis.getMaximumQuantileRoundTripError()).append(',');
		field(json, "mean", analysis.getMean()).append(',');
		field(json, "variance", analysis.getVariance()).append(',');
		field(json, "firstAbsoluteMoment",
				analysis.getFirstAbsoluteMoment()).append(',');
		field(json, "secondAbsoluteMoment",
				analysis.getSecondAbsoluteMoment()).append(',');
		field(json, "momentsStable", analysis.areMomentsStable()).append(',');
		quote(json, "absoluteMoments").append(':').append('[');
		List<AbsoluteMomentAnalysis> moments = analysis.getAbsoluteMoments();
		for (int i = 0; i < moments.size(); i++) {
			if (i > 0) json.append(',');
			append(json, moments.get(i));
		}
		json.append(']').append(',');
		quote(json, "findings").append(':');
		appendFindings(json, analysis.getFindings());
		json.append('}');
		return json.toString();
	}

	public static String toJson(NumericalDistributionBuildResult result) {
		if (result == null) throw new IllegalArgumentException("result must not be null");
		StringBuilder json = new StringBuilder(4096);
		json.append('{');
		field(json, "schemaVersion", 1).append(',');
		field(json, "type", "distributionBuildResult").append(',');
		field(json, "canBuild", result.canBuild()).append(',');
		quote(json, "analysis").append(':').append(toJson(result.getAnalysis()));
		json.append(',');
		Throwable failure = result.getFailure();
		field(json, "failureType", failure == null ? null
				: failure.getClass().getName()).append(',');
		field(json, "failureMessage", failure == null ? null : failure.getMessage());
		json.append('}');
		return json.toString();
	}

	private static void append(StringBuilder json, DiagnosticFinding finding) {
		json.append('{');
		field(json, "severity", finding.getSeverity().name()).append(',');
		field(json, "code", finding.getCode()).append(',');
		field(json, "message", finding.getMessage()).append(',');
		field(json, "x", finding.getX());
		json.append('}');
	}

	private static void appendFindings(StringBuilder json,
			List<DiagnosticFinding> findings) {
		json.append('[');
		for (int i = 0; i < findings.size(); i++) {
			if (i > 0) json.append(',');
			append(json, findings.get(i));
		}
		json.append(']');
	}

	private static void append(StringBuilder json, AbsoluteMomentAnalysis moment) {
		json.append('{');
		field(json, "order", moment.getOrder()).append(',');
		field(json, "splitPoint", moment.getSplitPoint()).append(',');
		field(json, "leftValue", moment.getLeftValue()).append(',');
		field(json, "rightValue", moment.getRightValue()).append(',');
		field(json, "value", moment.getValue()).append(',');
		field(json, "leftStable", moment.isLeftStable()).append(',');
		field(json, "rightStable", moment.isRightStable()).append(',');
		field(json, "stable", moment.isStable()).append(',');
		quote(json, "leftStability").append(':');
		append(json, moment.getLeftStability());
		json.append(',');
		quote(json, "rightStability").append(':');
		append(json, moment.getRightStability());
		json.append('}');
	}

	private static void append(StringBuilder json,
			IntegrationStabilityResult stability) {
		json.append(stability == null ? "null" : stability.toJson());
	}

	private static void append(StringBuilder json, double[] values) {
		json.append('[');
		for (int i = 0; i < values.length; i++) {
			if (i > 0) json.append(',');
			number(json, values[i]);
		}
		json.append(']');
	}

	private static StringBuilder field(StringBuilder json, String name, String value) {
		quote(json, name).append(':');
		return value == null ? json.append("null") : quote(json, value);
	}

	private static StringBuilder field(StringBuilder json, String name, double value) {
		quote(json, name).append(':');
		return number(json, value);
	}

	private static StringBuilder field(StringBuilder json, String name, long value) {
		quote(json, name).append(':').append(value);
		return json;
	}

	private static StringBuilder field(StringBuilder json, String name, boolean value) {
		quote(json, name).append(':').append(value);
		return json;
	}

	private static StringBuilder number(StringBuilder json, double value) {
		return Double.isFinite(value) ? json.append(Double.toString(value))
				: json.append("null");
	}

	private static StringBuilder quote(StringBuilder json, String value) {
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
