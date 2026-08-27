/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Dependency-free versioned JSON serialization for inference diagnostics. */
public final class McmcJson {
	private McmcJson() {}
	public static String toJson(McmcDiagnosticReport report) {
		if (report == null) return "null";
		StringBuilder json = new StringBuilder();
		json.append('{'); string(json, "schema").append(':'); string(json, "jdistlib.mcmc-diagnostics/1");
		json.append(','); string(json, "chains").append(':').append(report.chains());
		json.append(','); string(json, "drawsPerChain").append(':').append(report.drawsPerChain());
		json.append(','); string(json, "reliable").append(':').append(report.reliable());
		json.append(','); string(json, "parameters").append(':').append('[');
		for (int i = 0; i < report.parameters().size(); i++) {
			if (i > 0) json.append(',');
			ParameterDiagnostics p = report.parameters().get(i);
			json.append('{'); field(json, "name", p.name()).append(',');
			numberField(json, "mean", p.mean()).append(',');
			numberField(json, "sd", p.standardDeviation()).append(',');
			numberField(json, "median", p.median()).append(',');
			numberField(json, "q025", p.lowerQuantile()).append(',');
			numberField(json, "q975", p.upperQuantile()).append(',');
			numberField(json, "rHat", p.rHat()).append(',');
			numberField(json, "bulkEss", p.bulkEffectiveSampleSize()).append(',');
			numberField(json, "tailEss", p.tailEffectiveSampleSize()).append(',');
			numberField(json, "mcse", p.monteCarloStandardError()).append(',');
			string(json, "reliable").append(':').append(p.reliable()).append('}');
		}
		json.append(']').append(','); string(json, "sampler").append(':').append('{');
		SamplerDiagnostics s = report.sampler();
		numberField(json, "meanAcceptance", s.meanAcceptanceProbability()).append(',');
		string(json, "divergences").append(':').append(s.divergences()).append(',');
		string(json, "treeDepthSaturations").append(':').append(s.treeDepthSaturations()).append(',');
		string(json, "maximumTreeDepth").append(':').append(s.maximumTreeDepth()).append(',');
		numberField(json, "eBfmi", s.energyBayesianFractionMissingInformation()).append(',');
		string(json, "numericalFailures").append(':').append(s.numericalFailures()).append('}');
		json.append(','); string(json, "warnings").append(':').append('[');
		for (int i = 0; i < report.warnings().size(); i++) {
			if (i > 0) json.append(','); string(json, report.warnings().get(i));
		}
		return json.append(']').append('}').toString();
	}
	private static StringBuilder numberField(StringBuilder json, String name, double value) {
		string(json, name).append(':');
		return json.append(Double.isFinite(value) ? Double.toString(value) : "null");
	}
	private static StringBuilder field(StringBuilder json, String name, String value) {
		string(json, name).append(':'); return string(json, value);
	}
	static StringBuilder string(StringBuilder json, String value) {
		json.append('"');
		for (int i = 0; i < value.length(); i++) {
			char c = value.charAt(i);
			if (c == '"' || c == '\\') json.append('\\').append(c);
			else if (c == '\n') json.append("\\n");
			else if (c == '\r') json.append("\\r");
			else if (c == '\t') json.append("\\t");
			else if (c < 32) json.append(String.format("\\u%04x", (int) c));
			else json.append(c);
		}
		return json.append('"');
	}
}
