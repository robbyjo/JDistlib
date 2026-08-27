/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Self-contained headless HTML report composed from diagnostic data and SVGs. */
public final class InferenceHtmlReport {
	private InferenceHtmlReport() {}
	public static String render(String title, McmcDiagnosticReport report,
			ModelGraph graph, ChartSpec... charts) {
		if (title == null || report == null || charts == null)
			throw new IllegalArgumentException("title, report and charts are required");
		StringBuilder html = new StringBuilder("<!doctype html><html><head><meta charset=\"utf-8\">")
				.append("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">")
				.append("<title>").append(InferenceGraphExport.escapeXml(title)).append("</title>")
				.append("<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}")
				.append("table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;padding:.45rem;text-align:right}")
				.append("th:first-child,td:first-child{text-align:left}.warning{color:#8b2d20}.chart{margin:1.5rem 0;border:1px solid #ddd}</style></head><body><h1>")
				.append(InferenceGraphExport.escapeXml(title)).append("</h1><p>Chains: ")
				.append(report.chains()).append("; retained draws per chain: ").append(report.drawsPerChain())
				.append("; overall status: <strong>").append(report.reliable() ? "reliable" : "needs attention")
				.append("</strong>.</p>");
		if (!report.warnings().isEmpty()) {
			html.append("<ul class=\"warning\">");
			for (String warning : report.warnings()) html.append("<li>")
					.append(InferenceGraphExport.escapeXml(warning)).append("</li>");
			html.append("</ul>");
		}
		SamplerDiagnostics sampler = report.sampler();
		html.append("<h2>Sampler health</h2><table><thead><tr><th>Mean acceptance</th><th>Divergences</th><th>Depth saturations</th><th>Maximum depth</th><th>Minimum E-BFMI</th><th>Failures</th></tr></thead><tbody><tr><td>")
				.append(sampler.meanAcceptanceProbability()).append("</td><td>")
				.append(sampler.divergences()).append("</td><td>")
				.append(sampler.treeDepthSaturations()).append("</td><td>")
				.append(sampler.maximumTreeDepth()).append("</td><td>")
				.append(sampler.energyBayesianFractionMissingInformation()).append("</td><td>")
				.append(sampler.numericalFailures()).append("</td></tr></tbody></table><h2>Parameter summaries</h2>");
		html.append("<table><thead><tr><th>Parameter</th><th>Mean</th><th>SD</th><th>2.5%</th><th>50%</th><th>97.5%</th><th>R-hat</th><th>Bulk ESS</th><th>Tail ESS</th><th>MCSE</th></tr></thead><tbody>");
		for (ParameterDiagnostics p : report.parameters()) html.append("<tr><td>")
				.append(InferenceGraphExport.escapeXml(p.name())).append("</td><td>").append(p.mean())
				.append("</td><td>").append(p.standardDeviation()).append("</td><td>")
				.append(p.lowerQuantile()).append("</td><td>").append(p.median())
				.append("</td><td>").append(p.upperQuantile()).append("</td><td>")
				.append(p.rHat()).append("</td><td>").append(p.bulkEffectiveSampleSize())
				.append("</td><td>").append(p.tailEffectiveSampleSize()).append("</td><td>")
				.append(p.monteCarloStandardError()).append("</td></tr>");
		html.append("</tbody></table>");
		for (ChartSpec chart : charts) html.append("<div class=\"chart\">")
				.append(chart.toSvg(900, 420)).append("</div>");
		if (graph != null) html.append("<h2>Model graph (Graphviz DOT)</h2><pre>")
				.append(InferenceGraphExport.escapeXml(ModelGraphExport.toDot(graph))).append("</pre>");
		return html.append("</body></html>").toString();
	}
}
