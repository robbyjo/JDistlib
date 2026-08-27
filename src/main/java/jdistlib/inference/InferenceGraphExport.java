/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Locale;

/** Dependency-free JSON, tidy CSV, and SVG adapters for chart-neutral data. */
public final class InferenceGraphExport {
	private static final String[] COLORS = {"#315b7d", "#c65d21", "#3c8c62",
			"#8a5aa6", "#b08b22", "#2f8f9d", "#a3475b", "#555555"};
	private InferenceGraphExport() {}

	public static String toJson(ChartSpec chart) {
		if (chart == null) return "null";
		StringBuilder json = new StringBuilder();
		json.append('{'); field(json, "schema", "jdistlib.chart/1").append(',');
		field(json, "title", chart.title()).append(','); field(json, "xLabel", chart.xLabel()).append(',');
		field(json, "yLabel", chart.yLabel()).append(','); field(json, "type", chart.type().name()).append(',');
		McmcJson.string(json, "series").append(':').append('[');
		for (int s = 0; s < chart.series().size(); s++) {
			if (s > 0) json.append(','); ChartSpec.Series series = chart.series().get(s);
			json.append('{'); field(json, "name", series.name()).append(',');
			McmcJson.string(json, "points").append(':').append('[');
			for (int i = 0; i < series.size(); i++) {
				if (i > 0) json.append(',');
				json.append('[').append(Double.toString(series.xAt(i))).append(',')
						.append(Double.toString(series.yAt(i))).append(']');
			}
			json.append(']').append('}');
		}
		return json.append(']').append('}').toString();
	}

	public static String toCsv(ChartSpec chart) {
		if (chart == null) throw new IllegalArgumentException("chart is required");
		StringBuilder csv = new StringBuilder("series,x,y\n");
		for (ChartSpec.Series series : chart.series()) {
			for (int i = 0; i < series.size(); i++) csv.append(csv(series.name())).append(',')
					.append(Double.toString(series.xAt(i))).append(',')
					.append(Double.toString(series.yAt(i))).append('\n');
		}
		return csv.toString();
	}

	public static String toSvg(ChartSpec chart, int width, int height) {
		if (chart == null || width < 240 || height < 180)
			throw new IllegalArgumentException("chart and practical SVG dimensions are required");
		double[] bounds = bounds(chart);
		double left = 70.0, right = width - 24.0, top = 48.0, bottom = height - 58.0;
		StringBuilder svg = new StringBuilder();
		svg.append("<svg xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" width=\"").append(width)
				.append("\" height=\"").append(height).append("\" viewBox=\"0 0 ")
				.append(width).append(' ').append(height).append("\"><title>")
				.append(escapeXml(chart.title())).append("</title><desc>")
				.append(escapeXml(chart.type().name().toLowerCase(Locale.ROOT)
						+ " chart of " + chart.yLabel() + " by " + chart.xLabel()))
				.append("</desc>");
		svg.append("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");
		text(svg, width / 2.0, 24, chart.title(), "middle", 16);
		svg.append("<line x1=\"").append(left).append("\" y1=\"").append(bottom)
				.append("\" x2=\"").append(right).append("\" y2=\"").append(bottom)
				.append("\" stroke=\"#333\"/><line x1=\"").append(left).append("\" y1=\"")
				.append(top).append("\" x2=\"").append(left).append("\" y2=\"").append(bottom)
				.append("\" stroke=\"#333\"/>");
		text(svg, (left + right) / 2.0, height - 14, chart.xLabel(), "middle", 12);
		text(svg, 8, 18, chart.yLabel(), "start", 12);
		text(svg, left, bottom + 18, format(bounds[0]), "middle", 10);
		text(svg, right, bottom + 18, format(bounds[1]), "middle", 10);
		text(svg, left - 8, bottom, format(bounds[2]), "end", 10);
		text(svg, left - 8, top + 4, format(bounds[3]), "end", 10);
		for (int s = 0; s < chart.series().size(); s++) {
			ChartSpec.Series series = chart.series().get(s);
			String color = COLORS[s % COLORS.length];
			if (chart.type() == ChartSpec.Type.LINE) {
				svg.append("<polyline fill=\"none\" stroke=\"").append(color)
						.append("\" stroke-width=\"1.5\" points=\"");
				for (int i = 0; i < series.size(); i++)
					svg.append(scale(series.xAt(i), bounds[0], bounds[1], left, right))
							.append(',').append(scale(series.yAt(i), bounds[2], bounds[3], bottom, top)).append(' ');
				svg.append("\"/>");
			} else if (chart.type() == ChartSpec.Type.SCATTER) {
				for (int i = 0; i < series.size(); i++) svg.append("<circle cx=\"")
						.append(scale(series.xAt(i), bounds[0], bounds[1], left, right)).append("\" cy=\"")
						.append(scale(series.yAt(i), bounds[2], bounds[3], bottom, top))
						.append("\" r=\"2\" fill=\"").append(color).append("\" fill-opacity=\"0.55\"/>");
			} else {
				double barWidth = Math.max(1.0, (right - left) / Math.max(1, series.size() * chart.series().size()));
				for (int i = 0; i < series.size(); i++) {
					double center = scale(series.xAt(i), bounds[0], bounds[1], left, right)
							+ (s - (chart.series().size() - 1) / 2.0) * barWidth;
					double yValue = scale(series.yAt(i), bounds[2], bounds[3], bottom, top);
					svg.append("<rect x=\"").append(center - barWidth / 2.0).append("\" y=\"")
							.append(yValue).append("\" width=\"").append(barWidth)
							.append("\" height=\"").append(Math.max(0.0, bottom - yValue))
							.append("\" fill=\"").append(color).append("\" fill-opacity=\"0.7\"/>");
				}
			}
			text(svg, right - 4, top + 14 + 14 * s, series.name(), "end", 10, color);
		}
		return svg.append("</svg>").toString();
	}

	private static double[] bounds(ChartSpec chart) {
		double xmin = Double.POSITIVE_INFINITY, xmax = Double.NEGATIVE_INFINITY;
		double ymin = Double.POSITIVE_INFINITY, ymax = Double.NEGATIVE_INFINITY;
		for (ChartSpec.Series series : chart.series()) {
			for (int i = 0; i < series.size(); i++)
				if (Double.isFinite(series.xAt(i)) && Double.isFinite(series.yAt(i))) {
				xmin = Math.min(xmin, series.xAt(i)); xmax = Math.max(xmax, series.xAt(i));
				ymin = Math.min(ymin, series.yAt(i)); ymax = Math.max(ymax, series.yAt(i));
			}
		}
		if (!Double.isFinite(xmin)) return new double[] {0, 1, 0, 1};
		if (xmin == xmax) { xmin -= 0.5; xmax += 0.5; }
		if (ymin == ymax) { ymin -= 0.5; ymax += 0.5; }
		if (chart.type() == ChartSpec.Type.BAR && ymin > 0.0) ymin = 0.0;
		return new double[] {xmin, xmax, ymin, ymax};
	}
	private static double scale(double value, double low, double high, double outLow, double outHigh) {
		return outLow + (value - low) / (high - low) * (outHigh - outLow);
	}
	private static String format(double value) { return String.format(Locale.ROOT, "%.4g", value); }
	private static String csv(String value) { return "\"" + value.replace("\"", "\"\"") + "\""; }
	private static StringBuilder field(StringBuilder json, String name, String value) {
		McmcJson.string(json, name).append(':'); return McmcJson.string(json, value);
	}
	private static void text(StringBuilder svg, double x, double y, String value,
			String anchor, int size) { text(svg, x, y, value, anchor, size, "#222"); }
	private static void text(StringBuilder svg, double x, double y, String value,
			String anchor, int size, String color) {
		svg.append("<text x=\"").append(x).append("\" y=\"").append(y)
				.append("\" text-anchor=\"").append(anchor).append("\" font-family=\"sans-serif\" font-size=\"")
				.append(size).append("\" fill=\"").append(color).append("\">")
				.append(escapeXml(value)).append("</text>");
	}
	static String escapeXml(String value) {
		return value.replace("&", "&amp;").replace("<", "&lt;")
				.replace(">", "&gt;").replace("\"", "&quot;");
	}
}
