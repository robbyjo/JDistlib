/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Chart-neutral immutable dataset suitable for SVG, JSON, CSV, or UI adapters. */
public final class ChartSpec {
	public enum Type { LINE, SCATTER, BAR }
	public static final class Series {
		private final String name;
		private final double[] x;
		private final double[] y;
		public Series(String name, double[] x, double[] y) {
			if (name == null || x == null || y == null || x.length != y.length)
				throw new IllegalArgumentException("series name and matching x/y data are required");
			this.name = name; this.x = x.clone(); this.y = y.clone();
		}
		public String name() { return name; }
		public double[] x() { return x.clone(); }
		public double[] y() { return y.clone(); }
		public int size() { return x.length; }
		/** Returns one horizontal coordinate without copying the series. */
		public double xAt(int index) { return x[index]; }
		/** Returns one vertical coordinate without copying the series. */
		public double yAt(int index) { return y[index]; }
	}
	private final String title;
	private final String xLabel;
	private final String yLabel;
	private final Type type;
	private final List<Series> series;

	public ChartSpec(String title, String xLabel, String yLabel, Type type,
			List<Series> series) {
		if (title == null || xLabel == null || yLabel == null || type == null
				|| series == null || series.isEmpty())
			throw new IllegalArgumentException("chart metadata and series are required");
		this.title = title; this.xLabel = xLabel; this.yLabel = yLabel; this.type = type;
		this.series = Collections.unmodifiableList(new ArrayList<Series>(series));
	}
	public String title() { return title; }
	public String xLabel() { return xLabel; }
	public String yLabel() { return yLabel; }
	public Type type() { return type; }
	public List<Series> series() { return series; }
	public String toJson() { return InferenceGraphExport.toJson(this); }
	public String toCsv() { return InferenceGraphExport.toCsv(this); }
	public String toSvg(int width, int height) {
		return InferenceGraphExport.toSvg(this, width, height);
	}
}
