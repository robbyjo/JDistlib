/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Selected-coordinate columns read from a chunked draw store. */
public final class ColumnarDraws {
	private final int[] coordinates, retainedIndices; private final double[][] values; private final double[] logDensities;
	private final boolean[] accepted, divergent;
	ColumnarDraws(int[] coordinates, int[] retainedIndices, double[][] values, double[] logDensities, boolean[] accepted, boolean[] divergent) {
		this.coordinates = coordinates.clone(); this.retainedIndices = retainedIndices.clone(); this.values = copy(values);
		this.logDensities = logDensities.clone(); this.accepted = accepted.clone(); this.divergent = divergent.clone(); }
	public int size() { return retainedIndices.length; } public int[] coordinates() { return coordinates.clone(); }
	public int[] retainedIndices() { return retainedIndices.clone(); } public double[][] values() { return copy(values); }
	public double[] column(int selectedColumn) { return values[selectedColumn].clone(); } public double[] logDensities() { return logDensities.clone(); }
	public boolean[] accepted() { return accepted.clone(); } public boolean[] divergent() { return divergent.clone(); }
	private static double[][] copy(double[][] values) { double[][] result = new double[values.length][]; for (int i = 0; i < values.length; i++) result[i] = values[i].clone(); return result; }
}
