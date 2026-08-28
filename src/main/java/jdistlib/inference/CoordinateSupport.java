/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Typed support for one coordinate in a mixed continuous/discrete state. */
public final class CoordinateSupport {
	public enum Kind { REAL, BOUNDED_REAL, INTEGER, CATEGORICAL }
	private final Kind kind; private final double lower, upper;
	private CoordinateSupport(Kind kind, double lower, double upper) { this.kind = kind; this.lower = lower; this.upper = upper; }
	public static CoordinateSupport real() { return new CoordinateSupport(Kind.REAL, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY); }
	public static CoordinateSupport boundedReal(double lower, double upper) {
		if (!Double.isFinite(lower) || !Double.isFinite(upper) || !(lower < upper)) throw new IllegalArgumentException("finite ordered bounds required");
		return new CoordinateSupport(Kind.BOUNDED_REAL, lower, upper);
	}
	public static CoordinateSupport integer() { return new CoordinateSupport(Kind.INTEGER, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY); }
	public static CoordinateSupport integer(int lower, int upper) {
		if (lower > upper) throw new IllegalArgumentException("ordered integer bounds required");
		return new CoordinateSupport(Kind.INTEGER, lower, upper);
	}
	public static CoordinateSupport categorical(int categories) {
		if (categories < 2) throw new IllegalArgumentException("at least two categories required");
		return new CoordinateSupport(Kind.CATEGORICAL, 0.0, categories - 1.0);
	}
	public static CoordinateSupport binary() { return categorical(2); }
	public Kind kind() { return kind; }
	public boolean discrete() { return kind == Kind.INTEGER || kind == Kind.CATEGORICAL; }
	public boolean finite() { return Double.isFinite(lower) && Double.isFinite(upper); }
	public double lower() { return lower; }
	public double upper() { return upper; }
	public int categoryCount() { return kind == Kind.CATEGORICAL ? (int) upper + 1 : 0; }
	public boolean contains(double value) {
		if (!Double.isFinite(value) || value < lower || value > upper) return false;
		return !discrete() || value == Math.rint(value);
	}
}
