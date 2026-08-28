/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

/** Open/closed interval on which a cumulant-generating function exists. */
public final class TransformDomain {
	private final double lower;
	private final double upper;
	private final boolean lowerIncluded;
	private final boolean upperIncluded;

	public TransformDomain(double lower, boolean lowerIncluded, double upper, boolean upperIncluded) {
		if (Double.isNaN(lower) || Double.isNaN(upper) || lower > upper)
			throw new IllegalArgumentException("invalid transform domain");
		this.lower = lower;
		this.upper = upper;
		this.lowerIncluded = lowerIncluded;
		this.upperIncluded = upperIncluded;
	}

	public static TransformDomain allReal() {
		return new TransformDomain(Double.NEGATIVE_INFINITY, false, Double.POSITIVE_INFINITY, false);
	}

	public boolean contains(double value) {
		return (value > lower || lowerIncluded && value == lower)
				&& (value < upper || upperIncluded && value == upper);
	}
	public double getLower() { return lower; }
	public double getUpper() { return upper; }
	public boolean isLowerIncluded() { return lowerIncluded; }
	public boolean isUpperIncluded() { return upperIncluded; }
}
