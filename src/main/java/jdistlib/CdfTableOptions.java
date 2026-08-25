/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Immutable settings for an adaptive monotone numerical CDF table. */
public final class CdfTableOptions {
	private final double tolerance;
	private final int initialIntervals;
	private final int maximumNodes;
	private final int refinementPasses;

	private CdfTableOptions(Builder builder) {
		tolerance = builder.tolerance;
		initialIntervals = builder.initialIntervals;
		maximumNodes = builder.maximumNodes;
		refinementPasses = builder.refinementPasses;
	}

	public static Builder builder() { return new Builder(); }
	public static CdfTableOptions defaults() { return builder().build(); }
	public double getTolerance() { return tolerance; }
	public int getInitialIntervals() { return initialIntervals; }
	public int getMaximumNodes() { return maximumNodes; }
	public int getRefinementPasses() { return refinementPasses; }

	public static final class Builder {
		private double tolerance = 1e-10;
		private int initialIntervals = 32;
		private int maximumNodes = 4097;
		private int refinementPasses = 12;

		private Builder() {}
		public Builder tolerance(double value) { tolerance = value; return this; }
		public Builder initialIntervals(int value) {
			initialIntervals = value;
			return this;
		}
		public Builder maximumNodes(int value) { maximumNodes = value; return this; }
		public Builder refinementPasses(int value) {
			refinementPasses = value;
			return this;
		}
		public CdfTableOptions build() {
			if (!(tolerance > 0.0) || !Double.isFinite(tolerance)) {
				throw new IllegalArgumentException("CDF-table tolerance must be finite and positive");
			}
			if (initialIntervals < 4 || initialIntervals > 4096) {
				throw new IllegalArgumentException("initialIntervals must be between 4 and 4096");
			}
			if (maximumNodes < initialIntervals + 1 || maximumNodes > 1000000) {
				throw new IllegalArgumentException("maximumNodes is inconsistent or excessive");
			}
			if (refinementPasses < 0 || refinementPasses > 30) {
				throw new IllegalArgumentException("refinementPasses must be between 0 and 30");
			}
			return new CdfTableOptions(this);
		}
	}
}
