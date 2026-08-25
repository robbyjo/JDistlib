/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

/** Immutable settings for absolute-moment diagnostics. */
public final class MomentAnalysisOptions {
	private final double[] orders;
	private final double splitPoint;

	private MomentAnalysisOptions(Builder builder) {
		orders = builder.orders.clone();
		splitPoint = builder.splitPoint;
	}

	public static Builder builder() { return new Builder(); }
	public static MomentAnalysisOptions defaults() { return builder().build(); }
	public double[] getOrders() { return orders.clone(); }
	public double getSplitPoint() { return splitPoint; }

	/** Builder for moment orders and the left/right reporting boundary. */
	public static final class Builder {
		private double[] orders = {1.0, 2.0};
		private double splitPoint;

		private Builder() {}

		public Builder orders(double... values) {
			orders = values == null ? new double[0] : values.clone();
			return this;
		}

		public Builder splitPoint(double value) {
			splitPoint = value;
			return this;
		}

		public MomentAnalysisOptions build() {
			if (orders.length == 0 || orders.length > 64) {
				throw new IllegalArgumentException(
						"between 1 and 64 absolute-moment orders are required");
			}
			if (!Double.isFinite(splitPoint)) {
				throw new IllegalArgumentException("splitPoint must be finite");
			}
			Arrays.sort(orders);
			for (int i = 0; i < orders.length; i++) {
				if (!Double.isFinite(orders[i]) || orders[i] < 0.0) {
					throw new IllegalArgumentException(
							"absolute-moment orders must be finite and nonnegative");
				}
				if (i > 0 && orders[i] == orders[i - 1]) {
					throw new IllegalArgumentException("absolute-moment orders must be unique");
				}
			}
			return new MomentAnalysisOptions(this);
		}
	}
}
