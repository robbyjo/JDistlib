/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

import java.util.Arrays;
import java.util.function.BooleanSupplier;

/** Immutable options for hardened numerical integration. */
public final class IntegrationOptions {
	/** Available integration strategies. */
	public enum Method {
		/** QUADPACK DQAGS/DQAGI, matching the historical implementation. */
		QUADPACK,
		/** Double-exponential tanh-sinh quadrature for finite intervals. */
		TANH_SINH,
		/**
		 * Double-exponential tanh-sinh, exp-sinh, or sinh-sinh quadrature,
		 * selected according to the interval bounds.
		 */
		DOUBLE_EXPONENTIAL,
		/** QUADPACK with a matching double-exponential fallback. */
		AUTO
	}

	private static final double R_TOLERANCE = Math.pow(Math.ulp(1.0), 0.25);

	private final double absoluteTolerance;
	private final double relativeTolerance;
	private final int subdivisions;
	private final int maxEvaluations;
	private final double[] breakpoints;
	private final BooleanSupplier cancellation;
	private final Method method;
	private final int tanhSinhMaxLevels;

	private IntegrationOptions(Builder builder) {
		absoluteTolerance = builder.absoluteTolerance;
		relativeTolerance = builder.relativeTolerance;
		subdivisions = builder.subdivisions;
		maxEvaluations = builder.maxEvaluations;
		breakpoints = builder.breakpoints.clone();
		cancellation = builder.cancellation;
		method = builder.method;
		tanhSinhMaxLevels = builder.tanhSinhMaxLevels;
	}

	/** Returns a builder initialized to R-compatible integration defaults. */
	public static Builder builder() { return new Builder(); }

	/** Returns R-compatible QUADPACK defaults. */
	public static IntegrationOptions defaults() { return builder().build(); }

	/** Returns a builder initialized from this object. */
	public Builder toBuilder() {
		return new Builder()
				.tolerances(absoluteTolerance, relativeTolerance)
				.subdivisions(subdivisions)
				.maxEvaluations(maxEvaluations)
				.breakpoints(breakpoints)
				.cancellation(cancellation)
				.method(method)
				.tanhSinhMaxLevels(tanhSinhMaxLevels);
	}

	public double getAbsoluteTolerance() { return absoluteTolerance; }
	public double getRelativeTolerance() { return relativeTolerance; }
	public int getSubdivisions() { return subdivisions; }
	public int getMaxEvaluations() { return maxEvaluations; }
	public double[] getBreakpoints() { return breakpoints.clone(); }
	public BooleanSupplier getCancellation() { return cancellation; }
	public Method getMethod() { return method; }
	public int getTanhSinhMaxLevels() { return tanhSinhMaxLevels; }

	/** Builder for {@link IntegrationOptions}. */
	public static final class Builder {
		private double absoluteTolerance = R_TOLERANCE;
		private double relativeTolerance = R_TOLERANCE;
		private int subdivisions = 100;
		private int maxEvaluations = Integer.MAX_VALUE;
		private double[] breakpoints = new double[0];
		private BooleanSupplier cancellation;
		private Method method = Method.QUADPACK;
		private int tanhSinhMaxLevels = 12;

		private Builder() {}

		public Builder tolerances(double absolute, double relative) {
			absoluteTolerance = absolute;
			relativeTolerance = relative;
			return this;
		}

		public Builder subdivisions(int value) {
			subdivisions = value;
			return this;
		}

		public Builder maxEvaluations(int value) {
			maxEvaluations = value;
			return this;
		}

		/**
		 * Declares finite locations at which the interval should be split. Values
		 * outside a particular integration interval are ignored.
		 */
		public Builder breakpoints(double... values) {
			breakpoints = values == null ? new double[0] : values.clone();
			return this;
		}

		public Builder cancellation(BooleanSupplier value) {
			cancellation = value;
			return this;
		}

		public Builder method(Method value) {
			method = value;
			return this;
		}

		public Builder tanhSinhMaxLevels(int value) {
			tanhSinhMaxLevels = value;
			return this;
		}

		public IntegrationOptions build() {
			if (!Double.isFinite(absoluteTolerance)
					|| !Double.isFinite(relativeTolerance)
					|| absoluteTolerance < 0.0 || relativeTolerance < 0.0) {
				throw new IllegalArgumentException("integration tolerances must be nonnegative");
			}
			if (subdivisions < 1) {
				throw new IllegalArgumentException("subdivisions must be positive");
			}
			if (subdivisions > 1000000) {
				throw new IllegalArgumentException(
						"subdivisions above 1000000 risk excessive workspace allocation");
			}
			if (maxEvaluations < 1) {
				throw new IllegalArgumentException("maxEvaluations must be positive");
			}
			if (method == null) throw new IllegalArgumentException("method must not be null");
			if (tanhSinhMaxLevels < 1 || tanhSinhMaxLevels > 20) {
				throw new IllegalArgumentException("tanh-sinh levels must be between 1 and 20");
			}
			for (double point : breakpoints) {
				if (!Double.isFinite(point)) {
					throw new IllegalArgumentException("breakpoints must be finite");
				}
			}
			Arrays.sort(breakpoints);
			return new IntegrationOptions(this);
		}
	}
}
