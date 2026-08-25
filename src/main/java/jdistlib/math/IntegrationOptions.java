/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

/** Immutable options for hardened numerical integration. */
public final class IntegrationOptions {
	/** Where opt-in hardened callback evaluations execute. */
	public enum CallbackExecution {
		/** Evaluate on the integrating thread with no worker overhead. */
		CALLER_THREAD,
		/** Evaluate on a private daemon worker so a time limit can release the caller. */
		ISOLATED_DAEMON
	}

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
	private final long maxTotalNanos;
	private final long maxCallbackNanos;
	private final CallbackExecution callbackExecution;

	private IntegrationOptions(Builder builder) {
		absoluteTolerance = builder.absoluteTolerance;
		relativeTolerance = builder.relativeTolerance;
		subdivisions = builder.subdivisions;
		maxEvaluations = builder.maxEvaluations;
		breakpoints = builder.breakpoints.clone();
		cancellation = builder.cancellation;
		method = builder.method;
		tanhSinhMaxLevels = builder.tanhSinhMaxLevels;
		maxTotalNanos = builder.maxTotalNanos;
		maxCallbackNanos = builder.maxCallbackNanos;
		callbackExecution = builder.callbackExecution;
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
				.tanhSinhMaxLevels(tanhSinhMaxLevels)
				.maxTotalTime(maxTotalNanos, TimeUnit.NANOSECONDS)
				.maxCallbackTime(maxCallbackNanos, TimeUnit.NANOSECONDS)
				.callbackExecution(callbackExecution);
	}

	public double getAbsoluteTolerance() { return absoluteTolerance; }
	public double getRelativeTolerance() { return relativeTolerance; }
	public int getSubdivisions() { return subdivisions; }
	public int getMaxEvaluations() { return maxEvaluations; }
	public double[] getBreakpoints() { return breakpoints.clone(); }
	public BooleanSupplier getCancellation() { return cancellation; }
	public Method getMethod() { return method; }
	public int getTanhSinhMaxLevels() { return tanhSinhMaxLevels; }
	/** Total wall-clock limit, or {@link Long#MAX_VALUE} when disabled. */
	public long getMaxTotalNanos() { return maxTotalNanos; }
	/** Per-evaluation wall-clock limit, or {@link Long#MAX_VALUE} when disabled. */
	public long getMaxCallbackNanos() { return maxCallbackNanos; }
	public CallbackExecution getCallbackExecution() { return callbackExecution; }

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
		private long maxTotalNanos = Long.MAX_VALUE;
		private long maxCallbackNanos = Long.MAX_VALUE;
		private CallbackExecution callbackExecution = CallbackExecution.CALLER_THREAD;

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

		/**
		 * Sets a benchmark-oriented total wall-clock budget. A value of
		 * {@link Long#MAX_VALUE} nanoseconds disables it.
		 */
		public Builder maxTotalTime(long value, TimeUnit unit) {
			maxTotalNanos = toNanos(value, unit, "maxTotalTime");
			return this;
		}

		/**
		 * Sets a benchmark-oriented wall-clock limit for one callback evaluation.
		 * Caller-thread execution observes this limit after a callback returns;
		 * isolated execution can return when the limit expires.
		 */
		public Builder maxCallbackTime(long value, TimeUnit unit) {
			maxCallbackNanos = toNanos(value, unit, "maxCallbackTime");
			return this;
		}

		/** Selects direct or opt-in private-daemon callback execution. */
		public Builder callbackExecution(CallbackExecution value) {
			callbackExecution = value;
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
			if (callbackExecution == null) {
				throw new IllegalArgumentException("callbackExecution must not be null");
			}
			if (callbackExecution == CallbackExecution.ISOLATED_DAEMON
					&& maxCallbackNanos == Long.MAX_VALUE
					&& maxTotalNanos == Long.MAX_VALUE) {
				throw new IllegalArgumentException(
						"isolated callback execution requires a callback or total time limit");
			}
			for (double point : breakpoints) {
				if (!Double.isFinite(point)) {
					throw new IllegalArgumentException("breakpoints must be finite");
				}
			}
			Arrays.sort(breakpoints);
			return new IntegrationOptions(this);
		}

		private static long toNanos(long value, TimeUnit unit, String name) {
			if (unit == null) throw new IllegalArgumentException(name + " unit must not be null");
			if (value < 1) throw new IllegalArgumentException(name + " must be positive");
			if (value == Long.MAX_VALUE && unit == TimeUnit.NANOSECONDS) {
				return Long.MAX_VALUE;
			}
			long nanos = unit.toNanos(value);
			return nanos <= 0L ? Long.MAX_VALUE : nanos;
		}
	}
}
