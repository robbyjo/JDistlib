/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

/** Immutable wall-clock cost profile for integrand callback evaluations. */
public final class CallbackProfile {
	private static final CallbackProfile EMPTY = new CallbackProfile(0, 0, 0L, 0L, 0L);

	private final int attemptedEvaluations;
	private final int completedEvaluations;
	private final long totalCallbackNanos;
	private final long maximumCallbackNanos;
	private final long integrationWallNanos;

	CallbackProfile(int attemptedEvaluations, int completedEvaluations,
			long totalCallbackNanos, long maximumCallbackNanos,
			long integrationWallNanos) {
		this.attemptedEvaluations = attemptedEvaluations;
		this.completedEvaluations = completedEvaluations;
		this.totalCallbackNanos = totalCallbackNanos;
		this.maximumCallbackNanos = maximumCallbackNanos;
		this.integrationWallNanos = integrationWallNanos;
	}

	/** Empty profile used by legacy integration overloads. */
	public static CallbackProfile empty() { return EMPTY; }

	/** Combines sequential profiles, saturating nanosecond totals on overflow. */
	public static CallbackProfile combine(CallbackProfile... profiles) {
		if (profiles == null || profiles.length == 0) return EMPTY;
		int attempted = 0;
		int completed = 0;
		long total = 0L;
		long maximum = 0L;
		long wall = 0L;
		for (CallbackProfile profile : profiles) {
			if (profile == null) continue;
			attempted = saturatingAdd(attempted, profile.attemptedEvaluations);
			completed = saturatingAdd(completed, profile.completedEvaluations);
			total = saturatingAdd(total, profile.totalCallbackNanos);
			maximum = Math.max(maximum, profile.maximumCallbackNanos);
			wall = saturatingAdd(wall, profile.integrationWallNanos);
		}
		return new CallbackProfile(attempted, completed, total, maximum, wall);
	}

	public int getAttemptedEvaluations() { return attemptedEvaluations; }
	public int getCompletedEvaluations() { return completedEvaluations; }
	public long getTotalCallbackNanos() { return totalCallbackNanos; }
	public long getMaximumCallbackNanos() { return maximumCallbackNanos; }
	public long getIntegrationWallNanos() { return integrationWallNanos; }
	public double getAverageCallbackNanos() {
		return completedEvaluations == 0 ? 0.0
				: (double) totalCallbackNanos / completedEvaluations;
	}

	private static long saturatingAdd(long left, long right) {
		return Long.MAX_VALUE - left < right ? Long.MAX_VALUE : left + right;
	}

	private static int saturatingAdd(int left, int right) {
		return Integer.MAX_VALUE - left < right ? Integer.MAX_VALUE : left + right;
	}
}
