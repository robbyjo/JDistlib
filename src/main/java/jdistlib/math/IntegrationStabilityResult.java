/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

/** Results from repeating an integral under stricter and differently split settings. */
public final class IntegrationStabilityResult {
	private final IntegrationResult baseline;
	private final IntegrationResult tightened;
	private final IntegrationResult split;
	private final double maximumDiscrepancy;
	private final double allowedDiscrepancy;
	private final boolean stable;

	IntegrationStabilityResult(IntegrationResult baseline,
			IntegrationResult tightened, IntegrationResult split,
			double maximumDiscrepancy, double allowedDiscrepancy,
			boolean stable) {
		this.baseline = baseline;
		this.tightened = tightened;
		this.split = split;
		this.maximumDiscrepancy = maximumDiscrepancy;
		this.allowedDiscrepancy = allowedDiscrepancy;
		this.stable = stable;
	}

	public IntegrationResult getBaseline() { return baseline; }
	public IntegrationResult getTightened() { return tightened; }
	public IntegrationResult getSplit() { return split; }
	public double getMaximumDiscrepancy() { return maximumDiscrepancy; }
	public double getAllowedDiscrepancy() { return allowedDiscrepancy; }
	public boolean isStable() { return stable; }

	/** Returns a concise human-readable assessment. */
	public String message() {
		if (!baseline.isSuccess()) {
			return "baseline integration failed: " + baseline.detailedMessage();
		}
		if (!tightened.isSuccess()) {
			return "tightened integration failed: " + tightened.detailedMessage();
		}
		if (!split.isSuccess()) {
			return "split integration failed: " + split.detailedMessage();
		}
		return stable ? "stable across repeated integrations"
				: "integration changed more than its error/tolerance allowance";
	}

	/** Returns an RFC 8259 JSON diagnostic record. */
	public String toJson() { return IntegrationJson.toJson(this); }
}
