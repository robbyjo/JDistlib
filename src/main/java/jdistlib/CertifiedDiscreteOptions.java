/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Immutable truncation settings for certified infinite discrete supports. */
public final class CertifiedDiscreteOptions {
	private final double omittedProbabilityTolerance;
	private final int minimumTerms;
	private final int maximumTerms;

	private CertifiedDiscreteOptions(Builder builder) {
		omittedProbabilityTolerance = builder.omittedProbabilityTolerance;
		minimumTerms = builder.minimumTerms;
		maximumTerms = builder.maximumTerms;
	}

	public static Builder builder() { return new Builder(); }
	public static CertifiedDiscreteOptions defaults() { return builder().build(); }
	public double getOmittedProbabilityTolerance() {
		return omittedProbabilityTolerance;
	}
	public int getMinimumTerms() { return minimumTerms; }
	public int getMaximumTerms() { return maximumTerms; }

	public static final class Builder {
		private double omittedProbabilityTolerance = 1e-12;
		private int minimumTerms = 16;
		private int maximumTerms = 1000000;
		private Builder() {}
		public Builder omittedProbabilityTolerance(double value) {
			omittedProbabilityTolerance = value;
			return this;
		}
		public Builder minimumTerms(int value) { minimumTerms = value; return this; }
		public Builder maximumTerms(int value) { maximumTerms = value; return this; }
		public CertifiedDiscreteOptions build() {
			if (!(omittedProbabilityTolerance > 0.0)
					|| omittedProbabilityTolerance >= 1.0
					|| !Double.isFinite(omittedProbabilityTolerance)) {
				throw new IllegalArgumentException(
						"omitted-probability tolerance must lie strictly between zero and one");
			}
			if (minimumTerms < 1 || maximumTerms < minimumTerms
					|| maximumTerms > 1000000) {
				throw new IllegalArgumentException("invalid certified-tail term limits");
			}
			return new CertifiedDiscreteOptions(this);
		}
	}
}
