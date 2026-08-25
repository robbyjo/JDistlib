/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Factory methods for common caller-certified infinite-series tail bounds. */
public final class DiscreteTailBounds {
	private DiscreteTailBounds() {}

	/**
	 * Geometric/ratio-test bound. The caller promises every successive omitted
	 * weight is at most {@code maximumRatio} times its predecessor.
	 */
	public static DiscreteTailBound geometricRatio(double maximumRatio) {
		if (!(maximumRatio >= 0.0 && maximumRatio < 1.0)) {
			throw new IllegalArgumentException("maximumRatio must lie in [0, 1)");
		}
		return (firstOmitted, firstWeight) -> firstWeight / (1.0 - maximumRatio);
	}

	/**
	 * Right-tail integral-test bound for weights proportional to
	 * {@code (k + offset)^(-exponent)}.
	 */
	public static DiscreteTailBound rightPowerLaw(double exponent, double offset) {
		validatePowerLaw(exponent, offset);
		return (firstOmitted, firstWeight) -> powerBound(firstOmitted + offset,
				firstWeight, exponent);
	}

	/**
	 * Left-tail integral-test bound for weights proportional to
	 * {@code (-k + offset)^(-exponent)}.
	 */
	public static DiscreteTailBound leftPowerLaw(double exponent, double offset) {
		validatePowerLaw(exponent, offset);
		return (firstOmitted, firstWeight) -> powerBound(-firstOmitted + offset,
				firstWeight, exponent);
	}

	/** Power-law bound suitable for either outward tail around zero. */
	public static DiscreteTailBound symmetricPowerLaw(double exponent,
			double offset) {
		validatePowerLaw(exponent, offset);
		return (firstOmitted, firstWeight) -> powerBound(
				Math.abs((double) firstOmitted) + offset, firstWeight, exponent);
	}

	/** Uses a fixed finite upper bound, which must include the first term. */
	public static DiscreteTailBound constant(double upperBound) {
		if (!(upperBound >= 0.0) || !Double.isFinite(upperBound)) {
			throw new IllegalArgumentException("upperBound must be finite and nonnegative");
		}
		return (firstOmitted, firstWeight) -> upperBound;
	}

	/**
	 * Activates a remainder certificate only after a caller-verified finite
	 * prefix has been included.
	 */
	public static DiscreteTailBound afterFinitePrefix(long firstRemainder,
			DiscreteTailBound remainder) {
		if (remainder == null) {
			throw new IllegalArgumentException("remainder certificate must not be null");
		}
		return (firstOmitted, firstWeight) -> {
			if (firstOmitted < firstRemainder) return Double.MAX_VALUE;
			return remainder.upperBound(firstOmitted, firstWeight);
		};
	}

	/** Left-moving counterpart of {@link #afterFinitePrefix(long, DiscreteTailBound)}. */
	public static DiscreteTailBound beforeFinitePrefix(long firstRemainder,
			DiscreteTailBound remainder) {
		if (remainder == null) {
			throw new IllegalArgumentException("remainder certificate must not be null");
		}
		return (firstOmitted, firstWeight) -> {
			if (firstOmitted > firstRemainder) return Double.MAX_VALUE;
			return remainder.upperBound(firstOmitted, firstWeight);
		};
	}

	private static void validatePowerLaw(double exponent, double offset) {
		if (!(exponent > 1.0) || !Double.isFinite(exponent)
				|| !(offset >= 0.0) || !Double.isFinite(offset)) {
			throw new IllegalArgumentException(
					"power-law exponent must exceed one and offset must be nonnegative");
		}
	}

	private static double powerBound(double coordinate, double firstWeight,
			double exponent) {
		if (!(coordinate > 0.0)) return Double.POSITIVE_INFINITY;
		return firstWeight * (1.0 + coordinate / (exponent - 1.0));
	}
}
