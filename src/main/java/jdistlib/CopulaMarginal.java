/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.function.DoubleUnaryOperator;

import jdistlib.generic.GenericDistribution;

/** A scalar marginal together with its continuity/atom contract. */
public final class CopulaMarginal {
	/** Measure used by a joint likelihood contribution. */
	public enum Kind { CONTINUOUS, DISCRETE }

	private final GenericDistribution distribution;
	private final Kind kind;
	private final DoubleUnaryOperator leftCumulative;

	private CopulaMarginal(GenericDistribution distribution, Kind kind,
			DoubleUnaryOperator leftCumulative) {
		if (distribution == null) throw new IllegalArgumentException("distribution must not be null");
		this.distribution = distribution;
		this.kind = kind;
		this.leftCumulative = leftCumulative;
	}

	/** Declares an atom-free marginal. */
	public static CopulaMarginal continuous(GenericDistribution distribution) {
		return new CopulaMarginal(distribution, Kind.CONTINUOUS, null);
	}

	/**
	 * Declares a discrete marginal, deriving {@code F(x-)} as {@code F(x)-p(x)}.
	 * Use the overload with an explicit left-limit function when a legacy
	 * distribution's mass and CDF conventions do not satisfy that identity.
	 */
	public static CopulaMarginal discrete(GenericDistribution distribution) {
		return discrete(distribution, x -> Math.max(0.0,
				distribution.cumulative(x, true, false)
				- distribution.density(x, false)));
	}

	/** Declares a discrete marginal with an explicit left-limit CDF. */
	public static CopulaMarginal discrete(GenericDistribution distribution,
			DoubleUnaryOperator leftCumulative) {
		if (leftCumulative == null)
			throw new IllegalArgumentException("left-limit CDF must not be null");
		return new CopulaMarginal(distribution, Kind.DISCRETE, leftCumulative);
	}

	public GenericDistribution getDistribution() { return distribution; }
	public Kind getKind() { return kind; }
	public boolean isContinuous() { return kind == Kind.CONTINUOUS; }
	public boolean isDiscrete() { return kind == Kind.DISCRETE; }

	double cumulative(double x) {
		return distribution.cumulative(x, true, false);
	}

	double leftCumulative(double x) {
		return isDiscrete() ? leftCumulative.applyAsDouble(x) : cumulative(x);
	}

	double logDensityOrMass(double x) { return distribution.density(x, true); }
	double quantile(double probability) {
		return distribution.quantile(probability, true, false);
	}
}
