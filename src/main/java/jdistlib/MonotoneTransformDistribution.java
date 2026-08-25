/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.UnivariateFunction;

/** Distribution induced by a differentiable, strictly monotone transformation. */
public final class MonotoneTransformDistribution extends GenericDistribution
		implements SupportedDistribution, AtomAwareDistribution {
	private final GenericDistribution base;
	private final UnivariateFunction forward;
	private final UnivariateFunction inverse;
	private final UnivariateFunction logAbsInverseDerivative;
	private final boolean increasing;
	private final double lower;
	private final double upper;

	public MonotoneTransformDistribution(GenericDistribution base,
			UnivariateFunction forward, UnivariateFunction inverse,
			UnivariateFunction logAbsInverseDerivative, boolean increasing,
			double lower, double upper) {
		if (base == null || forward == null || inverse == null
				|| logAbsInverseDerivative == null || Double.isNaN(lower)
				|| Double.isNaN(upper) || !(lower < upper)) {
			throw new IllegalArgumentException("transform functions and ordered output bounds are required");
		}
		this.base = base;
		this.forward = forward;
		this.inverse = inverse;
		this.logAbsInverseDerivative = logAbsInverseDerivative;
		this.increasing = increasing;
		this.lower = lower;
		this.upper = upper;
	}

	/** Creates the distribution of {@code shift + scale * X}. */
	public static MonotoneTransformDistribution affine(GenericDistribution base,
			double shift, double scale) {
		if (!Double.isFinite(shift) || !Double.isFinite(scale) || scale == 0.0) {
			throw new IllegalArgumentException("affine shift and nonzero scale must be finite");
		}
		double baseLower = base instanceof SupportedDistribution
				? ((SupportedDistribution) base).getLowerBound() : base.quantile(0.0);
		double baseUpper = base instanceof SupportedDistribution
				? ((SupportedDistribution) base).getUpperBound() : base.quantile(1.0);
		double first = shift + scale * baseLower;
		double second = shift + scale * baseUpper;
		return new MonotoneTransformDistribution(base,
				x -> shift + scale * x, y -> (y - shift) / scale,
				y -> -Math.log(Math.abs(scale)), scale > 0.0,
				Math.min(first, second), Math.max(first, second));
	}

	public GenericDistribution getBaseDistribution() { return base; }
	@Override public double getLowerBound() { return lower; }
	@Override public double getUpperBound() { return upper; }

	@Override public double density(double y, boolean log) {
		if (Double.isNaN(y)) return Double.NaN;
		if (y < lower || y > upper) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double atom = atomProbability(y);
		if (atom > 0.0) return log ? Math.log(atom) : atom;
		double x = inverse.eval(y);
		double logDensity = base.density(x, true) + logAbsInverseDerivative.eval(y);
		return log ? logDensity : Math.exp(logDensity);
	}

	@Override public double cumulative(double y, boolean lowerTail, boolean logP) {
		if (Double.isNaN(y)) return Double.NaN;
		if (y < lower) return DistributionUtil.boundary(false, lowerTail, logP);
		if (y > upper) return DistributionUtil.boundary(true, lowerTail, logP);
		double x = inverse.eval(y);
		return base.cumulative(increasing ? x : Math.nextDown(x),
				increasing ? lowerTail : !lowerTail, logP);
	}

	@Override public double atomProbability(double y) {
		if (Double.isNaN(y) || y < lower || y > upper
				|| !(base instanceof AtomAwareDistribution)) return 0.0;
		return ((AtomAwareDistribution) base).atomProbability(inverse.eval(y));
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return forward.eval(base.quantile(p,
				increasing ? lowerTail : !lowerTail, logP));
	}

	@Override public double random() {
		return forward.eval(base.quantile(random.nextDouble()));
	}
}
