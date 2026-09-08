/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.AtomAwareDistribution;
import jdistlib.Gamma;
import jdistlib.NegBinomial;
import jdistlib.Poisson;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Delaporte count: Poisson(lambda) plus NB(shape, successProbability). */
public final class DelaporteDistribution extends GenericDistribution
		implements AtomAwareDistribution, SupportedDistribution {
	private final double lambda, shape, probability;
	public DelaporteDistribution(double lambda, double shape, double probability) {
		if (!(lambda >= 0.0) || !(shape > 0.0) || !(probability > 0.0 && probability <= 1.0)
				|| !Double.isFinite(lambda) || !Double.isFinite(shape))
			throw new IllegalArgumentException("invalid Delaporte parameters");
		this.lambda = lambda; this.shape = shape; this.probability = probability;
	}
	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x < 0 || !Double.isFinite(x) || x != Math.rint(x)) return log ? Double.NEGATIVE_INFINITY : 0.0;
		if (lambda == 0.0) return NegBinomial.density(x, shape, probability, log);
		if (probability == 1.0) return Poisson.density(x, lambda, log);
		int n = CountMixtureUtil.index(x);
		double answer = Double.NEGATIVE_INFINITY;
		for (int k = 0; k <= n; k++) answer = ProbabilityMath.add(answer,
				Poisson.density(k, lambda, true) + NegBinomial.density(n-k, shape, probability, true));
		return log ? answer : Math.exp(answer);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x < 0 || x == Double.POSITIVE_INFINITY)
			return CountMixtureUtil.endpoint(x >= 0, lowerTail, logP);
		if (lambda == 0.0) return NegBinomial.cumulative(x, shape, probability, lowerTail, logP);
		if (probability == 1.0) return Poisson.cumulative(x, lambda, lowerTail, logP);
		int n = CountMixtureUtil.index(Math.floor(x));
		// Condition on the Poisson count, evaluating the requested NB tail directly.
		double result = lowerTail ? Double.NEGATIVE_INFINITY : Poisson.cumulative(n, lambda, false, true);
		for (int k = 0; k <= n; k++) result = ProbabilityMath.add(result,
				Poisson.density(k, lambda, true) + NegBinomial.cumulative(n-k, shape, probability, lowerTail, true));
		result = Math.min(0.0, result);
		return logP ? result : Math.exp(result);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (lambda == 0.0) return NegBinomial.quantile(p, shape, probability, lowerTail, logP);
		if (probability == 1.0) return Poisson.quantile(p, lambda, lowerTail, logP);
		return CountMixtureUtil.quantile(this, p, lowerTail, logP);
	}
	@Override public double random() {
		double mixing = probability == 1.0 ? 0.0 : Gamma.random(shape, (1.0-probability)/probability, random);
		return Poisson.random(lambda + mixing, random);
	}
	@Override public double atomProbability(double x) { return density(x, false); }
	@Override public double getLowerBound() { return 0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
