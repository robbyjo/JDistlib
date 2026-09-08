/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.AtomAwareDistribution;
import jdistlib.NegBinomial;
import jdistlib.Poisson;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Polya-Aeppli count: Poisson clusters with shifted-geometric cluster sizes. */
public final class PolyaAeppliDistribution extends GenericDistribution
		implements AtomAwareDistribution, SupportedDistribution {
	private final double lambda, probability;
	public PolyaAeppliDistribution(double lambda, double probability) {
		if (!(lambda >= 0.0) || !(probability > 0.0 && probability <= 1.0) || !Double.isFinite(lambda))
			throw new IllegalArgumentException("invalid Polya-Aeppli parameters");
		this.lambda = lambda; this.probability = probability;
	}
	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x < 0 || !Double.isFinite(x) || x != Math.rint(x)) return log ? Double.NEGATIVE_INFINITY : 0.0;
		if (probability == 1.0 || lambda == 0.0) return Poisson.density(x, lambda, log);
		int n = CountMixtureUtil.index(x);
		if (n == 0) return log ? -lambda : Math.exp(-lambda);
		double term = -lambda + Math.log(lambda) + Math.log(probability) + (n-1.0)*Math.log1p(-probability);
		double result = term;
		double logRatio = Math.log(lambda) + Math.log(probability) - Math.log1p(-probability);
		for (int k = 1; k < n; k++) {
			term += logRatio + Math.log(n-k) - Math.log(k) - Math.log(k+1.0);
			result = ProbabilityMath.add(result, term);
		}
		return log ? result : Math.exp(result);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x < 0 || x == Double.POSITIVE_INFINITY)
			return CountMixtureUtil.endpoint(x >= 0, lowerTail, logP);
		if (probability == 1.0 || lambda == 0.0) return Poisson.cumulative(x, lambda, lowerTail, logP);
		int n = CountMixtureUtil.index(Math.floor(x));
		double result = lowerTail ? -lambda : Poisson.cumulative(n, lambda, false, true);
		for (int k = 1; k <= n; k++) result = ProbabilityMath.add(result,
				Poisson.density(k, lambda, true) + NegBinomial.cumulative(n-k, k, probability, lowerTail, true));
		result = Math.min(0.0, result);
		return logP ? result : Math.exp(result);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (probability == 1.0 || lambda == 0.0) return Poisson.quantile(p, lambda, lowerTail, logP);
		return CountMixtureUtil.quantile(this, p, lowerTail, logP);
	}
	@Override public double random() {
		double clusters = Poisson.random(lambda, random);
		if (probability == 1.0 || clusters == 0.0) return clusters;
		// A sum of shifted geometric counts is k + NB(k,p), avoiding k draws.
		return clusters + NegBinomial.random(clusters, probability, random);
	}
	@Override public double atomProbability(double x) { return density(x, false); }
	@Override public double getLowerBound() { return 0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
