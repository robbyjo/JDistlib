/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/** Prentice generalized-F survival distribution used by flexsurv. */
public final class GeneralizedF extends GenericDistribution
		implements SupportedDistribution {
	private final double mu;
	private final double sigma;
	private final double q;
	private final double p;

	public GeneralizedF(double mu, double sigma, double q, double p) {
		this.mu = mu;
		this.sigma = sigma;
		this.q = q;
		this.p = p;
	}

	private static boolean invalid(double mu, double sigma, double q, double p) {
		return !Double.isFinite(mu) || !(sigma > 0.0) || !Double.isFinite(sigma)
				|| !Double.isFinite(q) || !(p >= 0.0) || !Double.isFinite(p);
	}

	private static double generalizedGammaDensity(double x, double mu,
			double sigma, double q, boolean log) {
		if (q == 0.0) return LogNormal.density(x, mu, sigma, log);
		if (x <= 0.0) return log ? Double.NEGATIVE_INFINITY : 0.0;
		double w = (Math.log(x) - mu) / sigma;
		double inverseQSquared = 1.0 / (q * q);
		double value = -Math.log(sigma * x) + Math.log(Math.abs(q))
				* (1.0 - 2.0 * inverseQSquared)
				+ inverseQSquared * (q * w - Math.exp(q * w))
				- MathFunctions.lgammafn(inverseQSquared);
		return log ? value : Math.exp(value);
	}

	private static double generalizedGammaCumulative(double x, double mu,
			double sigma, double q, boolean lowerTail, boolean logP) {
		if (q == 0.0) return LogNormal.cumulative(x, mu, sigma, lowerTail, logP);
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		double shape = 1.0 / (q * q);
		double transformed = Math.exp(q * (Math.log(x) - mu) / sigma) * shape;
		return Gamma.cumulative(transformed, shape, 1.0,
				q > 0.0 ? lowerTail : !lowerTail, logP);
	}

	private static double generalizedGammaQuantile(double probability, double mu,
			double sigma, double q, boolean lowerTail, boolean logP) {
		if (q == 0.0) return LogNormal.quantile(probability, mu, sigma,
				lowerTail, logP);
		double shape = 1.0 / (q * q);
		double gamma = Gamma.quantile(probability, shape, 1.0,
				q > 0.0 ? lowerTail : !lowerTail, logP);
		return Math.exp(mu + sigma * Math.log(q * q * gamma) / q);
	}

	public static double density(double x, double mu, double sigma, double q,
			double p, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(mu) || Double.isNaN(sigma)
				|| Double.isNaN(q) || Double.isNaN(p)) return x + mu + sigma + q + p;
		if (invalid(mu, sigma, q, p)) return Double.NaN;
		if (p == 0.0) return generalizedGammaDensity(x, mu, sigma, q, log);
		if (x <= 0.0 || x == Double.POSITIVE_INFINITY) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double delta = Math.sqrt(q * q + 2.0 * p);
		double s1 = 2.0 / (delta * (delta + q));
		double s2 = 2.0 / (delta * (delta - q));
		double logX = Math.log(x);
		double logRatio = delta * (logX - mu) / sigma + Math.log(s1 / s2);
		double value = Math.log(delta) + s1 * delta * (logX - mu) / sigma
				+ s1 * Math.log(s1 / s2) - Math.log(sigma * x)
				- (s1 + s2) * logOnePlusExp(logRatio)
				- MathFunctions.lbeta(s1, s2);
		return log ? value : Math.exp(value);
	}

	private static double logOnePlusExp(double x) {
		return x > 0.0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
	}

	public static double cumulative(double x, double mu, double sigma, double q,
			double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(mu) || Double.isNaN(sigma)
				|| Double.isNaN(q) || Double.isNaN(p)) return x + mu + sigma + q + p;
		if (invalid(mu, sigma, q, p)) return Double.NaN;
		if (p == 0.0) {
			return generalizedGammaCumulative(x, mu, sigma, q, lowerTail, logP);
		}
		if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x == Double.POSITIVE_INFINITY) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double delta = Math.sqrt(q * q + 2.0 * p);
		double s1 = 2.0 / (delta * (delta + q));
		double s2 = 2.0 / (delta * (delta - q));
		double logOdds = Math.log(s1 / s2)
				+ delta * (Math.log(x) - mu) / sigma;
		if (logOdds > 0.0) {
			double complement = Math.exp(-logOnePlusExp(logOdds));
			return Beta.cumulative(complement, s2, s1, !lowerTail, logP);
		}
		double u = Math.exp(logOdds - logOnePlusExp(logOdds));
		return Beta.cumulative(u, s1, s2, lowerTail, logP);
	}

	public static double quantile(double probability, double mu, double sigma,
			double q, double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(probability) || invalid(mu, sigma, q, p)
				|| DistributionUtil.invalidProbability(probability, logP)) return Double.NaN;
		if (p == 0.0) {
			return generalizedGammaQuantile(probability, mu, sigma, q,
					lowerTail, logP);
		}
		double delta = Math.sqrt(q * q + 2.0 * p);
		double s1 = 2.0 / (delta * (delta + q));
		double s2 = 2.0 / (delta * (delta - q));
		double value = F.quantile(probability, 2.0 * s1, 2.0 * s2,
				lowerTail, logP);
		return Math.exp(mu + sigma * Math.log(value) / delta);
	}

	public static double random(double mu, double sigma, double q, double p,
			RandomEngine random) {
		if (invalid(mu, sigma, q, p)) return Double.NaN;
		if (p == 0.0) {
			if (q == 0.0) return LogNormal.random(mu, sigma, random);
			double shape = 1.0 / (q * q);
			double gamma = Gamma.random(shape, 1.0, random);
			return Math.exp(mu + sigma * Math.log(q * q * gamma) / q);
		}
		double delta = Math.sqrt(q * q + 2.0 * p);
		double s1 = 2.0 / (delta * (delta + q));
		double s2 = 2.0 / (delta * (delta - q));
		return Math.exp(mu + sigma * Math.log(F.random(2.0 * s1, 2.0 * s2,
				random)) / delta);
	}

	@Override public double density(double x, boolean log) {
		return density(x, mu, sigma, q, p, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mu, sigma, q, p, lowerTail, logP);
	}
	@Override public double quantile(double probability, boolean lowerTail, boolean logP) {
		return quantile(probability, mu, sigma, q, p, lowerTail, logP);
	}
	@Override public double random() { return random(mu, sigma, q, p, random); }
	@Override public double getLowerBound() { return 0.0; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
