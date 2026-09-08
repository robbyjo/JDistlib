/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Tukey lambda distribution defined by its symmetric quantile function. */
public final class TukeyLambda extends GenericDistribution
		implements SupportedDistribution {
	private final double lambda;

	public TukeyLambda(double lambda) { this.lambda = lambda; }

	public static double quantile(double p, double lambda, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(p) || Double.isNaN(lambda)) return p + lambda;
		if (!Double.isFinite(lambda) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
        double lp=logP?p:Math.log(p),lq=logP?DistributionUtil.logOneMinusExp(p):Math.log1p(-p);
        if(!lowerTail){double t=lp;lp=lq;lq=t;}
        if(lambda==0.0)return lp-lq;
        if(lp==lq)return 0.0;
        // Factor the larger exponential; expm1 retains the lambda -> 0 limit.
        if(lambda*(lp-lq)>0) return -Math.exp(lambda*lp)*Math.expm1(lambda*(lq-lp))/lambda;
        return Math.exp(lambda*lq)*Math.expm1(lambda*(lp-lq))/lambda;
	}

    private static double logSmallProbability(double x,double lambda) {
        double target=-Math.abs(x),lo=-Double.MAX_VALUE,hi=-Math.log(2.0);
        // Work in log probability, so finite extreme quantiles remain reachable.
        lo=-1.0;
        while(quantile(lo,lambda,true,true)>target && lo>-1e307)lo*=2;
        for(int i=0;i<100;i++){double mid=lo+(hi-lo)/2;if(mid==lo||mid==hi)break;if(quantile(mid,lambda,true,true)>=target)hi=mid;else lo=mid;}
        return lo+(hi-lo)/2;
    }

	public static double cumulative(double x, double lambda, boolean lowerTail,
			boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(lambda)) return x + lambda;
		if (!Double.isFinite(lambda)) return Double.NaN;
		double bound = lambda > 0.0 ? 1.0 / lambda : Double.POSITIVE_INFINITY;
		if (x <= -bound) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x >= bound) return DistributionUtil.boundary(true, lowerTail, logP);
		double small = logSmallProbability(x, lambda);
		double value = lowerTail == (x < 0) ? small : DistributionUtil.logOneMinusExp(small);
		return logP ? value : Math.exp(value);
	}

	public static double density(double x, double lambda, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(lambda)) return x + lambda;
		if (!Double.isFinite(lambda)) return Double.NaN;
		double bound = lambda > 0.0 ? 1.0 / lambda : Double.POSITIVE_INFINITY;
		if (Math.abs(x) > bound || Double.isInfinite(x)) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (lambda == 1.0) return log ? -Math.log(2.0) : 0.5;
		if (lambda > 0.0 && Math.abs(x) == bound) {
			return lambda < 1.0 ? (log ? Double.NEGATIVE_INFINITY : 0.0) : (log ? 0.0 : 1.0);
		}
		double lp = logSmallProbability(x, lambda);
		double logDerivative = DistributionUtil.logAdd((lambda - 1.0) * lp,
				(lambda - 1.0) * DistributionUtil.logOneMinusExp(lp));
		return log ? -logDerivative : Math.exp(-logDerivative);
	}

	private static double logSum(double a, double b) {
		double high = Math.max(a, b);
		return high + Math.log(Math.exp(a - high) + Math.exp(b - high));
	}

	public static double random(double lambda, RandomEngine random) {
		if (!Double.isFinite(lambda)) return Double.NaN;
		return quantile(random.nextDouble(), lambda, true, false);
	}

	@Override public double density(double x, boolean log) { return density(x, lambda, log); }
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, lambda, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, lambda, lowerTail, logP);
	}
	@Override public double random() { return random(lambda, random); }
	@Override public double getLowerBound() {
		return lambda > 0.0 ? -1.0 / lambda : Double.NEGATIVE_INFINITY;
	}
	@Override public double getUpperBound() {
		return lambda > 0.0 ? 1.0 / lambda : Double.POSITIVE_INFINITY;
	}
}
