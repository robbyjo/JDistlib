/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 */
package jdistlib;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.rint;
import static jdistlib.math.MathFunctions.isInfinite;
import static jdistlib.math.MathFunctions.isNonInt;
import static jdistlib.math.MathFunctions.lgammafn;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Beta binomial distribution, taken from gamlss.dist package, plus some underflow guard.
 * Parameterization: mu, sigma, bd.
 *
 */
public class BetaBinomial extends GenericDistribution {
    private static boolean invalid(double mu, double sigma, double n) {
        return !(mu >= 0.0 && mu <= 1.0) || !(sigma > 0.0) || !Double.isFinite(sigma)
                || !(n >= 0.0) || n > Integer.MAX_VALUE || n != Math.rint(n);
    }
    public static final double density(double x, double mu, double sigma, double n, boolean logP) {
        if (invalid(mu,sigma,n) || Double.isNaN(x)) return Double.NaN;
        if (x < 0.0 || x > n || x != Math.rint(x)) return logP ? Double.NEGATIVE_INFINITY : 0.0;
        if (mu == 0.0 || mu == 1.0 || n == 0.0) return Binomial.density(x,n,mu,logP);
        double a=mu/sigma,b=(1.0-mu)/sigma;
        double value=jdistlib.math.MathFunctions.lchoose(n,x)
                +jdistlib.math.MathFunctions.lbeta(x+a,n-x+b)-jdistlib.math.MathFunctions.lbeta(a,b);
        return logP ? value : exp(value);
    }
    public static final double cumulative(double x, double mu, double sigma, double n, boolean lower, boolean logP) {
        if (invalid(mu,sigma,n) || Double.isNaN(x)) return Double.NaN;
        if (x < 0) return DistributionUtil.boundary(false,lower,logP);
        if (x >= n) return DistributionUtil.boundary(true,lower,logP);
        if (mu == 0.0 || mu == 1.0 || n == 0.0) return Binomial.cumulative(x,n,mu,lower,logP);
        int k=(int)Math.floor(x);
        // Sum the requested tail directly. Each subsequent mass costs four logs,
        // rather than another set of gamma functions.
        int from=lower?0:k+1, to=lower?k:(int)n;
        double term=density(from,mu,sigma,n,true), total=term;
        double a=mu/sigma,b=(1-mu)/sigma;
        for(int i=from;i<to;i++) {
            term+=log(n-i)-log(i+1.0)+log(i+a)-log(n-i-1.0+b);
            total=DistributionUtil.logAdd(total,term);
        }
        total=Math.min(total,0.0);
        return logP ? total : exp(total);
    }
    public static final double quantile(double p, double mu, double sigma, double n, boolean lower, boolean logP) {
        if (invalid(mu,sigma,n)) return Double.NaN;
        return DistributionUtil.discreteQuantile(p,lower,logP,0,n,(x,lt,lp)->cumulative(x,mu,sigma,n,lt,lp));
    }
    public static final double random(double mu, double sigma, double n, RandomEngine random) {
        if (invalid(mu,sigma,n)) return Double.NaN;
        if (mu==0.0 || mu==1.0) return n*mu;
        return Binomial.random(n,Beta.random(mu/sigma,(1-mu)/sigma,random),random);
    }
	public static final double[] random(int n, double mu, double sigma, double bd, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(mu, sigma, bd, random);
		return rand;
	}

	protected double mu, sigma;
	protected int bd;

	public BetaBinomial(double mu, double sigma, int bd) {
		this.mu = mu; this.sigma = sigma; this.bd = bd;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, mu, sigma, bd, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, mu, sigma, bd, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, mu, sigma, bd, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(mu, sigma, bd, random);
	}
}
