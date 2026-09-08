/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/** Location-scale slash distribution, {@code mu + sigma * Z / U}. */
public final class Slash extends GenericDistribution implements SupportedDistribution {
	private static final double PHI_ZERO = 0.39894228040143267794;
	private final double mu;
	private final double sigma;

	public Slash(double mu, double sigma) { this.mu = mu; this.sigma = sigma; }

	private static boolean invalid(double mu, double sigma) {
		return !Double.isFinite(mu) || !(sigma > 0.0) || !Double.isFinite(sigma);
	}

    public static double density(double x,double mu,double sigma,boolean logP) {
        if(invalid(mu,sigma)||Double.isNaN(x)) return Double.NaN;
        if(Double.isInfinite(x)) return logP?Double.NEGATIVE_INFINITY:0.0;
        double z=(x-mu)/sigma,zz=z*z,value;
        if(Math.abs(z)<1e-4) value=Math.log(PHI_ZERO)+Math.log(0.5-zz/8+zz*zz/48)-Math.log(sigma);
        else value=Math.log(PHI_ZERO)+Math.log(-Math.expm1(-zz/2))-2*Math.log(Math.abs(z))-Math.log(sigma);
        return logP?value:Math.exp(value);
    }
    private static double logSmallTail(double z) {
        if(z==0) return -Math.log(2.0);
        double second=Math.log(PHI_ZERO)+Math.log(-Math.expm1(-z*z/2))-Math.log(z);
        return DistributionUtil.logAdd(Normal.cumulative(z,0,1,false,true),second);
    }
    public static double cumulative(double x,double mu,double sigma,boolean lower,boolean logP) {
        if(invalid(mu,sigma)||Double.isNaN(x)) return Double.NaN;
        double z=(x-mu)/sigma,small=logSmallTail(Math.abs(z));
        double value=lower==(z<0)?small:DistributionUtil.logOneMinusExp(small);
        return logP?value:Math.exp(value);
    }
    public static double quantile(double p,double mu,double sigma,boolean lower,boolean logP) {
        if(invalid(mu,sigma)||Double.isNaN(p)||DistributionUtil.invalidProbability(p,logP)) return Double.NaN;
        double target=logP?p:Math.log(p);boolean negative=lower;
        if(target>-Math.log(2.0)){target=DistributionUtil.logOneMinusExp(target);negative=!negative;}
        if(target==-Math.log(2.0))return mu;
        if(target<-40) return mu+(negative?-1:1)*Math.exp(Math.log(sigma)+Math.log(PHI_ZERO)-target);
        double lo=0,hi=1;
        while(logSmallTail(hi)>target) hi*=2;
        for(int i=0;i<80;i++){double mid=lo+(hi-lo)/2;if(mid==lo||mid==hi)break;if(logSmallTail(mid)<=target)hi=mid;else lo=mid;}
        return mu+(negative?-1:1)*sigma*(lo+(hi-lo)/2);
    }
	public static double random(double mu, double sigma, RandomEngine random) {
		if (invalid(mu, sigma)) return Double.NaN;
		return mu + sigma * Normal.random_standard(random) / random.nextDouble();
	}

	@Override public double density(double x, boolean log) { return density(x, mu, sigma, log); }
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, mu, sigma, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, mu, sigma, lowerTail, logP);
	}
	@Override public double random() { return random(mu, sigma, random); }
	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
