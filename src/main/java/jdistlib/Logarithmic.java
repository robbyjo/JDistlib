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

import static java.lang.Math.*;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * @author Roby Joehanes
 */
public class Logarithmic extends GenericDistribution {
    public static final double density(double x, double mu, boolean logP) {
        if (!(mu>0.0 && mu<1.0) || Double.isNaN(x)) return Double.NaN;
        if (x<1.0 || !Double.isFinite(x) || x!=Math.rint(x)) return logP ? Double.NEGATIVE_INFINITY : 0.0;
        double v=x*log(mu)-log(x)-log(-log1p(-mu));
        return logP ? v : exp(v);
    }
    public static final double cumulative(double x, double mu, boolean lower, boolean logP) {
        if (!(mu>0.0 && mu<1.0) || Double.isNaN(x)) return Double.NaN;
        if(x<1.0) return DistributionUtil.boundary(false,lower,logP);
        if(x==Double.POSITIVE_INFINITY) return DistributionUtil.boundary(true,lower,logP);
        double k=floor(x), lm=log(mu), total=Double.NEGATIVE_INFINITY;
        if(lower || mu>.99 && k<100000) {
            for(double i=1;i<=k;i++) total=DistributionUtil.logAdd(total,i*lm-log(i));
            total-=log(-log1p(-mu));
            total=Math.min(0,total);
            if(!lower) total=DistributionUtil.logOneMinusExp(total);
        } else {
            double term=(k+1)*lm-log(k+1); total=term;
            for(double i=k+1;;i++) {
                term+=lm+log(i/(i+1));
                total=DistributionUtil.logAdd(total,term);
                // Bound all remaining terms by a geometric series.
                if(term+lm-log1p(-mu)<total-37) break;
            }
            total-=log(-log1p(-mu));
        }
        return logP ? total : exp(total);
    }
    public static final double quantile(double p,double mu,boolean lower,boolean logP) {
        return quantile(p,mu,lower,logP,10000);
    }
    /** The explicit maximum bounds the finite search; returns NaN if exceeded. */
    public static final double quantile(double p,double mu,boolean lower,boolean logP,int maximum) {
        if (!(mu>0.0 && mu<1.0) || maximum<1 || Double.isNaN(p) || DistributionUtil.invalidProbability(p,logP)) return Double.NaN;
        double lp=logP?p:log(p);
        if(lp==Double.NEGATIVE_INFINITY) return lower?1:Double.POSITIVE_INFINITY;
        if(lp==0) return lower?Double.POSITIVE_INFINITY:1;
        double value=DistributionUtil.discreteQuantile(lp,lower,true,1,maximum,(x,lt,l)->cumulative(x,mu,lt,l));
        double actual=cumulative(value,mu,lower,true);
        return (lower ? actual>=lp : actual<=lp) ? value : Double.NaN;
    }
	public static final double random(double mu, RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, mu, true, false);
		return u1;
	}

	public static final double[] random(int n, double mu, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(mu, random);
		return rand;
	}

	protected double mu;

	public Logarithmic(double mu) {
		this.mu = mu;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, mu, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, mu, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, mu, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(mu, random);
	}
}
