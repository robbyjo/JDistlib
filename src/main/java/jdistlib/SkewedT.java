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
import static jdistlib.math.Constants.M_LN2;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.RandomEngine;

/**
 * Skewed T distribution, from skewt package
 *
 */
public class SkewedT extends GenericDistribution {
    private static boolean invalid(double df,double gamma) { return !(df>0.0) || !(gamma>0.0) || !Double.isFinite(gamma); }
    public static final double density(double x,double df,double gamma,boolean logP) {
        if(invalid(df,gamma) || Double.isNaN(x)) return Double.NaN;
        double normalizer=DistributionUtil.logAdd(log(gamma),-log(gamma));
        double value=M_LN2-normalizer+T.density(x<0?gamma*x:x/gamma,df,true);
        return logP?value:exp(value);
    }
    public static final double cumulative(double x,double df,double gamma,boolean lower,boolean logP) {
        if(invalid(df,gamma) || Double.isNaN(x)) return Double.NaN;
        double lv=2*log(gamma),den=jdistlib.math.MathFunctions.log1pexp(lv);
        double logTail=x<0 ? M_LN2-den+T.cumulative(gamma*x,df,true,true)
                : M_LN2+lv-den+T.cumulative(x/gamma,df,false,true);
        double value=(lower==(x<0))?logTail:DistributionUtil.logOneMinusExp(logTail);
        return logP?value:exp(value);
    }
    public static final double quantile(double p,double df,double gamma,boolean lower,boolean logP) {
        if(invalid(df,gamma) || Double.isNaN(p) || DistributionUtil.invalidProbability(p,logP)) return Double.NaN;
        double lp=logP?p:log(p),lq=logP?DistributionUtil.logOneMinusExp(p):Math.log1p(-p);
        if(!lower){double t=lp;lp=lq;lq=t;}
        double lv=2*log(gamma),den=jdistlib.math.MathFunctions.log1pexp(lv);
        return lp < -den ? T.quantile(lp+den-M_LN2,df,true,true)/gamma
                : gamma*T.quantile(lq+den-lv-M_LN2,df,false,true);
    }
	public static final double random(double df, double gamma, RandomEngine random) {
		double u1 = random.nextDouble();
		u1 = (int) (134217728 * u1) + random.nextDouble();
		u1 = quantile(u1 / 134217728, df, gamma, true, false);
		return u1;
	}

	public static final double[] random(int n, double df, double gamma, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(df, gamma, random);
		return rand;
	}

	protected double df, gamma;

	public SkewedT(double df, double gamma) {
		this.df = df; this.gamma = gamma;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, df, gamma, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, df, gamma, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, df, gamma, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(df, gamma, random);
	}
}
