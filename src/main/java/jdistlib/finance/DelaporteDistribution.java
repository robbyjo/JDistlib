/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.AtomAwareDistribution;
import jdistlib.Gamma;
import jdistlib.Poisson;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;

/** Delaporte count: Poisson(lambda) plus NB(shape, successProbability). */
public final class DelaporteDistribution extends GenericDistribution implements AtomAwareDistribution,SupportedDistribution{
	private final double lambda,shape,probability;
	public DelaporteDistribution(double lambda,double shape,double probability){if(!(lambda>=0.0)||!(shape>0.0)||!(probability>0.0&&probability<=1.0)||!Double.isFinite(lambda)||!Double.isFinite(shape))throw new IllegalArgumentException("invalid Delaporte parameters");this.lambda=lambda;this.shape=shape;this.probability=probability;}
	@Override public double density(double x,boolean log){if(x<0||x!=Math.rint(x))return log?Double.NEGATIVE_INFINITY:0.0;int n=(int)x;double max=Double.NEGATIVE_INFINITY;double[] terms=new double[n+1];
		for(int k=0;k<=n;k++){int failures=n-k;double logNb=probability==1.0?(failures==0?0.0:Double.NEGATIVE_INFINITY):MathFunctions.lgammafn(failures+shape)-MathFunctions.lgammafn(shape)-MathFunctions.lgammafn(failures+1.0)+shape*Math.log(probability)+failures*Math.log1p(-probability);
			terms[k]=Poisson.density(k,lambda,true)+logNb;max=Math.max(max,terms[k]);}
		if(max==Double.NEGATIVE_INFINITY)return log?max:0.0;double sum=0.0;for(double term:terms)sum+=Math.exp(term-max);double answer=max+Math.log(sum);return log?answer:Math.exp(answer);}
	@Override public double cumulative(double x,boolean lowerTail,boolean logP){if(x<0){double v=lowerTail?0:1;return logP?Math.log(v):v;}int n=(int)Math.floor(x);double value=0.0;for(int k=0;k<=n;k++)value+=density(k,false);value=Math.min(1.0,value);if(!lowerTail)value=1-value;return logP?Math.log(value):value;}
	@Override public double quantile(double p,boolean lowerTail,boolean logP){if(logP)p=Math.exp(p);if(!lowerTail)p=1-p;if(p<0||p>1)return Double.NaN;double c=0;for(int n=0;n<100000;n++){c+=density(n,false);if(c>=p)return n;}return Double.POSITIVE_INFINITY;}
	@Override public double random(){double mixing=probability==1.0?0.0:Gamma.random(shape,(1.0-probability)/probability,random);return Poisson.random(lambda+mixing,random);}
	@Override public double atomProbability(double x){return density(x,false);}@Override public double getLowerBound(){return 0;}@Override public double getUpperBound(){return Double.POSITIVE_INFINITY;}
}
