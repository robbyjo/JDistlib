/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.AtomAwareDistribution;
import jdistlib.Poisson;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;

/** Polya-Aeppli count: Poisson clusters with shifted-geometric cluster sizes. */
public final class PolyaAeppliDistribution extends GenericDistribution implements AtomAwareDistribution,SupportedDistribution{
	private final double lambda,probability;
	public PolyaAeppliDistribution(double lambda,double probability){if(!(lambda>=0.0)||!(probability>0.0&&probability<=1.0)||!Double.isFinite(lambda))throw new IllegalArgumentException("invalid Polya-Aeppli parameters");this.lambda=lambda;this.probability=probability;}
	@Override public double density(double x,boolean log){if(x<0||x!=Math.rint(x))return log?Double.NEGATIVE_INFINITY:0.0;int n=(int)x;if(n==0)return log?-lambda:Math.exp(-lambda);if(lambda==0.0)return log?Double.NEGATIVE_INFINITY:0.0;double max=Double.NEGATIVE_INFINITY;double[] terms=new double[n];
		for(int clusters=1;clusters<=n;clusters++){double geometric=probability==1.0?(n==clusters?0.0:Double.NEGATIVE_INFINITY):clusters*Math.log(probability)+(n-clusters)*Math.log1p(-probability);
			double term=-lambda+clusters*Math.log(lambda)-MathFunctions.lgammafn(clusters+1.0)+MathFunctions.lgammafn(n)-MathFunctions.lgammafn(clusters)-MathFunctions.lgammafn(n-clusters+1.0)+geometric;terms[clusters-1]=term;max=Math.max(max,term);}
		double sum=0.0;for(double term:terms)sum+=Math.exp(term-max);double answer=max+Math.log(sum);return log?answer:Math.exp(answer);}
	@Override public double cumulative(double x,boolean lowerTail,boolean logP){if(x<0){double v=lowerTail?0:1;return logP?Math.log(v):v;}double value=0;for(int n=0;n<=(int)Math.floor(x);n++)value+=density(n,false);value=Math.min(1,value);if(!lowerTail)value=1-value;return logP?Math.log(value):value;}
	@Override public double quantile(double p,boolean lowerTail,boolean logP){if(logP)p=Math.exp(p);if(!lowerTail)p=1-p;if(p<0||p>1)return Double.NaN;double c=0;for(int n=0;n<100000;n++){c+=density(n,false);if(c>=p)return n;}return Double.POSITIVE_INFINITY;}
	@Override public double random(){int clusters=(int)Poisson.random(lambda,random),total=0;for(int i=0;i<clusters;i++)total+=1+(int)Math.floor(Math.log(Math.max(random.nextDouble(),Double.MIN_VALUE))/Math.log1p(-probability));return total;}
	@Override public double atomProbability(double x){return density(x,false);}@Override public double getLowerBound(){return 0;}@Override public double getUpperBound(){return Double.POSITIVE_INFINITY;}
}
