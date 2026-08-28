/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import java.util.Arrays;
import jdistlib.AtomAwareDistribution;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Immutable equally-spaced finite distribution used by exact/FFT/Panjer grids. */
public final class FiniteGridDistribution extends GenericDistribution implements AtomAwareDistribution,SupportedDistribution {
	private final double origin,step;private final double[] probabilities,cumulative;
	public FiniteGridDistribution(double origin,double step,double[] probabilities){if(!Double.isFinite(origin)||!(step>0.0)||probabilities==null||probabilities.length==0)throw new IllegalArgumentException("finite origin, positive step, and probabilities required");
		this.origin=origin;this.step=step;this.probabilities=probabilities.clone();this.cumulative=new double[probabilities.length];double total=0.0;
		for(double p:this.probabilities){if(!(p>=0.0)||!Double.isFinite(p))throw new IllegalArgumentException("finite nonnegative probabilities required");total+=p;}if(!(total>0.0))throw new IllegalArgumentException("positive total mass required");
		for(int i=0;i<this.probabilities.length;i++){this.probabilities[i]/=total;cumulative[i]=(i==0?0.0:cumulative[i-1])+this.probabilities[i];}cumulative[cumulative.length-1]=1.0;}
	public double getOrigin(){return origin;}public double getStep(){return step;}public double[] getProbabilities(){return probabilities.clone();}
	@Override public double density(double x,boolean log){double p=atomProbability(x);return log?Math.log(p):p;}
	@Override public double cumulative(double x,boolean lowerTail,boolean logP){int index=(int)Math.floor((x-origin)/step+1e-12);double p=index<0?0.0:index>=cumulative.length?1.0:cumulative[index];if(!lowerTail)p=1.0-p;return logP?Math.log(p):p;}
	@Override public double quantile(double p,boolean lowerTail,boolean logP){if(logP)p=Math.exp(p);if(!lowerTail)p=1.0-p;if(p<0.0||p>1.0||Double.isNaN(p))return Double.NaN;int index=Arrays.binarySearch(cumulative,p);if(index<0)index=-index-1;return origin+Math.min(index,probabilities.length-1)*step;}
	@Override public double random(){return quantile(random.nextDouble(),true,false);}
	@Override public double atomProbability(double x){double coordinate=(x-origin)/step;long index=Math.round(coordinate);return Math.abs(coordinate-index)<=1e-10&&index>=0&&index<probabilities.length?probabilities[(int)index]:0.0;}
	@Override public double getLowerBound(){return origin;}@Override public double getUpperBound(){return origin+(probabilities.length-1)*step;}
}
