/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import java.util.Arrays;
import jdistlib.AtomAwareDistribution;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Atom-aware risk-neutral law recovered from a convex piecewise-linear call curve. */
public final class OptionImpliedDistribution extends GenericDistribution
		implements AtomAwareDistribution, SupportedDistribution {
	private final double[] locations,masses,cumulative;
	OptionImpliedDistribution(double[] strikes,double[] calls){
		if(strikes.length<2||strikes.length!=calls.length||strikes[0]!=0.0||calls[calls.length-1]!=0.0)
			throw new IllegalArgumentException("curve must span strike zero through a zero-call boundary");
		double[] slopes=new double[strikes.length-1];for(int i=0;i<slopes.length;i++)slopes[i]=(calls[i+1]-calls[i])/(strikes[i+1]-strikes[i]);
		locations=strikes.clone();masses=new double[strikes.length];masses[0]=1.0+slopes[0];
		for(int i=1;i<strikes.length-1;i++)masses[i]=slopes[i]-slopes[i-1];masses[masses.length-1]=-slopes[slopes.length-1];
		cumulative=new double[masses.length];double sum=0.0;for(int i=0;i<masses.length;i++){masses[i]=Math.max(0.0,masses[i]);sum+=masses[i];}
		for(int i=0;i<masses.length;i++){masses[i]/=sum;cumulative[i]=(i==0?0.0:cumulative[i-1])+masses[i];}
	}
	@Override public double density(double x,boolean log){double value=atomProbability(x);return log?Math.log(value):value;}
	@Override public double cumulative(double x,boolean lowerTail,boolean logP){int index=Arrays.binarySearch(locations,x);if(index<0)index=-index-2;
		double value=index<0?0.0:cumulative[Math.min(index,cumulative.length-1)];value=Math.max(0.0,Math.min(1.0,value));if(!lowerTail)value=1.0-value;return logP?Math.log(value):value;}
	@Override public double quantile(double p,boolean lowerTail,boolean logP){if(logP)p=Math.exp(p);if(!lowerTail)p=1.0-p;if(p<0||p>1||Double.isNaN(p))return Double.NaN;
		int index=Arrays.binarySearch(cumulative,p);if(index<0)index=-index-1;return locations[Math.min(index,locations.length-1)];}
	@Override public double random(){return quantile(random.nextDouble(),true,false);}
	@Override public double atomProbability(double x){int index=Arrays.binarySearch(locations,x);return index<0?0.0:masses[index];}
	@Override public double getLowerBound(){return locations[0];}@Override public double getUpperBound(){return locations[locations.length-1];}
	public double[] getAtomLocations(){return locations.clone();}public double[] getAtomMasses(){return masses.clone();}
}
