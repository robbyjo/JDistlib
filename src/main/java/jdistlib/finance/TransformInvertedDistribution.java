/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;

/** Shared checked Fourier D/P/Q/R implementation for transform-defined laws. */
abstract class TransformInvertedDistribution extends GenericDistribution
		implements TransformDistribution,SupportedDistribution {
	private final FourierInversionOptions inversionOptions;
	TransformInvertedDistribution(FourierInversionOptions inversionOptions){
		this.inversionOptions=inversionOptions==null?FourierInversionOptions.defaults():inversionOptions;
	}
	public FourierInversionOptions getInversionOptions(){return inversionOptions;}
	public NumericalEstimate densityResult(double x){return DistributionTransforms.densityAdaptive(this,x,inversionOptions);}
	public NumericalEstimate cumulativeResult(double x){return DistributionTransforms.cumulativeAdaptive(this,x,inversionOptions);}
	@Override public double density(double x,boolean log){double value=densityResult(x).getValue();return log?Math.log(value):value;}
	@Override public double cumulative(double x,boolean lowerTail,boolean logP){double value=cumulativeResult(x).getValue();if(!lowerTail)value=1.0-value;return logP?Math.log(value):value;}
	@Override public double quantile(double p,boolean lowerTail,boolean logP){if(logP)p=Math.exp(p);if(!lowerTail)p=1.0-p;if(p<0.0||p>1.0||Double.isNaN(p))return Double.NaN;
		if(p==0.0)return Double.NEGATIVE_INFINITY;if(p==1.0)return Double.POSITIVE_INFINITY;double center=locationCenter(),span=initialSpan(),low=center-span,high=center+span;
		for(int guard=0;guard<60&&cumulative(low,true,false)>p;guard++){span*=2.0;low=center-span;}
		span=initialSpan();for(int guard=0;guard<60&&cumulative(high,true,false)<p;guard++){span*=2.0;high=center+span;}
		for(int i=0;i<64;i++){double middle=low+(high-low)/2.0;if(cumulative(middle,true,false)<p)low=middle;else high=middle;}return low+(high-low)/2.0;}
	@Override public double random(){double p;do p=random.nextDouble();while(p==0.0);return quantile(p,true,false);}
	@Override public double getLowerBound(){return Double.NEGATIVE_INFINITY;}@Override public double getUpperBound(){return Double.POSITIVE_INFINITY;}
	abstract double locationCenter();abstract double initialSpan();
}
