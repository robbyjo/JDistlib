/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.math.Complex;
import jdistlib.math.MathFunctions;

/** Positive tempered-stable subordinator increment with Laplace exponent. */
public final class PositiveTemperedStableDistribution extends TransformInvertedDistribution {
	private final double alpha,tempering,intensity,gamma;
	public PositiveTemperedStableDistribution(double alpha,double tempering,double intensity){super(null);
		if(!(alpha>0.0&&alpha<1.0)||!(tempering>0.0)||!(intensity>0.0))throw new IllegalArgumentException("positive tempered stable requires alpha in (0,1), tempering/intensity>0");
		this.alpha=alpha;this.tempering=tempering;this.intensity=intensity;this.gamma=MathFunctions.gammafn(-alpha);}
	@Override public Complex logCharacteristic(double frequency){Complex power=new Complex(tempering,-frequency).pow(new Complex(alpha,0.0));Complex difference=power.subtract(new Complex(Math.pow(tempering,alpha),0.0));return new Complex(intensity*gamma*difference.real(),intensity*gamma*difference.imaginary());}
	@Override public Complex logMomentGenerating(double argument){if(!momentGeneratingDomain().contains(argument))return new Complex(Double.POSITIVE_INFINITY,0.0);return new Complex(intensity*gamma*(Math.pow(tempering-argument,alpha)-Math.pow(tempering,alpha)),0.0);}
	@Override public TransformDomain momentGeneratingDomain(){return new TransformDomain(Double.NEGATIVE_INFINITY,false,tempering,false);}
	public double getAlpha(){return alpha;}public double getTempering(){return tempering;}public double getIntensity(){return intensity;}
	@Override public double getLowerBound(){return 0.0;}@Override double locationCenter(){return intensity*(-gamma)*alpha*Math.pow(tempering,alpha-1.0);}@Override double initialSpan(){return Math.max(1.0,locationCenter());}
	@Override public double density(double x,boolean log){if(x<=0.0)return log?Double.NEGATIVE_INFINITY:0.0;return super.density(x,log);}
	@Override public double cumulative(double x,boolean lowerTail,boolean logP){if(x<=0.0){double p=lowerTail?0.0:1.0;return logP?Math.log(p):p;}return super.cumulative(x,lowerTail,logP);}
	@Override public double quantile(double p,boolean lowerTail,boolean logP){double probability=logP?Math.exp(p):p;if(!lowerTail)probability=1.0-probability;if(probability==0.0)return 0.0;return super.quantile(p,lowerTail,logP);}
}
