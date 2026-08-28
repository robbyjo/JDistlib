/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.math.Complex;
import jdistlib.math.MathFunctions;

/** Normal-tempered-stable law defined by a tempered-stable normal variance mixture. */
public final class NormalTemperedStableDistribution extends TransformInvertedDistribution {
	private final double alpha,tempering,intensity,skew,scale,location,gamma;
	public NormalTemperedStableDistribution(double alpha,double tempering,double intensity,double skew,double scale,double location){this(alpha,tempering,intensity,skew,scale,location,null);}
	public NormalTemperedStableDistribution(double alpha,double tempering,double intensity,double skew,double scale,double location,FourierInversionOptions options){
		super(options);if(!(alpha>0.0&&alpha<1.0)||!(tempering>0.0)||!(intensity>0.0)||!(scale>0.0)
				||!Double.isFinite(alpha+tempering+intensity+skew+scale+location))throw new IllegalArgumentException("NTS requires alpha in (0,1), positive tempering/intensity/scale, and finite parameters");
		this.alpha=alpha;this.tempering=tempering;this.intensity=intensity;this.skew=skew;this.scale=scale;this.location=location;this.gamma=MathFunctions.gammafn(-alpha);
	}
	@Override public Complex logCharacteristic(double frequency){if(frequency==0.0)return Complex.ZERO;Complex base=new Complex(tempering+0.5*scale*scale*frequency*frequency,-skew*frequency);
		Complex power=base.pow(new Complex(alpha,0.0)).subtract(new Complex(Math.pow(tempering,alpha),0.0));return new Complex(intensity*gamma*power.real(),location*frequency+intensity*gamma*power.imaginary());}
	@Override public Complex logMomentGenerating(double argument){if(!momentGeneratingDomain().contains(argument))return new Complex(Double.POSITIVE_INFINITY,0.0);
		double base=tempering-skew*argument-0.5*scale*scale*argument*argument;return new Complex(location*argument+intensity*gamma*(Math.pow(base,alpha)-Math.pow(tempering,alpha)),0.0);}
	@Override public TransformDomain momentGeneratingDomain(){double root=Math.sqrt(skew*skew+2.0*scale*scale*tempering);return new TransformDomain((-skew-root)/(scale*scale),false,(-skew+root)/(scale*scale),false);}
	public double getAlpha(){return alpha;}public double getTempering(){return tempering;}public double getIntensity(){return intensity;}public double getSkew(){return skew;}public double getScale(){return scale;}public double getLocation(){return location;}
	@Override double locationCenter(){return location+intensity*(-gamma)*alpha*Math.pow(tempering,alpha-1.0)*skew;}
	@Override double initialSpan(){return Math.max(1.0,scale*Math.sqrt(intensity*(-gamma)*alpha*Math.pow(tempering,alpha-1.0)));}
}
