/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.math.Complex;

/** Meixner return law in the (scale, skew, shape, location) parameterization. */
public final class MeixnerDistribution extends TransformInvertedDistribution {
	private final double scale,skew,shape,location;
	public MeixnerDistribution(double scale,double skew,double shape,double location){this(scale,skew,shape,location,null);}
	public MeixnerDistribution(double scale,double skew,double shape,double location,FourierInversionOptions options){
		super(options);if(!(scale>0.0)||!(skew>-Math.PI&&skew<Math.PI)||!(shape>0.0)||!Double.isFinite(scale+skew+shape+location))throw new IllegalArgumentException("Meixner requires scale/shape>0, skew in (-pi,pi), and finite location");
		this.scale=scale;this.skew=skew;this.shape=shape;this.location=location;
	}
	@Override public Complex logCharacteristic(double frequency){Complex denominator=new Complex(scale*frequency/2.0,-skew/2.0).cosh();
		Complex ratio=new Complex(Math.cos(skew/2.0),0.0).divide(denominator).log();return new Complex(2.0*shape*ratio.real(),location*frequency+2.0*shape*ratio.imaginary());}
	@Override public Complex logMomentGenerating(double argument){if(!momentGeneratingDomain().contains(argument))return new Complex(Double.POSITIVE_INFINITY,0.0);
		return new Complex(location*argument+2.0*shape*(Math.log(Math.cos(skew/2.0))-Math.log(Math.cos((scale*argument+skew)/2.0))),0.0);}
	@Override public TransformDomain momentGeneratingDomain(){return new TransformDomain((-Math.PI-skew)/scale,false,(Math.PI-skew)/scale,false);}
	public double getScale(){return scale;}public double getSkew(){return skew;}public double getShape(){return shape;}public double getLocation(){return location;}
	@Override double locationCenter(){return location+shape*scale*Math.tan(skew/2.0);}@Override double initialSpan(){return Math.max(1.0,scale*Math.sqrt(shape)/(Math.cos(skew/2.0)));}
}
