/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.math.Complex;
import jdistlib.math.MathFunctions;

/** CGMY/KoBoL infinitely-divisible return law in (C,G,M,Y,location) form. */
public final class CgmyDistribution extends TransformInvertedDistribution {
	private final double c,g,m,y,location,gamma;
	public CgmyDistribution(double c,double g,double m,double y,double location){this(c,g,m,y,location,null);}
	public CgmyDistribution(double c,double g,double m,double y,double location,FourierInversionOptions options){
		super(options);if(!(c>0.0)||!(g>0.0)||!(m>0.0)||!(y>0.0&&y<2.0)||Math.abs(y-1.0)<1e-12
				||!Double.isFinite(c+g+m+y+location))throw new IllegalArgumentException("CGMY requires C,G,M>0, Y in (0,2) excluding one, and finite location");
		this.c=c;this.g=g;this.m=m;this.y=y;this.location=location;this.gamma=MathFunctions.gammafn(-y);
	}
	public double getC(){return c;}public double getG(){return g;}public double getM(){return m;}public double getY(){return y;}public double getLocation(){return location;}
	@Override public Complex logCharacteristic(double frequency){if(frequency==0.0)return Complex.ZERO;
		Complex first=new Complex(m,-frequency).pow(new Complex(y,0.0)).subtract(new Complex(Math.pow(m,y),0.0));
		Complex second=new Complex(g,frequency).pow(new Complex(y,0.0)).subtract(new Complex(Math.pow(g,y),0.0));
		Complex jump=first.add(second);return new Complex(c*gamma*jump.real(),location*frequency+c*gamma*jump.imaginary());}
	@Override public Complex logMomentGenerating(double argument){if(!momentGeneratingDomain().contains(argument))return new Complex(Double.POSITIVE_INFINITY,0.0);
		double value=location*argument+c*gamma*(Math.pow(m-argument,y)-Math.pow(m,y)+Math.pow(g+argument,y)-Math.pow(g,y));return new Complex(value,0.0);}
	@Override public TransformDomain momentGeneratingDomain(){return new TransformDomain(-g,false,m,false);}
	@Override double locationCenter(){return location;}@Override double initialSpan(){return Math.max(1.0,Math.sqrt(Math.abs(c*gamma*y*(y-1.0)*(Math.pow(m,y-2.0)+Math.pow(g,y-2.0)))));}
}
