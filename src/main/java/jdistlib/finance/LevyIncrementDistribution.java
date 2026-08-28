/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.Complex;

/** Time-scaled increment of an infinitely-divisible transform-defined unit law. */
public final class LevyIncrementDistribution extends TransformInvertedDistribution {
	private final TransformDistribution unitIncrement;private final double time,center,span;
	public LevyIncrementDistribution(TransformDistribution unitIncrement,double time){this(unitIncrement,time,null);}
	public LevyIncrementDistribution(TransformDistribution unitIncrement,double time,FourierInversionOptions options){super(options);if(unitIncrement==null||!(time>0.0)||!Double.isFinite(time))throw new IllegalArgumentException("unit increment and positive finite time required");this.unitIncrement=unitIncrement;this.time=time;
		double candidateCenter=0.0,candidateSpan=1.0,h=1e-4;
		if(unitIncrement.momentGeneratingDomain().contains(-h)&&unitIncrement.momentGeneratingDomain().contains(h)){
			double minus=unitIncrement.logMomentGenerating(-h).real(),zero=unitIncrement.logMomentGenerating(0.0).real(),plus=unitIncrement.logMomentGenerating(h).real();
			candidateCenter=time*(plus-minus)/(2*h);candidateSpan=Math.sqrt(Math.max(0.0,time*(plus-2*zero+minus)/(h*h)));}
		if((!Double.isFinite(candidateCenter)||!(candidateSpan>0.0))&&unitIncrement instanceof GenericDistribution){GenericDistribution law=(GenericDistribution)unitIncrement;
			candidateCenter=time*law.quantile(0.5,true,false);candidateSpan=Math.sqrt(time)*Math.abs(law.quantile(0.84,true,false)-law.quantile(0.5,true,false));}
		this.center=Double.isFinite(candidateCenter)?candidateCenter:0.0;this.span=Double.isFinite(candidateSpan)&&candidateSpan>0.0?Math.max(1.0,candidateSpan):1.0;}
	public double getTime(){return time;}public TransformDistribution getUnitIncrement(){return unitIncrement;}
	@Override public Complex logCharacteristic(double frequency){Complex value=unitIncrement.logCharacteristic(frequency);return new Complex(time*value.real(),time*value.imaginary());}
	@Override public Complex logMomentGenerating(double argument){Complex value=unitIncrement.logMomentGenerating(argument);return new Complex(time*value.real(),time*value.imaginary());}
	@Override public TransformDomain momentGeneratingDomain(){return unitIncrement.momentGeneratingDomain();}
	/** Exact composition of independent increments sharing this unit exponent. */
	public LevyIncrementDistribution plus(LevyIncrementDistribution other){if(other==null||other.unitIncrement!=unitIncrement)throw new IllegalArgumentException("increments must share the same unit-transform object");return new LevyIncrementDistribution(unitIncrement,time+other.time,getInversionOptions());}
	@Override double locationCenter(){return center;}@Override double initialSpan(){return span;}
}
