/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.NumericalContinuousDistribution;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.Complex;

/** Generalized-inverse-Gaussian law in (lambda, chi, psi), chi/psi positive. */
public final class GeneralizedInverseGaussianDistribution extends GenericDistribution
		implements SupportedDistribution,TransformDistribution {
	private final double lambda,chi,psi,logConstant;
	private final NumericalContinuousDistribution numerical;
	public GeneralizedInverseGaussianDistribution(double lambda,double chi,double psi){
		if(!Double.isFinite(lambda)||!(chi>0.0)||!(psi>0.0)||!Double.isFinite(chi+psi))throw new IllegalArgumentException("GIG requires finite lambda and positive chi/psi");
		this.lambda=lambda;this.chi=chi;this.psi=psi;double root=Math.sqrt(chi*psi);
		this.logConstant=0.5*lambda*(Math.log(psi)-Math.log(chi))-Math.log(2.0)-GeneralizedHyperbolicDistribution.logBesselK(root,lambda);
		this.numerical=NumericalContinuousDistribution.builder().logKernel(x->density(x,true)).support(0.0,Double.POSITIVE_INFINITY).withoutAnalysis().build();
	}
	@Override public double density(double x,boolean log){if(!(x>0.0))return log?Double.NEGATIVE_INFINITY:0.0;
		double value=logConstant+(lambda-1.0)*Math.log(x)-0.5*(chi/x+psi*x);return log?value:Math.exp(value);}
	@Override public double cumulative(double x,boolean lowerTail,boolean logP){return numerical.cumulative(x,lowerTail,logP);}
	@Override public double quantile(double p,boolean lowerTail,boolean logP){return numerical.quantile(p,lowerTail,logP);}
	@Override public double random(){return numerical.quantile(random.nextDouble(),true,false);}
	@Override public Complex logMomentGenerating(double argument){if(!momentGeneratingDomain().contains(argument))return new Complex(Double.POSITIVE_INFINITY,0.0);
		double adjusted=psi-2.0*argument;double value=0.5*lambda*(Math.log(psi)-Math.log(adjusted))
				+GeneralizedHyperbolicDistribution.logBesselK(Math.sqrt(chi*adjusted),lambda)
				-GeneralizedHyperbolicDistribution.logBesselK(Math.sqrt(chi*psi),lambda);return new Complex(value,0.0);}
	@Override public Complex logCharacteristic(double frequency){if(frequency==0.0)return Complex.ZERO;int panels=16384;double real=0.0,imaginary=0.0;
		for(int i=0;i<panels;i++){double u=(i+0.5)/panels,remaining=1.0-u,x=u/remaining,weight=density(x,false)/(remaining*remaining);real+=weight*Math.cos(frequency*x);imaginary+=weight*Math.sin(frequency*x);}return new Complex(real/panels,imaginary/panels).log();}
	@Override public TransformDomain momentGeneratingDomain(){return new TransformDomain(Double.NEGATIVE_INFINITY,false,psi/2.0,false);}
	@Override public double getLowerBound(){return 0.0;}@Override public double getUpperBound(){return Double.POSITIVE_INFINITY;}
	public double getLambda(){return lambda;}public double getChi(){return chi;}public double getPsi(){return psi;}
}
