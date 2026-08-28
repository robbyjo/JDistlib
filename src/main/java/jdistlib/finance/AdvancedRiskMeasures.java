/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.Complex;

/** Spectral, distortion/Choquet, and entropic risk measures. */
public final class AdvancedRiskMeasures {
	private AdvancedRiskMeasures() {}
	public interface SpectralWeight { double weight(double cumulativeProbability); }
	public interface Distortion { double value(double probability); }
	public static NumericalEstimate spectral(GenericDistribution distribution,RiskConvention convention,SpectralWeight spectrum){
		if(distribution==null||convention==null||spectrum==null)throw new IllegalArgumentException("distribution, convention, and spectrum required");
		double coarse=spectral(distribution,convention,spectrum,4096),fine=spectral(distribution,convention,spectrum,8192);
		return new NumericalEstimate(fine,Math.abs(fine-coarse),Double.isFinite(fine),12288,"spectral-quantile-risk",Double.isFinite(fine)?"":"weighted moment does not exist");}
	/** Choquet expectation integral Q(p) d g(p), with endpoint-preserving distortion g. */
	public static NumericalEstimate distortedExpectation(GenericDistribution distribution,RiskConvention convention,Distortion distortion){
		if(distribution==null||convention==null||distortion==null)throw new IllegalArgumentException("distribution, convention, and distortion required");
		double g0=distortion.value(0.0),g1=distortion.value(1.0);if(Math.abs(g0)>1e-12||Math.abs(g1-1.0)>1e-12)throw new IllegalArgumentException("distortion must map 0 to 0 and 1 to 1");
		double coarse=distorted(distribution,convention,distortion,4096),fine=distorted(distribution,convention,distortion,8192);
		return new NumericalEstimate(fine,Math.abs(fine-coarse),Double.isFinite(fine),12288,"distorted-Choquet-expectation",Double.isFinite(fine)?"":"distorted moment does not exist");}
	/** Entropic risk log E exp(theta*loss)/theta with an explicit MGF-domain check. */
	public static NumericalEstimate entropic(GenericDistribution distribution,double theta,RiskConvention convention){
		if(distribution==null||convention==null||theta==0.0||!Double.isFinite(theta))throw new IllegalArgumentException("distribution/convention and nonzero finite theta required");
		double argument=convention==RiskConvention.LOSS?theta:-theta;Complex logMgf=DistributionTransforms.logMomentGenerating(distribution,argument);
		double value=logMgf.real()/theta;boolean exists=Double.isFinite(value)&&Math.abs(logMgf.imaginary())<1e-8;
		return new NumericalEstimate(value,0.0,exists,1,"entropic-risk",exists?"":"required exponential moment does not exist");}
	/** Exponential Wang distortion g(p)=Phi(Phi^-1(p)+shift). */
	public static Distortion wang(double shift){if(!Double.isFinite(shift))throw new IllegalArgumentException("finite Wang shift required");return p->{if(p<=0.0)return 0.0;if(p>=1.0)return 1.0;double z=jdistlib.Normal.quantile(p,0,1,true,false);return jdistlib.Normal.cumulative(z+shift,0,1,true,false);};}
	private static double spectral(GenericDistribution distribution,RiskConvention convention,SpectralWeight spectrum,int panels){double weighted=0.0,normalizer=0.0;
		for(int i=0;i<panels;i++){double p=(i+0.5)/panels,w=spectrum.weight(p);if(!(w>=0.0)||!Double.isFinite(w))throw new IllegalArgumentException("spectrum must be finite and nonnegative");weighted+=w*lossQuantile(distribution,convention,p);normalizer+=w;}if(!(normalizer>0.0))throw new IllegalArgumentException("spectrum has zero mass");return weighted/normalizer;}
	private static double distorted(GenericDistribution distribution,RiskConvention convention,Distortion distortion,int panels){double total=0.0,previous=distortion.value(0.0);for(int i=0;i<panels;i++){double next=distortion.value((i+1.0)/panels);if(next<previous||!Double.isFinite(next))throw new IllegalArgumentException("distortion must be finite and nondecreasing");total+=lossQuantile(distribution,convention,(i+0.5)/panels)*(next-previous);previous=next;}return total;}
	private static double lossQuantile(GenericDistribution distribution,RiskConvention convention,double p){return convention==RiskConvention.LOSS?distribution.quantile(p,true,false):-distribution.quantile(1.0-p,true,false);}
}
