/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;

/** Fits caller-supplied parametric terminal laws directly to European option quotes. */
public final class OptionCalibration {
	private OptionCalibration() {}
	public interface Family extends DistributionFit.ParametricFamily {}
	public static Result fit(double forward,double discount,OptionObservation[] observations,
			Family family,double[] initial,int iterations){
		if(!(forward>0.0)||!(discount>0.0)||observations==null||observations.length<2||family==null)
			throw new IllegalArgumentException("forward, discount, observations, and family required");
		DistributionFit.CalibrationLoss loss=distribution->{double total=0.0;
			for(OptionObservation observation:observations){double predicted=discount*payoff(distribution,observation.getStrike(),observation.isCall());
				double residual=0.0;if(predicted<observation.getBid())residual=predicted-observation.getBid();
				else if(predicted>observation.getAsk())residual=predicted-observation.getAsk();total+=observation.getWeight()*residual*residual;}
			return total;};
		DistributionFit.Result fit=DistributionFit.fit(null,family,initial,null,loss,iterations,1e-6);
		double[] residuals=new double[observations.length];GenericDistribution law=fit.getDistribution();double maximum=0.0;
		for(int i=0;i<residuals.length;i++){residuals[i]=discount*payoff(law,observations[i].getStrike(),observations[i].isCall())-observations[i].getMid();maximum=Math.max(maximum,Math.abs(residuals[i]));}
		boolean identifiable=observations.length>=family.parameterCount()+1;
		return new Result(fit,residuals,identifiable,maximum,identifiable?"":"fewer independent quotes than parameters plus one");
	}
	private static double payoff(GenericDistribution law,double strike,boolean call){
		int panels=2048;double sum=0.0;for(int i=0;i<panels;i++){double value=law.quantile((i+0.5)/panels,true,false);sum+=Math.max(call?value-strike:strike-value,0.0);}return sum/panels;
	}
	public static final class Result{
		private final DistributionFit.Result fit;private final double[] residuals;private final boolean identifiable;private final double maximumResidual;private final String warning;
		private Result(DistributionFit.Result fit,double[] residuals,boolean identifiable,double maximumResidual,String warning){this.fit=fit;this.residuals=residuals;this.identifiable=identifiable;this.maximumResidual=maximumResidual;this.warning=warning;}
		public DistributionFit.Result getFit(){return fit;}public double[] getResiduals(){return residuals.clone();}
		public boolean isIdentifiable(){return identifiable;}public double getMaximumResidual(){return maximumResidual;}public String getWarning(){return warning;}
	}
}
