/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import java.util.Arrays;
import jdistlib.generic.GenericDistribution;

/** Bounded MLE/MAP fitting with censored and interval observations. */
public final class DistributionFit {
	private DistributionFit() {}

	public interface ParametricFamily {
		int parameterCount();
		double lowerBound(int parameter);
		double upperBound(int parameter);
		GenericDistribution distribution(double[] parameters);
	}
	public interface LogPrior { double value(double[] parameters); }
	public interface CalibrationLoss { double value(GenericDistribution distribution); }

	public static final class Observation {
		public enum Kind { EXACT, LEFT_CENSORED, RIGHT_CENSORED, INTERVAL }
		private final double lower, upper, weight;
		private final Kind kind;
		private Observation(double lower, double upper, double weight, Kind kind) {
			if (kind == null || !(weight > 0.0) || !Double.isFinite(weight)
					|| Double.isNaN(lower) || Double.isNaN(upper) || lower > upper)
				throw new IllegalArgumentException("invalid observation");
			this.lower=lower; this.upper=upper; this.weight=weight; this.kind=kind;
		}
		public static Observation exact(double value) { return new Observation(value,value,1.0,Kind.EXACT); }
		public static Observation exact(double value,double weight) { return new Observation(value,value,weight,Kind.EXACT); }
		public static Observation leftCensored(double upper) { return new Observation(Double.NEGATIVE_INFINITY,upper,1.0,Kind.LEFT_CENSORED); }
		public static Observation rightCensored(double lower) { return new Observation(lower,Double.POSITIVE_INFINITY,1.0,Kind.RIGHT_CENSORED); }
		public static Observation interval(double lower,double upper) { return new Observation(lower,upper,1.0,Kind.INTERVAL); }
		public double getLower(){return lower;} public double getUpper(){return upper;}
		public double getWeight(){return weight;} public Kind getKind(){return kind;}
	}

	public static final class Result {
		private final double[] parameters, standardErrors;
		private final double objective;
		private final int iterations;
		private final boolean converged;
		private final String message;
		private final GenericDistribution distribution;
		private Result(double[] parameters,double[] standardErrors,double objective,
				int iterations,boolean converged,String message,GenericDistribution distribution){
			this.parameters=parameters.clone();this.standardErrors=standardErrors.clone();this.objective=objective;
			this.iterations=iterations;this.converged=converged;this.message=message;this.distribution=distribution;
		}
		public double[] getParameters(){return parameters.clone();}
		public double[] getStandardErrors(){return standardErrors.clone();}
		public double getObjective(){return objective;} public int getIterations(){return iterations;}
		public boolean isConverged(){return converged;} public String getMessage(){return message;}
		public GenericDistribution getDistribution(){return distribution;}
	}

	public static Result maximumLikelihood(Observation[] observations, ParametricFamily family,
			double[] initial) { return fit(observations,family,initial,null,null,300,1e-7); }

	public static Result fit(Observation[] observations, ParametricFamily family, double[] initial,
			LogPrior prior, CalibrationLoss calibrationLoss, int maximumIterations, double tolerance) {
		if (family==null || initial==null || initial.length!=family.parameterCount()
				|| maximumIterations<1 || !(tolerance>0.0)) throw new IllegalArgumentException("invalid fit request");
		if ((observations==null || observations.length==0) && calibrationLoss==null)
			throw new IllegalArgumentException("observations or calibration loss required");
		double[] current=initial.clone(), step=new double[current.length];
		for(int j=0;j<current.length;j++){
			current[j]=clamp(current[j],family.lowerBound(j),family.upperBound(j));
			double range=family.upperBound(j)-family.lowerBound(j);
			step[j]=Double.isFinite(range)?Math.max(range*0.1,tolerance*10.0):Math.max(Math.abs(current[j])*0.25,0.25);
		}
		double best=objective(observations,family,current,prior,calibrationLoss);
		int iteration=0;
		for(;iteration<maximumIterations;iteration++){
			boolean improved=false;
			for(int j=0;j<current.length;j++){
				double original=current[j];
				for(int direction=-1;direction<=1;direction+=2){
					current[j]=clamp(original+direction*step[j],family.lowerBound(j),family.upperBound(j));
					double candidate=objective(observations,family,current,prior,calibrationLoss);
					if(candidate<best){best=candidate;original=current[j];improved=true;}
				}
				current[j]=original;
			}
			if(!improved){double largest=0.0;for(int j=0;j<step.length;j++){step[j]*=0.5;largest=Math.max(largest,step[j]);}
				if(largest<tolerance)break;}
		}
		double[] errors=standardErrors(observations,family,current,prior,calibrationLoss,best);
		boolean converged=iteration<maximumIterations && Double.isFinite(best);
		return new Result(current,errors,best,iteration,converged,
				converged?"coordinate search converged":"maximum iterations or non-finite objective",family.distribution(current));
	}

	private static double objective(Observation[] observations,ParametricFamily family,double[] parameters,
			LogPrior prior,CalibrationLoss loss){
		try{
			GenericDistribution distribution=family.distribution(parameters);
			double value=prior==null?0.0:-prior.value(parameters);
			if(observations!=null)for(Observation observation:observations){
				double contribution;
				if(observation.kind==Observation.Kind.EXACT) contribution=distribution.density(observation.lower,true);
				else if(observation.kind==Observation.Kind.LEFT_CENSORED) contribution=distribution.cumulative(observation.upper,true,true);
				else if(observation.kind==Observation.Kind.RIGHT_CENSORED) contribution=distribution.cumulative(observation.lower,false,true);
				else { double probability=distribution.cumulative(observation.upper,true,false)-distribution.cumulative(observation.lower,true,false);
					contribution=Math.log(probability); }
				value-=observation.weight*contribution;
			}
			if(loss!=null)value+=loss.value(distribution);
			return Double.isFinite(value)?value:Double.POSITIVE_INFINITY;
		}catch(IllegalArgumentException exception){return Double.POSITIVE_INFINITY;}
	}

	private static double[] standardErrors(Observation[] observations,ParametricFamily family,double[] p,
			LogPrior prior,CalibrationLoss loss,double center){
		double[] answer=new double[p.length], work=p.clone();
		for(int j=0;j<p.length;j++){
			double h=Math.max(1e-5,Math.abs(p[j])*1e-4), original=p[j];
			work[j]=clamp(original+h,family.lowerBound(j),family.upperBound(j));double plus=objective(observations,family,work,prior,loss);
			work[j]=clamp(original-h,family.lowerBound(j),family.upperBound(j));double minus=objective(observations,family,work,prior,loss);
			work[j]=original;double curvature=(plus-2.0*center+minus)/(h*h);
			answer[j]=curvature>0.0?1.0/Math.sqrt(curvature):Double.NaN;
		}
		return answer;
	}
	private static double clamp(double value,double lower,double upper){return Math.max(lower,Math.min(upper,value));}
}
