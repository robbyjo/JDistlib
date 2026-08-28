/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.Normal;
import jdistlib.generic.GenericDistribution;
import jdistlib.inference.ModelFactor;
import jdistlib.inference.ModelState;
import jdistlib.rng.MersenneTwister;

/** Option-price likelihood factors and posterior-predictive distribution adapters. */
public final class OptionInference {
	private OptionInference() {}
	public interface StatePriceModel { double price(ModelState state,OptionObservation observation); }
	public interface StateNoiseModel { double standardDeviation(ModelState state,OptionObservation observation); }
	public static ModelFactor likelihood(OptionObservation[] observations,StatePriceModel model,
			StateNoiseModel noise){
		if(observations==null||observations.length==0||model==null||noise==null)throw new IllegalArgumentException("quotes, price model, and noise model required");
		OptionObservation[] copy=observations.clone();
		return state->{double total=0.0;for(OptionObservation observation:copy){double predicted=model.price(state,observation),sd=noise.standardDeviation(state,observation);
			if(!(sd>0.0)||!Double.isFinite(predicted))return Double.NEGATIVE_INFINITY;
			if(observation.getAsk()>observation.getBid()){double upper=Normal.cumulative((observation.getAsk()-predicted)/sd,0,1,true,false);
				double lower=Normal.cumulative((observation.getBid()-predicted)/sd,0,1,true,false);total+=Math.log(Math.max(upper-lower,Double.MIN_VALUE));}
			else total+=Normal.density(observation.getMid(),predicted,sd,true);}return total;};
	}
	public enum Measure { RISK_NEUTRAL, PHYSICAL_PREDICTIVE }
	public interface DrawDistribution { GenericDistribution distribution(double[] posteriorDraw); }
	public static PosteriorEnsemble posterior(double[][] draws,DrawDistribution factory,Measure measure,long seed,String chainProvenance){
		return new PosteriorEnsemble(draws,factory,measure,seed,chainProvenance);
	}
	public static final class PosteriorEnsemble{
		private final double[][] draws;private final DrawDistribution factory;private final Measure measure;private final long seed;private final String provenance;
		private PosteriorEnsemble(double[][] draws,DrawDistribution factory,Measure measure,long seed,String provenance){
			if(draws==null||draws.length==0||factory==null||measure==null)throw new IllegalArgumentException("posterior draws, factory, and measure required");
			this.draws=new double[draws.length][];for(int i=0;i<draws.length;i++){if(draws[i]==null)throw new IllegalArgumentException("null draw");this.draws[i]=draws[i].clone();}
			this.factory=factory;this.measure=measure;this.seed=seed;this.provenance=provenance==null?"":provenance;
		}
		public Measure getMeasure(){return measure;}public long getSeed(){return seed;}public String getChainProvenance(){return provenance;}
		public DistributionApproximation terminalPriceDistribution(){MersenneTwister random=new MersenneTwister(seed);double[] values=new double[draws.length];
			for(int i=0;i<draws.length;i++)values[i]=factory.distribution(draws[i]).quantile(open(random),true,false);
			return new DistributionApproximation(new EmpiricalDistribution(values),new NumericalEstimate(1.0,1.0/Math.sqrt(values.length),true,values.length,"posterior-predictive-mixture","MC error depends on retained draws"),seed);}
		public DistributionApproximation payoffDistribution(double strike,boolean call){MersenneTwister random=new MersenneTwister(seed);double[] values=new double[draws.length];
			for(int i=0;i<draws.length;i++){double terminal=factory.distribution(draws[i]).quantile(open(random),true,false);values[i]=Math.max(call?terminal-strike:strike-terminal,0.0);}
			return new DistributionApproximation(new EmpiricalDistribution(values),new NumericalEstimate(1.0,1.0/Math.sqrt(values.length),true,values.length,"posterior-predictive-payoff",""),seed);}
		public NumericalEstimate tailProbability(double threshold,Tail tail){double[] values=new double[draws.length];for(int i=0;i<draws.length;i++)values[i]=factory.distribution(draws[i]).cumulative(threshold,tail==Tail.LOWER,false);return summarize(values,"posterior-tail-probability");}
		public NumericalEstimate valueAtRisk(double level,RiskConvention convention){double[] values=new double[draws.length];for(int i=0;i<draws.length;i++)values[i]=FinancialRisk.valueAtRisk(factory.distribution(draws[i]),level,convention);return summarize(values,"posterior-VaR");}
		public NumericalEstimate expectedShortfall(double level,RiskConvention convention){double[] values=new double[draws.length];for(int i=0;i<draws.length;i++)values[i]=FinancialRisk.expectedShortfall(factory.distribution(draws[i]),level,convention).getValue();return summarize(values,"posterior-expected-shortfall");}
		private NumericalEstimate summarize(double[] values,String strategy){double mean=0.0;for(double v:values)mean+=v;mean/=values.length;double variance=0.0;for(double v:values)variance+=(v-mean)*(v-mean);
			return new NumericalEstimate(mean,Math.sqrt(variance/Math.max(1,values.length-1)/values.length),Double.isFinite(mean),values.length,strategy,"MCSE assumes retained draws are effectively independent");}
		private static double open(MersenneTwister random){double p;do p=random.nextDouble();while(p==0.0);return p;}
	}
}
