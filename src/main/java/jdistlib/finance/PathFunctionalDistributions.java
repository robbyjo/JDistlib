/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Exact iid-observation extrema and simulated drawdowns for iid discrete-time increments. */
public final class PathFunctionalDistributions {
	private PathFunctionalDistributions() {}
	public static OrderStatisticDistribution runningMaximum(GenericDistribution observations,int steps){return OrderStatisticDistribution.maximum(observations,steps);}
	public static OrderStatisticDistribution runningMinimum(GenericDistribution observations,int steps){return OrderStatisticDistribution.minimum(observations,steps);}
	public static DistributionApproximation maximumDrawdown(GenericDistribution increments,int steps,int draws,long seed){return simulate(increments,steps,draws,seed,true);}
	public static DistributionApproximation terminalDrawdown(GenericDistribution increments,int steps,int draws,long seed){return simulate(increments,steps,draws,seed,false);}
	private static DistributionApproximation simulate(GenericDistribution increments,int steps,int draws,long seed,boolean maximum){if(increments==null||steps<1||draws<100)throw new IllegalArgumentException("increment law, positive steps, and draws>=100 required");
		RandomEngine random=new MersenneTwister(seed);double[] sample=new double[draws];for(int draw=0;draw<draws;draw++){double level=0.0,peak=0.0,maxDrawdown=0.0;for(int step=0;step<steps;step++){level+=increments.quantile(open(random),true,false);peak=Math.max(peak,level);maxDrawdown=Math.max(maxDrawdown,peak-level);}sample[draw]=maximum?maxDrawdown:peak-level;}
		return new DistributionApproximation(new EmpiricalDistribution(sample),new NumericalEstimate(1.0,1.0/Math.sqrt(draws),true,draws*steps,maximum?"Monte-Carlo-maximum-drawdown":"Monte-Carlo-terminal-drawdown","iid increment path model"),seed);}
	private static double open(RandomEngine random){double p;do p=random.nextDouble();while(p==0.0);return p;}
}
