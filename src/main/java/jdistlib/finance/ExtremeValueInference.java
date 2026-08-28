/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import java.util.Arrays;
import jdistlib.evd.GEV;
import jdistlib.evd.GeneralizedPareto;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;

/** GEV/GPD fitting, tail-index estimators, return levels, and threshold diagnostics. */
public final class ExtremeValueInference {
	private ExtremeValueInference() {}
	public static DistributionFit.Result fitGev(double[] blockMaxima) {
		double[] sorted=checked(blockMaxima);double mean=mean(sorted), sd=standardDeviation(sorted,mean);
		return DistributionFit.maximumLikelihood(exact(sorted),new DistributionFit.ParametricFamily(){
			public int parameterCount(){return 3;} public double lowerBound(int p){return p==1?1e-9:Double.NEGATIVE_INFINITY;}
			public double upperBound(int p){return Double.POSITIVE_INFINITY;}
			public GenericDistribution distribution(double[] p){return new GEV(p[0],p[1],p[2]);}
		},new double[]{mean,Math.max(sd,1e-3),0.0});
	}
	/** Probability-weighted-moment initialization returned through the common result contract. */
	public static DistributionFit.Result fitGevPwm(double[] blockMaxima) {
		double[] x=checked(blockMaxima);int n=x.length;double b0=mean(x),b1=0.0,b2=0.0;
		for(int i=0;i<n;i++){b1+=i*x[i]/(double)Math.max(1,n-1);b2+=i*(i-1.0)*x[i]/Math.max(1.0,(n-1.0)*(n-2.0));}
		b1/=n;b2/=n;double l1=b0,l2=2*b1-b0,l3=6*b2-6*b1+b0;
		double t3=l2==0.0?0.0:l3/l2;double shape=7.8590*(2.0/(3.0+t3)-Math.log(2.0)/Math.log(3.0))
				+2.9554*Math.pow(2.0/(3.0+t3)-Math.log(2.0)/Math.log(3.0),2.0);
		double scale=Math.max(1e-9,l2*shape/((1.0-Math.pow(2.0,-shape))*jdistlib.math.MathFunctions.gammafn(1.0+shape)));
		double location=l1-scale*(1.0-jdistlib.math.MathFunctions.gammafn(1.0+shape))/shape;
		return fitFromInitial(x,new double[]{location,scale,shape},true);
	}
	public static DistributionFit.Result fitGpd(double[] observations,double threshold) {
		double[] checked=checked(observations);int count=0;for(double value:checked)if(value>threshold)count++;
		if(count<3)throw new IllegalArgumentException("at least three strict threshold exceedances required");
		double[] excess=new double[count];int at=0;for(double value:checked)if(value>threshold)excess[at++]=value-threshold;
		double mean=mean(excess);
		return DistributionFit.maximumLikelihood(exact(excess),new DistributionFit.ParametricFamily(){
			public int parameterCount(){return 2;} public double lowerBound(int p){return p==0?1e-9:-1.0;}
			public double upperBound(int p){return p==0?Double.POSITIVE_INFINITY:2.0;}
			public GenericDistribution distribution(double[] p){return new GeneralizedPareto(0.0,p[0],p[1]);}
		},new double[]{Math.max(mean,1e-6),0.0});
	}
	public static NumericalEstimate hill(double[] observations,int upperOrderStatistics) {
		double[] x=checked(observations);int n=x.length;if(upperOrderStatistics<1||upperOrderStatistics>=n||x[n-upperOrderStatistics-1]<=0.0)
			throw new IllegalArgumentException("Hill requires 1 <= k < n and positive upper observations");
		double threshold=Math.log(x[n-upperOrderStatistics-1]),sum=0.0;
		for(int i=n-upperOrderStatistics;i<n;i++)sum+=Math.log(x[i])-threshold;
		double estimate=sum/upperOrderStatistics;
		return new NumericalEstimate(estimate,estimate/Math.sqrt(upperOrderStatistics),true,upperOrderStatistics,"Hill","asymptotic standard error");
	}
	public static NumericalEstimate pickands(double[] observations,int k) {
		double[] x=checked(observations);int n=x.length;if(k<1||4*k>=n)throw new IllegalArgumentException("Pickands requires 4k < n");
		double numerator=x[n-k]-x[n-2*k],denominator=x[n-2*k]-x[n-4*k];
		double value=Math.log(numerator/denominator)/Math.log(2.0);
		return new NumericalEstimate(value,Math.abs(value)/Math.sqrt(k),Double.isFinite(value),4*k,"Pickands","asymptotic standard error");
	}
	public static NumericalEstimate returnLevel(double location,double scale,double shape,double periods) {
		if(!(scale>0.0)||!(periods>1.0))throw new IllegalArgumentException("positive scale and periods > 1 required");
		double value=GEV.quantile(1.0-1.0/periods,location,scale,shape,true);
		return new NumericalEstimate(value,0.0,Double.isFinite(value),1,"GEV-return-level","");
	}
	public static ThresholdDiagnostics thresholds(double[] observations,double[] thresholds) {
		double[] x=checked(observations);if(thresholds==null||thresholds.length==0)throw new IllegalArgumentException("thresholds required");
		double[] meanExcess=new double[thresholds.length],hill=new double[thresholds.length];int[] counts=new int[thresholds.length];
		for(int j=0;j<thresholds.length;j++){double sum=0.0;for(double value:x)if(value>thresholds[j]){sum+=value-thresholds[j];counts[j]++;}
			meanExcess[j]=counts[j]==0?Double.NaN:sum/counts[j];hill[j]=counts[j]>1&&thresholds[j]>0.0?hill(x,Math.min(counts[j]-1,x.length-1)).getValue():Double.NaN;}
		return new ThresholdDiagnostics(thresholds,meanExcess,hill,counts);
	}
	public static NumericalEstimate bootstrapReturnLevel(double[] maxima,double periods,int replicates,long seed){
		double[] x=checked(maxima);if(replicates<20)throw new IllegalArgumentException("at least 20 replicates required");
		MersenneTwister random=new MersenneTwister(seed);double[] levels=new double[replicates];int success=0;
		for(int b=0;b<replicates;b++){double[] sample=new double[x.length];for(int i=0;i<x.length;i++)sample[i]=x[(int)(random.nextDouble()*x.length)];
			DistributionFit.Result fit=fitGev(sample);if(fit.isConverged()){double[] p=fit.getParameters();levels[success++]=returnLevel(p[0],p[1],p[2],periods).getValue();}}
		if(success<2)return new NumericalEstimate(Double.NaN,Double.POSITIVE_INFINITY,false,replicates,"bootstrap-return-level","too few successful fits");
		double center=0.0;for(int i=0;i<success;i++)center+=levels[i];center/=success;double variance=0.0;for(int i=0;i<success;i++)variance+=(levels[i]-center)*(levels[i]-center);
		return new NumericalEstimate(center,Math.sqrt(variance/(success-1)),true,replicates,"bootstrap-return-level",success<replicates?"some bootstrap fits failed":"");
	}
	public static final class ThresholdDiagnostics{
		private final double[] thresholds,meanExcess,tailIndex;private final int[] exceedances;
		private ThresholdDiagnostics(double[] t,double[] m,double[] h,int[] c){thresholds=t.clone();meanExcess=m;tailIndex=h;exceedances=c;}
		public double[] getThresholds(){return thresholds.clone();}public double[] getMeanExcess(){return meanExcess.clone();}
		public double[] getTailIndex(){return tailIndex.clone();}public int[] getExceedances(){return exceedances.clone();}
	}
	private static DistributionFit.Result fitFromInitial(double[] x,double[] initial,boolean gev){
		if(!gev)throw new AssertionError();return DistributionFit.maximumLikelihood(exact(x),new DistributionFit.ParametricFamily(){
			public int parameterCount(){return 3;}public double lowerBound(int p){return p==1?1e-9:Double.NEGATIVE_INFINITY;}
			public double upperBound(int p){return Double.POSITIVE_INFINITY;}public GenericDistribution distribution(double[] p){return new GEV(p[0],p[1],p[2]);}},initial);
	}
	private static DistributionFit.Observation[] exact(double[] x){DistributionFit.Observation[] o=new DistributionFit.Observation[x.length];for(int i=0;i<x.length;i++)o[i]=DistributionFit.Observation.exact(x[i]);return o;}
	private static double[] checked(double[] values){if(values==null||values.length<3)throw new IllegalArgumentException("at least three observations required");double[] x=values.clone();for(double v:x)if(!Double.isFinite(v))throw new IllegalArgumentException("finite observations required");Arrays.sort(x);return x;}
	private static double mean(double[] x){double s=0.0;for(double v:x)s+=v;return s/x.length;}
	private static double standardDeviation(double[] x,double mean){double s=0.0;for(double v:x)s+=(v-mean)*(v-mean);return Math.sqrt(s/Math.max(1,x.length-1));}
}
