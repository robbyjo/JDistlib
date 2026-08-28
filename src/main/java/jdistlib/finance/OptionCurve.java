/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import java.util.Arrays;
import java.util.Comparator;

/** Arbitrage-repaired European option curve and its implied risk-neutral law. */
public final class OptionCurve {
	private final double forward,discount,maturity;
	private final double[] strikes,calls,originalCalls;
	private final Diagnostics diagnostics;
	private final OptionImpliedDistribution distribution;
	private OptionCurve(double forward,double discount,double maturity,double[] strikes,double[] calls,
			double[] originalCalls,Diagnostics diagnostics,OptionImpliedDistribution distribution){
		this.forward=forward;this.discount=discount;this.maturity=maturity;this.strikes=strikes;this.calls=calls;
		this.originalCalls=originalCalls;this.diagnostics=diagnostics;this.distribution=distribution;
	}
	public static OptionCurve build(double forward,double discount,double maturity,OptionObservation... observations){
		if(!(forward>0.0)||!(discount>0.0)||!(maturity>0.0)||observations==null||observations.length<2)
			throw new IllegalArgumentException("positive forward/discount/maturity and at least two quotes required");
		OptionObservation[] sorted=observations.clone();Arrays.sort(sorted,Comparator.comparingDouble(OptionObservation::getStrike));
		int n=sorted.length+1;double[] k=new double[n],c=new double[n],original=new double[n],weight=new double[n];
		k[0]=0.0;c[0]=forward;original[0]=forward;weight[0]=1e12;
		for(int i=0;i<sorted.length;i++){OptionObservation quote=sorted[i];if(i>0&&quote.getStrike()==sorted[i-1].getStrike())throw new IllegalArgumentException("duplicate strikes are not supported");
			k[i+1]=quote.getStrike();double call=quote.isCall()?quote.getMid():quote.getMid()/discount+forward-quote.getStrike();
			if(quote.isCall())call/=discount;c[i+1]=call;original[i+1]=call;weight[i+1]=quote.getWeight();}
		for(int iteration=0;iteration<30;iteration++){for(int i=0;i<n;i++)c[i]=Math.max(Math.max(forward-k[i],0.0),Math.min(forward,c[i]));
			projectSlopes(k,c,weight);c[0]=forward;}
		double lastSlope=(c[n-1]-c[n-2])/(k[n-1]-k[n-2]);double terminal=k[n-1];
		if(c[n-1]>1e-12&&lastSlope<-1e-12)terminal+=c[n-1]/(-lastSlope);else terminal+=Math.max(1.0,forward*0.1);
		double[] fullK=Arrays.copyOf(k,n+1),fullC=Arrays.copyOf(c,n+1);fullK[n]=terminal;fullC[n]=0.0;
		projectSlopes(fullK,fullC,uniformWeights(n+1));fullC[0]=forward;fullC[n]=0.0;
		int repaired=0;double maximumResidual=0.0,weightedSquared=0.0,totalWeight=0.0;
		for(int i=1;i<n;i++){double residual=discount*(c[i]-original[i]);maximumResidual=Math.max(maximumResidual,Math.abs(residual));
			weightedSquared+=weight[i]*residual*residual;totalWeight+=weight[i];if(Math.abs(residual)>1e-10)repaired++;}
		Diagnostics diagnostics=new Diagnostics(repaired,maximumResidual,Math.sqrt(weightedSquared/Math.max(totalWeight,1e-300)),true,true,true);
		double[] fullOriginal=Arrays.copyOf(original,n+1);fullOriginal[n]=0.0;
		return new OptionCurve(forward,discount,maturity,fullK,fullC,fullOriginal,diagnostics,new OptionImpliedDistribution(fullK,fullC));
	}
	private static void projectSlopes(double[] k,double[] c,double[] weights){int m=k.length-1;double[] slope=new double[m],w=new double[m];int[] start=new int[m],end=new int[m];int blocks=0;
		for(int i=0;i<m;i++){double dx=k[i+1]-k[i];if(!(dx>0.0))throw new IllegalArgumentException("strikes must be strictly increasing");
			slope[blocks]=Math.max(-1.0,Math.min(0.0,(c[i+1]-c[i])/dx));w[blocks]=dx*Math.min(1e12,weights[Math.min(i+1,weights.length-1)]);start[blocks]=i;end[blocks]=i;blocks++;
			while(blocks>1&&slope[blocks-2]>slope[blocks-1]){double total=w[blocks-2]+w[blocks-1];slope[blocks-2]=(slope[blocks-2]*w[blocks-2]+slope[blocks-1]*w[blocks-1])/total;
				w[blocks-2]=total;end[blocks-2]=end[blocks-1];blocks--;}}
		double[] fitted=new double[m];for(int b=0;b<blocks;b++)for(int i=start[b];i<=end[b];i++)fitted[i]=slope[b];
		for(int i=0;i<m;i++)c[i+1]=c[i]+fitted[i]*(k[i+1]-k[i]);}
	private static double[] uniformWeights(int n){double[] w=new double[n];Arrays.fill(w,1.0);return w;}
	public double getForward(){return forward;}public double getDiscount(){return discount;}public double getMaturity(){return maturity;}
	public double[] getStrikes(){return strikes.clone();}public double[] getUndiscountedCalls(){return calls.clone();}
	public double[] getOriginalUndiscountedCalls(){return originalCalls.clone();}public Diagnostics getDiagnostics(){return diagnostics;}
	public OptionImpliedDistribution getDistribution(){return distribution;}
	public double terminalProbability(double threshold,Tail tail){return distribution.cumulative(threshold,tail==Tail.LOWER,false);}
	public double strikeIntervalProbability(double lower,double upper){return distribution.cumulative(upper,true,false)-distribution.cumulative(lower,true,false);}
	public static final class Diagnostics{
		private final int repaired;private final double maximumResidual,weightedRmse;private final boolean normalized,monotone,convex;
		private Diagnostics(int repaired,double max,double rmse,boolean normalized,boolean monotone,boolean convex){this.repaired=repaired;maximumResidual=max;weightedRmse=rmse;this.normalized=normalized;this.monotone=monotone;this.convex=convex;}
		public int getRepairedObservations(){return repaired;}public double getMaximumPriceResidual(){return maximumResidual;}
		public double getWeightedRmse(){return weightedRmse;}public boolean isNormalized(){return normalized;}
		public boolean isMonotone(){return monotone;}public boolean isConvex(){return convex;}
	}
}
