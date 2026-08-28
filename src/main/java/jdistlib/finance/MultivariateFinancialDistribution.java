/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.Gamma;
import jdistlib.Normal;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Multivariate GH/NIG/VG, symmetric stable, or normal-tempered-stable construction. */
public final class MultivariateFinancialDistribution {
	private enum Kind { GH, VG, STABLE, NTS }
	private final Kind kind;private final double[] location,skew;private final double[][] covariance,cholesky;
	private final double first,second,third;
	private final GeneralizedInverseGaussianDistribution gigMixer;
	private final PositiveTemperedStableDistribution temperedMixer;
	private MultivariateFinancialDistribution(Kind kind,double[] location,double[] skew,double[][] covariance,double first,double second,double third){
		if(location==null||skew==null||location.length==0||location.length!=skew.length)throw new IllegalArgumentException("matching location/skew vectors required");
		this.kind=kind;this.location=location.clone();this.skew=skew.clone();this.covariance=copy(covariance,location.length);this.cholesky=cholesky(this.covariance);this.first=first;this.second=second;this.third=third;
		this.gigMixer=kind==Kind.GH?new GeneralizedInverseGaussianDistribution(first,second,third):null;this.temperedMixer=kind==Kind.NTS?new PositiveTemperedStableDistribution(first,second,third):null;
	}
	/** Multivariate GH normal-variance mixture with W~GIG(lambda,chi,psi). */
	public static MultivariateFinancialDistribution generalizedHyperbolic(double lambda,double chi,double psi,double[] location,double[] skew,double[][] covariance){
		if(!(chi>0.0)||!(psi>0.0)||!Double.isFinite(lambda))throw new IllegalArgumentException("invalid GH mixing parameters");return new MultivariateFinancialDistribution(Kind.GH,location,skew,covariance,lambda,chi,psi);}
	public static MultivariateFinancialDistribution normalInverseGaussian(double delta,double gamma,double[] location,double[] skew,double[][] covariance){
		if(!(delta>0.0)||!(gamma>0.0))throw new IllegalArgumentException("positive delta/gamma required");return generalizedHyperbolic(-0.5,delta*delta,gamma*gamma,location,skew,covariance);}
	/** Multivariate VG with shared Gamma(shape,1) mixer. */
	public static MultivariateFinancialDistribution varianceGamma(double shape,double[] location,double[] skew,double[][] covariance){
		if(!(shape>0.0))throw new IllegalArgumentException("positive shape required");return new MultivariateFinancialDistribution(Kind.VG,location,skew,covariance,shape,0.0,0.0);}
	/** Symmetric elliptical alpha-stable law with CF exp(-(t' covariance t)^(alpha/2)). */
	public static MultivariateFinancialDistribution stable(double alpha,double[] location,double[][] covariance){
		if(!(alpha>0.0&&alpha<=2.0))throw new IllegalArgumentException("alpha must be in (0,2]");return new MultivariateFinancialDistribution(Kind.STABLE,location,new double[location.length],covariance,alpha,0.0,0.0);}
	/** Multivariate normal-tempered-stable normal variance mixture. */
	public static MultivariateFinancialDistribution normalTemperedStable(double alpha,double tempering,double intensity,double[] location,double[] skew,double[][] covariance){
		if(!(alpha>0.0&&alpha<1.0)||!(tempering>0.0)||!(intensity>0.0))throw new IllegalArgumentException("invalid NTS parameters");return new MultivariateFinancialDistribution(Kind.NTS,location,skew,covariance,alpha,tempering,intensity);}
	public int dimension(){return location.length;}public double[] getLocation(){return location.clone();}public double[] getSkew(){return skew.clone();}public double[][] getCovariance(){return copy(covariance,location.length);}
	public double[] random(long seed){return random(new MersenneTwister(seed));}
	public double[] random(RandomEngine random){if(random==null)throw new IllegalArgumentException("random engine required");double mixing;
		if(kind==Kind.GH)mixing=gigMixer.quantile(open(random),true,false);
		else if(kind==Kind.VG)mixing=Gamma.random(first,1.0,random);
		else if(kind==Kind.STABLE)mixing=first==2.0?1.0:positiveStable(first/2.0,random);
		else mixing=temperedMixer.quantile(open(random),true,false);
		double normalScale=kind==Kind.STABLE?Math.sqrt(2.0*mixing):Math.sqrt(mixing);double[] z=new double[dimension()],result=new double[dimension()];
		for(int i=0;i<z.length;i++)z[i]=Normal.random_standard(random);for(int i=0;i<result.length;i++){double gaussian=0.0;for(int j=0;j<=i;j++)gaussian+=cholesky[i][j]*z[j];result[i]=location[i]+mixing*skew[i]+normalScale*gaussian;}return result;}
	/** Exact scalar law for a linear combination of the vector. */
	public GenericDistribution linearCombination(double[] weights){if(weights==null||weights.length!=dimension())throw new IllegalArgumentException("weight dimension mismatch");double mu=dot(weights,location),beta=dot(weights,skew),q=quadratic(weights,covariance);
		if(!(q>0.0))throw new IllegalArgumentException("projection variance must be positive");
		if(kind==Kind.VG)return new VarianceGammaDistribution(first,beta,Math.sqrt(q),mu);
		if(kind==Kind.STABLE)return new StableDistribution(first,0.0,Math.sqrt(q),mu);
		if(kind==Kind.NTS)return new NormalTemperedStableDistribution(first,second,third,beta,Math.sqrt(q),mu);
		double delta=Math.sqrt(second*q),gamma=Math.sqrt(third/q),projectedBeta=beta/q,alpha=Math.sqrt(gamma*gamma+projectedBeta*projectedBeta);
		return new GeneralizedHyperbolicDistribution(first,alpha,projectedBeta,delta,mu);}
	private static double positiveStable(double alpha,RandomEngine random){double u=Math.PI*Math.max(Double.MIN_VALUE,random.nextDouble()),e=-Math.log(Math.max(Double.MIN_VALUE,random.nextDouble()));
		return Math.sin(alpha*u)/Math.pow(Math.sin(u),1.0/alpha)*Math.pow(Math.sin((1.0-alpha)*u)/e,(1.0-alpha)/alpha);}
	private static double open(RandomEngine random){double probability;do probability=random.nextDouble();while(probability==0.0);return probability;}
	private static double dot(double[] a,double[] b){double sum=0.0;for(int i=0;i<a.length;i++)sum+=a[i]*b[i];return sum;}
	private static double quadratic(double[] w,double[][] matrix){double sum=0.0;for(int i=0;i<w.length;i++)for(int j=0;j<w.length;j++)sum+=w[i]*matrix[i][j]*w[j];return sum;}
	private static double[][] copy(double[][] matrix,int n){if(matrix==null||matrix.length!=n)throw new IllegalArgumentException("square covariance required");double[][] result=new double[n][n];for(int i=0;i<n;i++){if(matrix[i]==null||matrix[i].length!=n)throw new IllegalArgumentException("square covariance required");result[i]=matrix[i].clone();}return result;}
	private static double[][] cholesky(double[][] matrix){int n=matrix.length;double[][] lower=new double[n][n];for(int i=0;i<n;i++)for(int j=0;j<=i;j++){double sum=matrix[i][j];for(int k=0;k<j;k++)sum-=lower[i][k]*lower[j][k];if(i==j){if(!(sum>0.0))throw new IllegalArgumentException("covariance must be positive definite");lower[i][j]=Math.sqrt(sum);}else lower[i][j]=sum/lower[j][j];}return lower;}
}
