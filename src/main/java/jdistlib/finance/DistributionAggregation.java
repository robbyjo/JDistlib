/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;
import jdistlib.AtomAwareDistribution;
import jdistlib.SupportedDistribution;
import jdistlib.math.Complex;
import org.jtransforms.fft.DoubleFFT_1D;

/** Reproducible aggregation, product/ratio, compound-sum, and scenario helpers. */
public final class DistributionAggregation {
	private DistributionAggregation() {}

	public static DistributionApproximation convolution(GenericDistribution first,
			GenericDistribution second, int draws, long seed) {
		return weightedSum(new GenericDistribution[] {first, second}, new double[] {1.0, 1.0}, draws, seed);
	}

	/** Exact convolution of finite integer-valued atom-aware distributions. */
	public static DistributionApproximation exactDiscreteConvolution(GenericDistribution... distributions){
		if(distributions==null||distributions.length==0)throw new IllegalArgumentException("at least one distribution required");double[] result={1.0};int origin=0,evaluations=0;
		for(GenericDistribution distribution:distributions){if(!(distribution instanceof AtomAwareDistribution)||!(distribution instanceof SupportedDistribution))throw new IllegalArgumentException("finite atom-aware support required");
			SupportedDistribution support=(SupportedDistribution)distribution;double lo=support.getLowerBound(),hi=support.getUpperBound();if(!Double.isFinite(lo)||!Double.isFinite(hi)||lo!=Math.rint(lo)||hi!=Math.rint(hi)||hi-lo>100000)throw new IllegalArgumentException("finite integer support required");
			int lower=(int)lo,upper=(int)hi;double[] mass=new double[upper-lower+1];for(int i=0;i<mass.length;i++){mass[i]=((AtomAwareDistribution)distribution).atomProbability(lower+i);evaluations++;}
			result=directConvolution(result,mass);origin+=lower;}
		return new DistributionApproximation(new FiniteGridDistribution(origin,1.0,result),new NumericalEstimate(1.0,0.0,true,evaluations,"exact-finite-discrete-convolution",""),0L);}

	/** Linear (non-cyclic) FFT convolution of two equal-step probability grids. */
	public static DistributionApproximation fftConvolution(double[] first,double firstOrigin,double[] second,double secondOrigin,double step){
		validateMass(first);validateMass(second);if(!(step>0.0)||!Double.isFinite(firstOrigin+secondOrigin))throw new IllegalArgumentException("positive step and finite origins required");
		int outputLength=first.length+second.length-1,n=1;while(n<outputLength)n<<=1;double[] a=new double[2*n],b=new double[2*n];System.arraycopy(first,0,a,0,first.length);System.arraycopy(second,0,b,0,second.length);
		DoubleFFT_1D fft=new DoubleFFT_1D(n);fft.realForwardFull(a);fft.realForwardFull(b);for(int i=0;i<2*n;i+=2){double ar=a[i],ai=a[i+1],br=b[i],bi=b[i+1];a[i]=ar*br-ai*bi;a[i+1]=ar*bi+ai*br;}fft.complexInverse(a,true);
		double[] mass=new double[outputLength];double negative=0.0;for(int i=0;i<outputLength;i++){if(a[2*i]<0.0)negative-=a[2*i];mass[i]=Math.max(0.0,a[2*i]);}
		return new DistributionApproximation(new FiniteGridDistribution(firstOrigin+secondOrigin,step,mass),new NumericalEstimate(1.0,negative,true,2*n,"zero-padded-FFT-convolution",negative==0.0?"":"negative roundoff mass clipped"),0L);}

	/** Panjer (a,b,0) recursion; countAtZero is P(N=0), including severity mass at zero. */
	public static DistributionApproximation panjerCompound(double a,double b,double countAtZero,double[] severity,int maximumLoss){
		if(!Double.isFinite(a+b)||!(countAtZero>0.0&&countAtZero<=1.0)||severity==null||severity.length==0||maximumLoss<0)throw new IllegalArgumentException("invalid Panjer request");validateMass(severity);
		double f0=severity[0],denominator=1.0-a*f0;if(!(denominator>0.0))throw new IllegalArgumentException("Panjer denominator must be positive");double[] aggregate=new double[maximumLoss+1];
		double countProbability=countAtZero,power=1.0;aggregate[0]=countAtZero;int countTerms=0;for(int n=1;n<100000;n++){countProbability*=a+b/n;if(countProbability<-1e-14||!Double.isFinite(countProbability))throw new IllegalArgumentException("parameters do not define a Panjer count law");countProbability=Math.max(0.0,countProbability);power*=f0;double term=countProbability*power;aggregate[0]+=term;countTerms=n;if(countProbability==0.0||(n>100&&term<1e-15))break;}
		for(int k=1;k<=maximumLoss;k++){double sum=0.0;for(int j=1;j<=k&&j<severity.length;j++)sum+=(a+b*j/(double)k)*severity[j]*aggregate[k-j];aggregate[k]=Math.max(0.0,sum/denominator);}
		double retained=0.0;for(double value:aggregate)retained+=value;double omitted=Math.max(0.0,1.0-retained);
		return new DistributionApproximation(new FiniteGridDistribution(0.0,1.0,aggregate),new NumericalEstimate(retained,omitted,omitted<1e-8,maximumLoss+countTerms,"Panjer-(a,b,0)-recursion",omitted<1e-8?"":"aggregate tail truncated"),0L);}

	/** Fourier-cosine density discretization on a caller-declared finite interval. */
	public static DistributionApproximation cosInversion(TransformDistribution distribution,double lower,double upper,int bins,int terms){
		if(distribution==null||!Double.isFinite(lower+upper)||!(lower<upper)||bins<16||terms<8)throw new IllegalArgumentException("invalid COS inversion request");double width=upper-lower,step=width/bins;double[] coefficients=new double[terms];
		for(int k=0;k<terms;k++){double frequency=k*Math.PI/width;Complex phi=distribution.logCharacteristic(frequency).exp();double angle=-frequency*lower;coefficients[k]=2.0/width*(phi.real()*Math.cos(angle)-phi.imaginary()*Math.sin(angle));}
		double[] mass=new double[bins];double total=0.0;for(int i=0;i<bins;i++){double x=lower+(i+0.5)*step,density=0.5*coefficients[0];for(int k=1;k<terms;k++)density+=coefficients[k]*Math.cos(k*Math.PI*(x-lower)/width);mass[i]=Math.max(0.0,density*step);total+=mass[i];}
		double error=Math.abs(1.0-total);return new DistributionApproximation(new FiniteGridDistribution(lower+0.5*step,step,mass),new NumericalEstimate(total,error,error<1e-4,terms+bins,"COS-transform-inversion",error<1e-4?"":"interval/series truncation mass is material"),0L);}

	public static DistributionApproximation weightedSum(GenericDistribution[] distributions,
			double[] weights, int draws, long seed) {
		validate(distributions, weights, draws);
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		for (int i = 0; i < draws; i++) {
			double value = 0.0;
			for (int j = 0; j < distributions.length; j++)
				value += weights[j] * inverseSample(distributions[j], random);
			sample[i] = value;
		}
		return approximation(sample, seed, "reproducible-Monte-Carlo-weighted-sum");
	}

	public static DistributionApproximation compoundSum(GenericDistribution count,
			GenericDistribution severity, int draws, int maximumCount, long seed) {
		if (count == null || severity == null || draws < 100 || maximumCount < 1)
			throw new IllegalArgumentException("laws are required, draws >= 100, maximumCount >= 1");
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		int truncated = 0;
		for (int i = 0; i < draws; i++) {
			int events = (int) Math.max(0.0, Math.rint(inverseSample(count, random)));
			if (events > maximumCount) { events = maximumCount; truncated++; }
			double total = 0.0;
			for (int event = 0; event < events; event++) total += inverseSample(severity, random);
			sample[i] = total;
		}
		double error = Math.max(1.0 / Math.sqrt(draws), truncated / (double) draws);
		return new DistributionApproximation(new EmpiricalDistribution(sample),
				new NumericalEstimate(1.0, error, truncated == 0, draws,
						"reproducible-Monte-Carlo-compound-sum",
						truncated == 0 ? "" : truncated + " counts truncated at maximumCount"), seed);
	}

	public static DistributionApproximation product(GenericDistribution first,
			GenericDistribution second, int draws, long seed) {
		return binary(first, second, draws, seed, false);
	}

	public static DistributionApproximation ratio(GenericDistribution numerator,
			GenericDistribution denominator, int draws, long seed) {
		return binary(numerator, denominator, draws, seed, true);
	}

	/** Applies caller-supplied scenarios to a base law with explicit seed provenance. */
	public static DistributionApproximation scenario(GenericDistribution base,
			ScenarioTransformation transformation, int draws, long seed) {
		if (base == null || transformation == null || draws < 100)
			throw new IllegalArgumentException("base/transformation required and draws >= 100");
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		for (int i = 0; i < draws; i++) sample[i] = transformation.apply(inverseSample(base, random));
		return approximation(sample, seed, "reproducible-Monte-Carlo-scenario");
	}

	public interface ScenarioTransformation { double apply(double value); }

	private static DistributionApproximation binary(GenericDistribution first,
			GenericDistribution second, int draws, long seed, boolean ratio) {
		if (first == null || second == null || draws < 100)
			throw new IllegalArgumentException("laws are required and draws >= 100");
		RandomEngine random = new MersenneTwister(seed);
		double[] sample = new double[draws];
		int rejected = 0;
		for (int i = 0; i < draws; i++) {
			double a = inverseSample(first, random);
			double b = inverseSample(second, random);
			if (ratio && b == 0.0) { i--; rejected++; if (rejected > draws) throw new IllegalArgumentException("denominator has excessive mass at zero"); }
			else sample[i] = ratio ? a / b : a * b;
		}
		return approximation(sample, seed, ratio ? "reproducible-Monte-Carlo-ratio" : "reproducible-Monte-Carlo-product");
	}

	private static DistributionApproximation approximation(double[] sample, long seed, String strategy) {
		return new DistributionApproximation(new EmpiricalDistribution(sample),
				new NumericalEstimate(1.0, 1.0 / Math.sqrt(sample.length), true,
						sample.length, strategy, "empirical Monte Carlo approximation"), seed);
	}

	private static double inverseSample(GenericDistribution distribution, RandomEngine random) {
		double probability;
		do probability = random.nextDouble(); while (probability == 0.0);
		return distribution.quantile(probability, true, false);
	}

	private static void validate(GenericDistribution[] distributions, double[] weights, int draws) {
		if (distributions == null || weights == null || distributions.length == 0
				|| distributions.length != weights.length || draws < 100)
			throw new IllegalArgumentException("matching nonempty laws/weights and draws >= 100 are required");
		for (int i = 0; i < weights.length; i++) if (distributions[i] == null || !Double.isFinite(weights[i]))
			throw new IllegalArgumentException("laws and finite weights are required");
	}
	private static double[] directConvolution(double[] first,double[] second){double[] result=new double[first.length+second.length-1];for(int i=0;i<first.length;i++)for(int j=0;j<second.length;j++)result[i+j]+=first[i]*second[j];return result;}
	private static void validateMass(double[] mass){if(mass==null||mass.length==0)throw new IllegalArgumentException("nonempty mass grid required");double total=0.0;for(double value:mass){if(!(value>=0.0)||!Double.isFinite(value))throw new IllegalArgumentException("finite nonnegative mass required");total+=value;}if(Math.abs(total-1.0)>1e-10)throw new IllegalArgumentException("probability mass must sum to one");}
}
