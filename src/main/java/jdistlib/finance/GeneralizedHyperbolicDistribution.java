/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.Bessel;
import jdistlib.math.Complex;

/**
 * Generalized-hyperbolic law in the canonical (lambda, alpha, beta, delta, mu)
 * parameterization, with alpha &gt; |beta| and delta &gt; 0.
 */
public class GeneralizedHyperbolicDistribution extends GenericDistribution
		implements SupportedDistribution, TransformDistribution {
	private final double lambda;
	private final double alpha;
	private final double beta;
	private final double delta;
	private final double mu;
	private final double gamma;
	private final double logConstant;

	public GeneralizedHyperbolicDistribution(double lambda, double alpha, double beta,
			double delta, double mu) {
		if (!Double.isFinite(lambda) || !(alpha > Math.abs(beta)) || !(delta > 0.0)
				|| !Double.isFinite(alpha) || !Double.isFinite(beta)
				|| !Double.isFinite(delta) || !Double.isFinite(mu))
			throw new IllegalArgumentException("GH requires finite parameters, alpha > |beta|, and delta > 0");
		this.lambda = lambda;
		this.alpha = alpha;
		this.beta = beta;
		this.delta = delta;
		this.mu = mu;
		this.gamma = Math.sqrt(alpha * alpha - beta * beta);
		this.logConstant = lambda * (Math.log(gamma) - Math.log(delta))
				- 0.5 * Math.log(2.0 * Math.PI) - logBesselK(delta * gamma, lambda);
	}

	public static GeneralizedHyperbolicDistribution normalInverseGaussian(double alpha,
			double beta, double delta, double mu) {
		return new GeneralizedHyperbolicDistribution(-0.5, alpha, beta, delta, mu);
	}

	public double getLambda() { return lambda; }
	public double getAlpha() { return alpha; }
	public double getBeta() { return beta; }
	public double getDelta() { return delta; }
	public double getMu() { return mu; }

	@Override public double density(double x, boolean log) {
		double z = x - mu;
		double radius = Math.hypot(delta, z);
		double answer = logConstant + logBesselK(alpha * radius, lambda - 0.5)
				+ (lambda - 0.5) * (Math.log(radius) - Math.log(alpha)) + beta * z;
		return log ? answer : Math.exp(answer);
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		NumericalEstimate estimate = integrateCdf(x);
		double value = lowerTail ? estimate.getValue() : 1.0 - estimate.getValue();
		value = Math.max(0.0, Math.min(1.0, value));
		return logP ? Math.log(value) : value;
	}

	@Override public double quantile(double probability, boolean lowerTail, boolean logP) {
		if (logP) probability = Math.exp(probability);
		if (!lowerTail) probability = 1.0 - probability;
		if (probability < 0.0 || probability > 1.0 || Double.isNaN(probability)) return Double.NaN;
		if (probability == 0.0) return Double.NEGATIVE_INFINITY;
		if (probability == 1.0) return Double.POSITIVE_INFINITY;
		double span = Math.max(delta, 1.0 / gamma);
		double low = mu - span;
		double high = mu + span;
		while (cumulative(low, true, false) > probability) { span *= 2.0; low = mu - span; }
		while (cumulative(high, true, false) < probability) { span *= 2.0; high = mu + span; }
		for (int i = 0; i < 70; i++) {
			double middle = low + (high - low) / 2.0;
			if (cumulative(middle, true, false) < probability) low = middle; else high = middle;
		}
		return low + (high - low) / 2.0;
	}

	@Override public double random() {
		return quantile(random.nextDouble(), true, false);
	}

	/** CDF plus a deterministic quadrature-difference error estimate. */
	public NumericalEstimate cumulativeResult(double x) { return integrateCdf(x); }

	@Override public Complex logCharacteristic(double frequency) {
		if (frequency == 0.0) return Complex.ZERO;
		/* Complex-order Bessel evaluation is not available; this stable numerical
		 * fallback preserves the public transform contract and phase. */
		return numericalLogCharacteristic(frequency);
	}

	@Override public Complex logMomentGenerating(double argument) {
		if (!momentGeneratingDomain().contains(argument))
			return new Complex(Double.POSITIVE_INFINITY, 0.0);
		double shiftedGamma = Math.sqrt(alpha * alpha - (beta + argument) * (beta + argument));
		double logMgf = mu * argument + lambda * (Math.log(gamma) - Math.log(shiftedGamma))
				+ logBesselK(delta * shiftedGamma, lambda) - logBesselK(delta * gamma, lambda);
		return new Complex(logMgf, 0.0);
	}

	@Override public TransformDomain momentGeneratingDomain() {
		return new TransformDomain(-alpha - beta, false, alpha - beta, false);
	}

	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }

	private NumericalEstimate integrateCdf(double x) {
		if (x == Double.NEGATIVE_INFINITY) return new NumericalEstimate(0.0, 0.0, true, 0, "GH-CDF", "");
		if (x == Double.POSITIVE_INFINITY) return new NumericalEstimate(1.0, 0.0, true, 0, "GH-CDF", "");
		double coarse = transformedIntegral(x, 2048);
		double fine = transformedIntegral(x, 4096);
		return new NumericalEstimate(Math.max(0.0, Math.min(1.0, fine)), Math.abs(fine - coarse),
				Double.isFinite(fine), 6144, "infinite-interval-midpoint", "");
	}

	private double transformedIntegral(double x, int panels) {
		/* Maps (-infinity,x] to (0,1); avoids dependence on mutable integration state. */
		double sum = 0.0;
		for (int i = 0; i < panels; i++) {
			double u = (i + 0.5) / panels;
			double complement = 1.0 - u;
			double y = x - u / complement;
			double jacobian = 1.0 / (complement * complement);
			if (Double.isFinite(y) && Double.isFinite(jacobian)) sum += density(y, false) * jacobian;
		}
		return sum / panels;
	}

	private Complex numericalLogCharacteristic(double frequency) {
		double real = 0.0;
		double imaginary = 0.0;
		int panels = 8192;
		for (int i = 0; i < panels; i++) {
			double u = (i + 0.5) / panels;
			double angle = Math.PI * (u - 0.5);
			double x = mu + Math.tan(angle);
			double jacobian = Math.PI / (Math.cos(angle) * Math.cos(angle));
			double weight = density(x, false) * jacobian;
			real += weight * Math.cos(frequency * x);
			imaginary += weight * Math.sin(frequency * x);
		}
		return new Complex(real / panels, imaginary / panels).log();
	}

	public static double logBesselK(double x, double order) {
		double scaled = Bessel.k(x, order, true);
		return Math.log(scaled) - x;
	}
}
