/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.Gamma;
import jdistlib.Normal;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.Complex;
import jdistlib.math.MathFunctions;

/** Variance-gamma law X=mu+theta*G+sigma*sqrt(G)*Z, G~Gamma(shape,1). */
public final class VarianceGammaDistribution extends GenericDistribution
		implements SupportedDistribution, TransformDistribution {
	private final double shape;
	private final double theta;
	private final double sigma;
	private final double mu;

	public VarianceGammaDistribution(double shape, double theta, double sigma, double mu) {
		if (!(shape > 0.0) || !(sigma > 0.0) || !Double.isFinite(shape)
				|| !Double.isFinite(theta) || !Double.isFinite(sigma) || !Double.isFinite(mu))
			throw new IllegalArgumentException("VG requires positive shape/sigma and finite parameters");
		this.shape = shape;
		this.theta = theta;
		this.sigma = sigma;
		this.mu = mu;
	}

	public double getShape() { return shape; }
	public double getTheta() { return theta; }
	public double getSigma() { return sigma; }
	public double getMu() { return mu; }

	@Override public double density(double x, boolean log) {
		double z = x - mu;
		double nu = shape - 0.5;
		double root = Math.sqrt(2.0 * sigma * sigma + theta * theta);
		if (z == 0.0 && nu <= 0.0) return Double.POSITIVE_INFINITY;
		if (z == 0.0) {
			double answer = Math.log(2.0) - Math.log(sigma) - 0.5 * Math.log(2.0 * Math.PI)
					- MathFunctions.lgammafn(shape) + MathFunctions.lgammafn(nu)
					+ (nu - 1.0) * Math.log(2.0) - 2.0 * nu * Math.log(root)
					+ 2.0 * nu * Math.log(sigma);
			return log ? answer : Math.exp(answer);
		}
		double absolute = Math.abs(z);
		double answer = Math.log(2.0) + theta * z / (sigma * sigma)
				- Math.log(sigma) - 0.5 * Math.log(2.0 * Math.PI)
				- MathFunctions.lgammafn(shape) + nu * (Math.log(absolute) - Math.log(root))
				+ GeneralizedHyperbolicDistribution.logBesselK(absolute * root / (sigma * sigma), nu);
		return log ? answer : Math.exp(answer);
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		double value = DistributionTransforms.cumulative(this, x).getValue();
		if (!lowerTail) value = 1.0 - value;
		return logP ? Math.log(value) : value;
	}

	@Override public double quantile(double probability, boolean lowerTail, boolean logP) {
		if (logP) probability = Math.exp(probability);
		if (!lowerTail) probability = 1.0 - probability;
		if (probability < 0.0 || probability > 1.0 || Double.isNaN(probability)) return Double.NaN;
		if (probability == 0.0) return Double.NEGATIVE_INFINITY;
		if (probability == 1.0) return Double.POSITIVE_INFINITY;
		double standardDeviation = Math.sqrt(shape * (sigma * sigma + theta * theta));
		double center = mu + shape * theta;
		double low = center - 8.0 * standardDeviation;
		double high = center + 8.0 * standardDeviation;
		while (cumulative(low, true, false) > probability) low -= 8.0 * standardDeviation;
		while (cumulative(high, true, false) < probability) high += 8.0 * standardDeviation;
		for (int i = 0; i < 70; i++) {
			double middle = low + (high - low) / 2.0;
			if (cumulative(middle, true, false) < probability) low = middle; else high = middle;
		}
		return low + (high - low) / 2.0;
	}

	@Override public double random() {
		double mixing = Gamma.random(shape, 1.0, random);
		return mu + theta * mixing + sigma * Math.sqrt(mixing) * Normal.random_standard(random);
	}

	@Override public Complex logCharacteristic(double frequency) {
		Complex base = new Complex(1.0 + 0.5 * sigma * sigma * frequency * frequency,
				-theta * frequency);
		Complex log = base.log();
		return new Complex(-shape * log.real(), mu * frequency - shape * log.imaginary());
	}

	@Override public Complex logMomentGenerating(double argument) {
		if (!momentGeneratingDomain().contains(argument)) return new Complex(Double.POSITIVE_INFINITY, 0.0);
		return new Complex(mu * argument - shape * Math.log(1.0 - theta * argument
				- 0.5 * sigma * sigma * argument * argument), 0.0);
	}

	@Override public TransformDomain momentGeneratingDomain() {
		double discriminant = Math.sqrt(theta * theta + 2.0 * sigma * sigma);
		return new TransformDomain((-theta - discriminant) / (sigma * sigma), false,
				(-theta + discriminant) / (sigma * sigma), false);
	}

	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }
}
