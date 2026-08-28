/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.Cauchy;
import jdistlib.Normal;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.Complex;

/** Univariate alpha-stable law in Nolan's S1 parameterization. */
public final class StableDistribution extends GenericDistribution
		implements SupportedDistribution, TransformDistribution {
	private final double alpha;
	private final double beta;
	private final double scale;
	private final double location;

	public StableDistribution(double alpha, double beta, double scale, double location) {
		if (!(alpha > 0.0 && alpha <= 2.0) || !(beta >= -1.0 && beta <= 1.0)
				|| !(scale > 0.0) || !Double.isFinite(alpha) || !Double.isFinite(beta)
				|| !Double.isFinite(scale) || !Double.isFinite(location))
			throw new IllegalArgumentException("stable requires alpha in (0,2], beta in [-1,1], scale > 0");
		this.alpha = alpha;
		this.beta = beta;
		this.scale = scale;
		this.location = location;
	}

	/** Converts Nolan S0 location to this class's canonical S1 location. */
	public static StableDistribution fromS0(double alpha, double beta, double scale,
			double locationS0) {
		double shift = alpha == 1.0 ? beta * 2.0 / Math.PI * scale * Math.log(scale)
				: beta * scale * Math.tan(Math.PI * alpha / 2.0);
		return new StableDistribution(alpha, beta, scale, locationS0 - shift);
	}

	public double getAlpha() { return alpha; }
	public double getBeta() { return beta; }
	public double getScale() { return scale; }
	public double getLocation() { return location; }
	public boolean momentExists(double order) { return order >= 0.0 && order < alpha; }

	@Override public double density(double x, boolean log) {
		double value;
		if (alpha == 2.0) value = Normal.density(x, location, Math.sqrt(2.0) * scale, false);
		else if (alpha == 1.0 && beta == 0.0) value = Cauchy.density(x, location, scale, false);
		else value = fourierDensity(x, 120.0, 16384);
		return log ? Math.log(value) : value;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		double value;
		if (alpha == 2.0) value = Normal.cumulative(x, location, Math.sqrt(2.0) * scale, true, false);
		else if (alpha == 1.0 && beta == 0.0) value = Cauchy.cumulative(x, location, scale, true, false);
		else value = DistributionTransforms.cumulative(this, x).getValue();
		if (!lowerTail) value = 1.0 - value;
		return logP ? Math.log(value) : value;
	}

	@Override public double quantile(double probability, boolean lowerTail, boolean logP) {
		if (logP) probability = Math.exp(probability);
		if (!lowerTail) probability = 1.0 - probability;
		if (alpha == 2.0) return Normal.quantile(probability, location, Math.sqrt(2.0) * scale, true, false);
		if (alpha == 1.0 && beta == 0.0) return Cauchy.quantile(probability, location, scale, true, false);
		if (probability <= 0.0) return probability == 0.0 ? Double.NEGATIVE_INFINITY : Double.NaN;
		if (probability >= 1.0) return probability == 1.0 ? Double.POSITIVE_INFINITY : Double.NaN;
		double span = scale;
		double low = location - span;
		double high = location + span;
		while (cumulative(low, true, false) > probability) { span *= 2.0; low = location - span; }
		while (cumulative(high, true, false) < probability) { span *= 2.0; high = location + span; }
		for (int i = 0; i < 70; i++) {
			double middle = low + (high - low) / 2.0;
			if (cumulative(middle, true, false) < probability) low = middle; else high = middle;
		}
		return low + (high - low) / 2.0;
	}

	@Override public double random() {
		double v = Math.PI * (random.nextDouble() - 0.5);
		double w = -Math.log(Math.max(random.nextDouble(), Double.MIN_VALUE));
		double sample;
		if (alpha == 1.0) {
			double halfPi = Math.PI / 2.0;
			sample = 2.0 / Math.PI * ((halfPi + beta * v) * Math.tan(v)
					- beta * Math.log(halfPi * w * Math.cos(v) / (halfPi + beta * v)));
		} else {
			double zeta = beta * Math.tan(Math.PI * alpha / 2.0);
			double b = Math.atan(zeta) / alpha;
			double s = Math.pow(1.0 + zeta * zeta, 1.0 / (2.0 * alpha));
			sample = s * Math.sin(alpha * (v + b)) / Math.pow(Math.cos(v), 1.0 / alpha)
					* Math.pow(Math.cos(v - alpha * (v + b)) / w, (1.0 - alpha) / alpha);
		}
		double scaleShift = alpha == 1.0
				? beta * 2.0 / Math.PI * scale * Math.log(scale) : 0.0;
		return location + scaleShift + scale * sample;
	}

	@Override public Complex logCharacteristic(double frequency) {
		if (frequency == 0.0) return Complex.ZERO;
		double absolute = Math.abs(scale * frequency);
		double real = -Math.pow(absolute, alpha);
		double imaginary = location * frequency;
		if (alpha == 1.0) imaginary -= Math.pow(absolute, alpha) * beta
				* Math.signum(frequency) * 2.0 / Math.PI * Math.log(Math.abs(frequency));
		else imaginary += Math.pow(absolute, alpha) * beta * Math.signum(frequency)
				* Math.tan(Math.PI * alpha / 2.0);
		return new Complex(real, imaginary);
	}

	@Override public Complex logMomentGenerating(double argument) {
		if (argument == 0.0) return Complex.ZERO;
		if (alpha == 2.0) return new Complex(location * argument + scale * scale * argument * argument, 0.0);
		return new Complex(Double.POSITIVE_INFINITY, 0.0);
	}

	@Override public TransformDomain momentGeneratingDomain() {
		return alpha == 2.0 ? TransformDomain.allReal() : new TransformDomain(0.0, true, 0.0, true);
	}

	@Override public double getLowerBound() { return Double.NEGATIVE_INFINITY; }
	@Override public double getUpperBound() { return Double.POSITIVE_INFINITY; }

	private double fourierDensity(double x, double maximum, int panels) {
		double width = maximum / panels;
		double sum = 0.0;
		for (int i = 0; i < panels; i++) {
			double t = (i + 0.5) * width;
			Complex phi = logCharacteristic(t).exp();
			sum += phi.real() * Math.cos(t * x) + phi.imaginary() * Math.sin(t * x);
		}
		return Math.max(0.0, width * sum / Math.PI);
	}
}
