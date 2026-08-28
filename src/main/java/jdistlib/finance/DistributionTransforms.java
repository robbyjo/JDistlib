/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.NumericalContinuousDistribution;
import jdistlib.SupportedDistribution;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.Complex;
import jdistlib.math.IntegrationOptions;

/** Numerical transform, cumulant, Fourier-inversion, and Esscher-tilt helpers. */
public final class DistributionTransforms {
	private static final int PANELS = 8192;
	private DistributionTransforms() {}

	public static Complex logCharacteristic(GenericDistribution distribution, double frequency) {
		if (distribution instanceof TransformDistribution)
			return ((TransformDistribution) distribution).logCharacteristic(frequency);
		Complex value = quantileTransform(distribution, frequency, false);
		return value.log();
	}

	public static Complex logMomentGenerating(GenericDistribution distribution, double argument) {
		if (distribution instanceof TransformDistribution) {
			TransformDistribution transformed = (TransformDistribution) distribution;
			if (!transformed.momentGeneratingDomain().contains(argument))
				return new Complex(Double.POSITIVE_INFINITY, 0.0);
			return transformed.logMomentGenerating(argument);
		}
		return quantileTransform(distribution, argument, true).log();
	}

	/** Numerical cumulant obtained by centered finite differences of log M(t). */
	public static NumericalEstimate cumulant(GenericDistribution distribution, int order) {
		if (distribution == null || order < 1 || order > 4)
			throw new IllegalArgumentException("orders 1 through 4 are supported");
		double h = 1e-3;
		double m2 = logMomentGenerating(distribution, -2.0 * h).real();
		double m1 = logMomentGenerating(distribution, -h).real();
		double m0 = logMomentGenerating(distribution, 0.0).real();
		double p1 = logMomentGenerating(distribution, h).real();
		double p2 = logMomentGenerating(distribution, 2.0 * h).real();
		double value;
		if (order == 1) value = (p1 - m1) / (2.0 * h);
		else if (order == 2) value = (p1 - 2.0 * m0 + m1) / (h * h);
		else if (order == 3) value = (p2 - 2.0 * p1 + 2.0 * m1 - m2) / (2.0 * h * h * h);
		else value = (p2 - 4.0 * p1 + 6.0 * m0 - 4.0 * m1 + m2) / Math.pow(h, 4.0);
		return new NumericalEstimate(value, Math.abs(value) * 1e-5, Double.isFinite(value),
				5 * PANELS, "finite-difference-cumulant", Double.isFinite(value) ? "" : "moment does not exist");
	}

	/** Gil-Pelaez inversion of a characteristic function. */
	public static NumericalEstimate cumulative(TransformDistribution distribution, double x) {
		if (distribution == null || Double.isNaN(x)) throw new IllegalArgumentException("invalid input");
		double coarse = invertCdf(distribution, x, PANELS / 2, 80.0);
		double fine = invertCdf(distribution, x, PANELS, 120.0);
		double value = Math.max(0.0, Math.min(1.0, fine));
		return new NumericalEstimate(value, Math.abs(fine - coarse), Double.isFinite(fine),
				PANELS + PANELS / 2, "gil-pelaez", Double.isFinite(fine) ? "" : "Fourier inversion failed");
	}

	/** Constructs a normalized exponentially tilted continuous distribution. */
	public static TiltResult esscherTilt(GenericDistribution distribution, double theta) {
		if (distribution == null) throw new IllegalArgumentException("distribution is required");
		Complex logMgf = logMomentGenerating(distribution, theta);
		if (!Double.isFinite(logMgf.real()) || Math.abs(logMgf.imaginary()) > 1e-9)
			throw new IllegalArgumentException("moment-generating function does not exist at tilt");
		double lower = distribution instanceof SupportedDistribution
				? ((SupportedDistribution) distribution).getLowerBound() : Double.NEGATIVE_INFINITY;
		double upper = distribution instanceof SupportedDistribution
				? ((SupportedDistribution) distribution).getUpperBound() : Double.POSITIVE_INFINITY;
		NumericalContinuousDistribution tilted = NumericalContinuousDistribution.builder()
				.logKernel(x -> theta * x + distribution.density(x, true) - logMgf.real())
				.support(lower, upper)
				.integrationOptions(IntegrationOptions.builder().tolerances(0.0, 1e-9)
						.subdivisions(400).maxEvaluations(500000).build())
				.withoutAnalysis().build();
		double error = Math.abs(tilted.getNormalizationConstant() - 1.0);
		return new TiltResult(tilted, theta, logMgf.real(), new NumericalEstimate(1.0,
				error, error < 1e-7, tilted.getNormalizationResult().neval,
				"esscher-normalization", error < 1e-7 ? "" : "normalization tolerance exceeded"));
	}

	private static Complex quantileTransform(GenericDistribution distribution, double argument,
			boolean mgf) {
		if (distribution == null || !Double.isFinite(argument))
			throw new IllegalArgumentException("distribution and finite argument are required");
		double real = 0.0;
		double imaginary = 0.0;
		for (int i = 0; i < PANELS; i++) {
			double x = distribution.quantile((i + 0.5) / PANELS, true, false);
			if (mgf) real += Math.exp(argument * x);
			else {
				real += Math.cos(argument * x);
				imaginary += Math.sin(argument * x);
			}
		}
		return new Complex(real / PANELS, imaginary / PANELS);
	}

	private static double invertCdf(TransformDistribution distribution, double x,
			int panels, double maximum) {
		double width = maximum / panels;
		double sum = 0.0;
		for (int i = 0; i < panels; i++) {
			double t = (i + 0.5) * width;
			Complex phi = distribution.logCharacteristic(t).exp();
			double imaginary = phi.imaginary() * Math.cos(t * x) - phi.real() * Math.sin(t * x);
			sum += imaginary / t;
		}
		return 0.5 - width * sum / Math.PI;
	}

	/** Result retaining the tilted law and its normalization diagnostics. */
	public static final class TiltResult {
		private final NumericalContinuousDistribution distribution;
		private final double theta;
		private final double logNormalizer;
		private final NumericalEstimate normalization;
		private TiltResult(NumericalContinuousDistribution distribution, double theta,
				double logNormalizer, NumericalEstimate normalization) {
			this.distribution = distribution;
			this.theta = theta;
			this.logNormalizer = logNormalizer;
			this.normalization = normalization;
		}
		public NumericalContinuousDistribution getDistribution() { return distribution; }
		public double getTheta() { return theta; }
		public double getLogNormalizer() { return logNormalizer; }
		public NumericalEstimate getNormalization() { return normalization; }
	}
}
