/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Widely applicable information criterion from pointwise log-likelihood draws. */
public final class Waic {
	private Waic() {}
	public static Result compute(PointwiseLogLikelihoodDraws draws) {
		if (draws == null || draws.draws() < 2 || draws.observations() < 1)
			throw new IllegalArgumentException("WAIC requires at least two draws and one observation");
		double[][] values = draws.values();
		int observations = draws.observations(), samples = draws.draws();
		double[] pointwise = new double[observations], variances = new double[observations];
		List<Integer> highVariance = new ArrayList<Integer>();
		double elpd = 0.0, effectiveParameters = 0.0;
		for (int observation = 0; observation < observations; observation++) {
			double[] column = new double[samples];
			for (int draw = 0; draw < samples; draw++) {
				column[draw] = values[draw][observation];
				if (!Double.isFinite(column[draw])) throw new IllegalArgumentException("WAIC log likelihoods must be finite");
			}
			variances[observation] = PredictiveMath.sampleVariance(column);
			pointwise[observation] = PredictiveMath.logMeanExp(column) - variances[observation];
			elpd += pointwise[observation]; effectiveParameters += variances[observation];
			if (variances[observation] > 0.4) highVariance.add(Integer.valueOf(observation));
		}
		return new Result(draws.metadata(), pointwise, variances, highVariance, elpd,
				effectiveParameters, Math.sqrt(observations * PredictiveMath.sampleVariance(pointwise)));
	}
	public static final class Result {
		private final ObservationMetadata metadata;
		private final double[] pointwiseElpd, logLikelihoodVariance;
		private final List<Integer> highVariance;
		private final double elpd, effectiveParameters, standardError;
		private Result(ObservationMetadata metadata, double[] pointwiseElpd, double[] variance,
				List<Integer> highVariance, double elpd, double effectiveParameters, double standardError) {
			this.metadata = metadata; this.pointwiseElpd = pointwiseElpd; this.logLikelihoodVariance = variance;
			this.highVariance = Collections.unmodifiableList(new ArrayList<Integer>(highVariance));
			this.elpd = elpd; this.effectiveParameters = effectiveParameters; this.standardError = standardError;
		}
		public ObservationMetadata metadata() { return metadata; }
		public double elpd() { return elpd; }
		public double waic() { return -2.0 * elpd; }
		public double effectiveNumberOfParameters() { return effectiveParameters; }
		public double standardError() { return standardError; }
		public double[] pointwiseElpd() { return pointwiseElpd.clone(); }
		public double[] logLikelihoodVariance() { return logLikelihoodVariance.clone(); }
		public List<Integer> highVarianceObservations() { return highVariance; }
		public boolean reliable() { return highVariance.isEmpty(); }
	}
}
