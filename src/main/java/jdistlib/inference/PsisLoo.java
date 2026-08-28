/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Pareto-smoothed importance-sampling leave-one-out cross-validation. */
public final class PsisLoo {
	private PsisLoo() {}
	/** Optional exact or refitted LOO calculation used when importance sampling is unreliable. */
	public interface ExactLooFallback { double elpd(int observation, String observationName); }

	public static Result compute(PointwiseLogLikelihoodDraws draws) { return compute(draws, null); }
	public static Result compute(PointwiseLogLikelihoodDraws draws, ExactLooFallback fallback) {
		if (draws == null || draws.draws() < 5 || draws.observations() < 1)
			throw new IllegalArgumentException("PSIS-LOO requires at least five draws and one observation");
		double[][] values = draws.values();
		int observations = draws.observations(), samples = draws.draws();
		double[] pointwise = new double[observations], lpd = new double[observations];
		double[] paretoK = new double[observations], effectiveSampleSize = new double[observations];
		boolean[] fallbackUsed = new boolean[observations];
		List<Integer> unreliable = new ArrayList<Integer>();
		for (int observation = 0; observation < observations; observation++) {
			double[] logLikelihood = new double[samples], logRatios = new double[samples];
			for (int draw = 0; draw < samples; draw++) {
				logLikelihood[draw] = values[draw][observation];
				logRatios[draw] = -logLikelihood[draw];
			}
			lpd[observation] = PredictiveMath.logMeanExp(logLikelihood);
			ParetoSmoothedImportanceSampling.Result smoothing = ParetoSmoothedImportanceSampling.smooth(logRatios);
			paretoK[observation] = smoothing.paretoK();
			double[] logWeights = smoothing.logWeights(), weighted = new double[samples];
			double squaredWeightSum = 0.0;
			for (int draw = 0; draw < samples; draw++) {
				weighted[draw] = logLikelihood[draw] + logWeights[draw];
				double weight = Math.exp(logWeights[draw]);
				squaredWeightSum += weight * weight;
			}
			effectiveSampleSize[observation] = 1.0 / squaredWeightSum;
			pointwise[observation] = PredictiveMath.logSumExp(weighted);
			if (!smoothing.reliable()) {
				unreliable.add(Integer.valueOf(observation));
				if (fallback != null) {
					double replacement = fallback.elpd(observation, draws.metadata().name(observation));
					if (!Double.isFinite(replacement)) throw new IllegalArgumentException("fallback ELPD must be finite");
					pointwise[observation] = replacement;
					fallbackUsed[observation] = true;
				}
			}
		}
		double elpd = 0.0, pLoo = 0.0;
		for (int observation = 0; observation < observations; observation++) {
			elpd += pointwise[observation];
			pLoo += lpd[observation] - pointwise[observation];
		}
		double standardError = Math.sqrt(observations * PredictiveMath.sampleVariance(pointwise));
		return new Result(draws.metadata(), pointwise, paretoK, effectiveSampleSize, fallbackUsed,
				unreliable, elpd, pLoo, standardError);
	}

	public static final class Result {
		private final ObservationMetadata metadata;
		private final double[] pointwiseElpd, paretoK, effectiveSampleSize;
		private final boolean[] fallbackUsed;
		private final List<Integer> unreliable;
		private final double elpd, pLoo, standardError;
		private Result(ObservationMetadata metadata, double[] pointwiseElpd, double[] paretoK,
				double[] effectiveSampleSize, boolean[] fallbackUsed, List<Integer> unreliable,
				double elpd, double pLoo, double standardError) {
			this.metadata = metadata; this.pointwiseElpd = pointwiseElpd; this.paretoK = paretoK;
			this.effectiveSampleSize = effectiveSampleSize; this.fallbackUsed = fallbackUsed;
			this.unreliable = Collections.unmodifiableList(new ArrayList<Integer>(unreliable));
			this.elpd = elpd; this.pLoo = pLoo; this.standardError = standardError;
		}
		public ObservationMetadata metadata() { return metadata; }
		public int observationCount() { return pointwiseElpd.length; }
		public double elpd() { return elpd; }
		public double looic() { return -2.0 * elpd; }
		public double effectiveNumberOfParameters() { return pLoo; }
		public double standardError() { return standardError; }
		public double[] pointwiseElpd() { return pointwiseElpd.clone(); }
		public double[] paretoK() { return paretoK.clone(); }
		public double[] effectiveSampleSize() { return effectiveSampleSize.clone(); }
		public boolean[] fallbackUsed() { return fallbackUsed.clone(); }
		public List<Integer> unreliableObservations() { return unreliable; }
		/** True when all k diagnostics are acceptable or every unacceptable value was replaced. */
		public boolean reliable() {
			for (int index : unreliable) if (!fallbackUsed[index]) return false;
			return true;
		}
	}
}
