/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Warmup-only candidate-specific Gaussian birth adaptation with checkpointable moments. */
public final class AdaptiveGaussianRjBirthProposal implements RjBirthProposal {
	private static final class Moments { long count; double mean, products; Moments(double mean) { this.mean = mean; } }
	private final double initialMean, initialScale, minimumScale; private final Moments[] moments; private boolean frozen;
	public AdaptiveGaussianRjBirthProposal(int candidates, double initialMean, double initialScale, double minimumScale) {
		if (candidates < 1 || !Double.isFinite(initialMean) || !(initialScale > 0.0) || !(minimumScale > 0.0)
				|| !Double.isFinite(initialScale) || !Double.isFinite(minimumScale)) throw new IllegalArgumentException("candidate count and finite positive Gaussian scales required");
		this.initialMean = initialMean; this.initialScale = initialScale; this.minimumScale = minimumScale; moments = new Moments[candidates];
		for (int i = 0; i < candidates; i++) moments[i] = new Moments(initialMean);
	}
	@Override public double sample(int candidate, ReversibleJumpState state, RandomEngine random) {
		Moments value = moments(candidate); return value.mean + scale(value) * random.nextGaussian();
	}
	@Override public double logDensity(double value, int candidate, ReversibleJumpState state) {
		Moments momentsValue = moments(candidate); double scale = scale(momentsValue), standardized = (value - momentsValue.mean) / scale;
		return -0.5 * standardized * standardized - Math.log(scale) - 0.5 * Math.log(2.0 * Math.PI);
	}
	@Override public void warmupUpdate(int candidate, double value, ReversibleJumpState state, boolean accepted) {
		if (frozen || !accepted || !Double.isFinite(value)) return; Moments current = moments(candidate); current.count++;
		double difference = value - current.mean; current.mean += difference / current.count; current.products += difference * (value - current.mean);
	}
	@Override public Map<String, double[]> adaptationState() {
		Map<String, double[]> result = new LinkedHashMap<String, double[]>();
		for (int candidate = 0; candidate < moments.length; candidate++) { Moments value = moments[candidate]; result.put("candidate/" + candidate,
				new double[] {value.count, value.mean, value.products, frozen ? 1.0 : 0.0}); }
		return result;
	}
	@Override public void restoreAdaptation(Map<String, double[]> state) {
		if (state == null) throw new IllegalArgumentException("adaptation state required"); resetAdaptation();
		for (int candidate = 0; candidate < moments.length; candidate++) {
			double[] values = state.get("candidate/" + candidate); if (values == null) continue;
			if (values.length != 4 || values[0] < 0.0 || values[0] != Math.rint(values[0]) || values[0] > Long.MAX_VALUE
					|| !Double.isFinite(values[1]) || values[2] < 0.0 || !Double.isFinite(values[2])
					|| (values[3] != 0.0 && values[3] != 1.0)) throw new IllegalArgumentException("invalid birth adaptation state");
			moments[candidate].count = (long) values[0]; moments[candidate].mean = values[1]; moments[candidate].products = values[2]; frozen |= values[3] != 0.0;
		}
	}
	@Override public void freezeAdaptation() { frozen = true; }
	@Override public void resetAdaptation() {
		frozen = false; for (Moments value : moments) { value.count = 0L; value.mean = initialMean; value.products = 0.0; }
	}
	public double mean(int candidate) { return moments(candidate).mean; }
	public double standardDeviation(int candidate) { return scale(moments(candidate)); }
	private Moments moments(int candidate) { if (candidate < 0 || candidate >= moments.length) throw new IllegalArgumentException("candidate out of range"); return moments[candidate]; }
	private double scale(Moments value) { return value.count < 2L ? initialScale : Math.max(minimumScale, Math.sqrt(value.products / (value.count - 1.0))); }
}
