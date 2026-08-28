/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Exact in-memory RJ restart point including ragged state, schedule, adaptation, and RNG. */
public final class ReversibleJumpCheckpoint {
	private final ReversibleJumpState state; private final double logJoint; private final int completedIterations;
	private final RandomEngine random; private final String[] moveNames; private final double[] moveWeights;
	private final Map<String, double[]> adaptationState; private final boolean warmupComplete;
	public ReversibleJumpCheckpoint(ReversibleJumpState state, double logJoint, int completedIterations,
			RandomEngine random, String[] moveNames, double[] moveWeights,
			Map<String, double[]> adaptationState, boolean warmupComplete) {
		if (state == null || Double.isNaN(logJoint) || completedIterations < 0 || random == null
				|| moveNames == null || moveWeights == null || moveNames.length != moveWeights.length || adaptationState == null)
			throw new IllegalArgumentException("complete RJ checkpoint state required");
		this.state = state; this.logJoint = logJoint; this.completedIterations = completedIterations; this.random = random.clone();
		this.moveNames = moveNames.clone(); this.moveWeights = moveWeights.clone(); this.warmupComplete = warmupComplete;
		Map<String, double[]> copy = new LinkedHashMap<String, double[]>();
		for (int i = 0; i < this.moveNames.length; i++) if (this.moveNames[i] == null || !(this.moveWeights[i] > 0.0)
				|| !Double.isFinite(this.moveWeights[i])) throw new IllegalArgumentException("valid move schedule required");
		for (Map.Entry<String, double[]> entry : adaptationState.entrySet()) {
			if (entry.getKey() == null || entry.getValue() == null) throw new IllegalArgumentException("valid adaptation entries required");
			double[] values = entry.getValue().clone();
			for (double value : values) if (!Double.isFinite(value)) throw new IllegalArgumentException("finite adaptation entries required");
			copy.put(entry.getKey(), values);
		}
		this.adaptationState = Collections.unmodifiableMap(copy);
	}
	public ReversibleJumpState state() { return state; }
	public double logJoint() { return logJoint; }
	public int completedIterations() { return completedIterations; }
	public RandomEngine random() { return random.clone(); }
	public String[] moveNames() { return moveNames.clone(); }
	public double[] moveWeights() { return moveWeights.clone(); }
	public Map<String, double[]> adaptationState() {
		Map<String, double[]> copy = new LinkedHashMap<String, double[]>();
		for (Map.Entry<String, double[]> entry : adaptationState.entrySet()) copy.put(entry.getKey(), entry.getValue().clone());
		return Collections.unmodifiableMap(copy);
	}
	public boolean warmupComplete() { return warmupComplete; }
}
