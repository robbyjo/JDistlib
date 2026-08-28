/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Per-model isotropic random walk with warmup-only Robbins-Monro scale adaptation. */
public final class AdaptiveRjRandomWalkKernel implements ReversibleJumpWithinModelKernel {
	private static final class ScaleState { double logScale; int updates; ScaleState(double value) { logScale = value; } }
	private final String name; private final double initialScale, targetAcceptance;
	private final Map<Long, ScaleState> scales = new LinkedHashMap<Long, ScaleState>(); private boolean frozen;
	public AdaptiveRjRandomWalkKernel(String name, double initialScale, double targetAcceptance) {
		if (name == null || name.trim().isEmpty() || !(initialScale > 0.0) || !Double.isFinite(initialScale)
				|| !(targetAcceptance > 0.0 && targetAcceptance < 1.0)) throw new IllegalArgumentException("name, scale, and target acceptance required");
		this.name = name; this.initialScale = initialScale; this.targetAcceptance = targetAcceptance;
	}
	@Override public String name() { return name; }
	@Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) { return state.dimension() > 0; }
	@Override public ReversibleJumpWithinModelTransition update(ReversibleJumpState state, double currentLogJoint,
			ReversibleJumpTarget target, RandomEngine random, boolean warmup) {
		if (state == null || target == null || random == null || !Double.isFinite(currentLogJoint))
			throw new IllegalArgumentException("valid current RJ state required");
		ScaleState adaptation = scaleState(state.modelId()); double scale = Math.exp(adaptation.logScale);
		double[] proposal = state.parameters(); for (int i = 0; i < proposal.length; i++) proposal[i] += scale * random.nextGaussian();
		ReversibleJumpState proposed = new ReversibleJumpState(state.modelId(), proposal);
		double proposedLogJoint = target.logJoint(proposed);
		double probability = Double.isFinite(proposedLogJoint) ? Math.min(1.0, Math.exp(proposedLogJoint - currentLogJoint)) : 0.0;
		boolean accepted = random.nextDouble() < probability;
		if (warmup && !frozen) {
			adaptation.updates++; double rate = 1.0 / Math.sqrt(adaptation.updates + 10.0);
			adaptation.logScale += rate * ((accepted ? 1.0 : 0.0) - targetAcceptance);
			adaptation.logScale = Math.max(Math.log(1e-8), Math.min(Math.log(1e4), adaptation.logScale));
		}
		return accepted ? new ReversibleJumpWithinModelTransition(proposed, proposedLogJoint, true, probability)
				: new ReversibleJumpWithinModelTransition(state, currentLogJoint, false, probability);
	}
	public double scale(long modelId) { return Math.exp(scaleState(modelId).logScale); }
	@Override public Map<String, double[]> adaptationState() {
		Map<String, double[]> result = new LinkedHashMap<String, double[]>();
		for (Map.Entry<Long, ScaleState> entry : scales.entrySet())
			result.put(name + "/model/" + entry.getKey(), new double[] {entry.getValue().logScale, entry.getValue().updates, frozen ? 1.0 : 0.0});
		return result;
	}
	@Override public void restoreAdaptation(Map<String, double[]> state) {
		if (state == null) throw new IllegalArgumentException("adaptation state required"); scales.clear(); frozen = false;
		String prefix = name + "/model/";
		for (Map.Entry<String, double[]> entry : state.entrySet()) if (entry.getKey().startsWith(prefix)) {
			double[] values = entry.getValue(); if (values == null || values.length != 3) throw new IllegalArgumentException("invalid random-walk adaptation state");
			long model = Long.parseLong(entry.getKey().substring(prefix.length()));
			if (model < 0L || !Double.isFinite(values[0]) || values[1] < 0.0 || values[1] != Math.rint(values[1])
					|| values[1] > Integer.MAX_VALUE || (values[2] != 0.0 && values[2] != 1.0))
				throw new IllegalArgumentException("invalid random-walk adaptation state");
			ScaleState restored = new ScaleState(values[0]); restored.updates = (int) values[1];
			scales.put(Long.valueOf(model), restored); frozen |= values[2] != 0.0;
		}
	}
	@Override public void freezeAdaptation() { frozen = true; }
	@Override public void resetAdaptation() { scales.clear(); frozen = false; }
	private ScaleState scaleState(long modelId) {
		ScaleState result = scales.get(Long.valueOf(modelId));
		if (result == null) { result = new ScaleState(Math.log(initialScale)); scales.put(Long.valueOf(modelId), result); }
		return result;
	}
}
