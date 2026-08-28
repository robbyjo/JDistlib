/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** General Java-only RJMCMC acceptance engine with warmup-frozen move and within-model adaptation. */
public final class ReversibleJumpSampler {
	private final ReversibleJumpMove[] moves; private final double[] initialWeights;
	private final ReversibleJumpWithinModelKernel[] within;
	public ReversibleJumpSampler(ReversibleJumpMove[] moves, double[] moveWeights,
			ReversibleJumpWithinModelKernel... withinModelKernels) {
		if (moves == null || moveWeights == null || moves.length == 0 || moves.length != moveWeights.length || withinModelKernels == null)
			throw new IllegalArgumentException("moves, weights, and within-model schedule required");
		this.moves = moves.clone(); this.initialWeights = moveWeights.clone(); this.within = withinModelKernels.clone();
		for (int i = 0; i < this.moves.length; i++) {
			if (this.moves[i] == null || this.moves[i].name() == null || this.moves[i].name().trim().isEmpty()
					|| !(this.initialWeights[i] > 0.0) || !Double.isFinite(this.initialWeights[i])) throw new IllegalArgumentException("valid named moves and weights required");
			for (int j = 0; j < i; j++) if (this.moves[j].name().equals(this.moves[i].name())) throw new IllegalArgumentException("move names must be unique");
		}
		for (int i = 0; i < this.within.length; i++) {
			if (this.within[i] == null || this.within[i].name() == null || this.within[i].name().trim().isEmpty()) throw new IllegalArgumentException("valid within-model kernels required");
			for (int j = 0; j < i; j++) if (this.within[j].name().equals(this.within[i].name())) throw new IllegalArgumentException("within-model kernel names must be unique");
		}
	}
	public ReversibleJumpResult sample(ReversibleJumpTarget target, ReversibleJumpState initialState,
			ReversibleJumpSamplingOptions options, RandomEngine random) {
		for (ReversibleJumpMove move : moves) move.resetAdaptation();
		for (ReversibleJumpWithinModelKernel kernel : within) kernel.resetAdaptation();
		return run(target, initialState, Double.NaN, options, random, initialWeights.clone(), 0, false);
	}
	public ReversibleJumpResult resume(ReversibleJumpTarget target, ReversibleJumpCheckpoint checkpoint,
			ReversibleJumpSamplingOptions options) {
		if (checkpoint == null || !checkpoint.warmupComplete() || options == null || options.warmupIterations() != 0)
			throw new IllegalArgumentException("resume requires a completed-warmup checkpoint and zero new warmup");
		String[] names = checkpoint.moveNames(); if (names.length != moves.length) throw new IllegalArgumentException("checkpoint move schedule mismatch");
		for (int i = 0; i < names.length; i++) if (!names[i].equals(moves[i].name())) throw new IllegalArgumentException("checkpoint move schedule mismatch");
		Map<String, double[]> adaptation = checkpoint.adaptationState();
		for (ReversibleJumpMove move : moves) {
			move.resetAdaptation(); move.restoreAdaptation(componentState(adaptation, "move/" + move.name() + "/")); move.freezeAdaptation();
		}
		for (ReversibleJumpWithinModelKernel kernel : within) { kernel.resetAdaptation(); kernel.restoreAdaptation(adaptation); kernel.freezeAdaptation(); }
		return run(target, checkpoint.state(), checkpoint.logJoint(), options, checkpoint.random(), checkpoint.moveWeights(), checkpoint.completedIterations(), true);
	}
	private ReversibleJumpResult run(ReversibleJumpTarget target, ReversibleJumpState initialState, double suppliedLogJoint,
			ReversibleJumpSamplingOptions options, RandomEngine random, double[] weights, int priorCompleted, boolean resumed) {
		if (target == null || initialState == null || options == null || random == null) throw new IllegalArgumentException("target, state, options, and random required");
		validateState(target, initialState); double logJoint = Double.isNaN(suppliedLogJoint) ? target.logJoint(initialState) : suppliedLogJoint;
		List<ReversibleJumpState> draws = new ArrayList<ReversibleJumpState>(); List<Double> values = new ArrayList<Double>();
		List<ReversibleJumpIterationStats> stats = new ArrayList<ReversibleJumpIterationStats>(); List<String> warnings = new ArrayList<String>();
		long[] attempts = new long[moves.length], accepts = new long[moves.length], invalid = new long[moves.length];
		if (!Double.isFinite(logJoint)) return result(draws, values, stats, initialState, logJoint, priorCompleted, random, weights,
				attempts, accepts, invalid, warnings, ReversibleJumpResult.Status.INVALID_INITIAL_STATE, false);
		if (options.warmupIterations() == 0) {
			for (ReversibleJumpMove move : moves) move.freezeAdaptation();
			for (ReversibleJumpWithinModelKernel kernel : within) kernel.freezeAdaptation();
		}
		ReversibleJumpState state = initialState; int total = options.warmupIterations() + options.sampleIterations() * options.thinning(), completed = 0, retained = 0;
		for (int iteration = 0; iteration < total; iteration++) {
			boolean warmup = iteration < options.warmupIterations(); int withinAttempts = 0, withinAccepts = 0;
			for (ReversibleJumpWithinModelKernel kernel : within) if (kernel.applicable(state, target)) {
				ReversibleJumpWithinModelTransition transition = kernel.update(state, logJoint, target, random, warmup);
				validateState(target, transition.state());
				if (transition.state().modelId() != state.modelId()) throw new IllegalStateException("within-model kernel changed model id");
				if (!Double.isFinite(transition.logJoint())) {
					warnings.add("non-finite within-model log joint at iteration " + iteration);
					return result(draws, values, stats, transition.state(), transition.logJoint(), priorCompleted + completed,
							random, weights, attempts, accepts, invalid, warnings,
							ReversibleJumpResult.Status.NUMERICAL_FAILURE, false);
				}
				state = transition.state(); logJoint = transition.logJoint(); withinAttempts++;
				if (transition.accepted()) withinAccepts++;
			}
			long fromModel = state.modelId(); int moveIndex = chooseMove(state, target, weights, random); String moveName = null;
			boolean jumpAttempted = moveIndex >= 0, jumpAccepted = false, invalidProposal = false;
			double probability = jumpAttempted ? 0.0 : 1.0, logRatio = jumpAttempted ? Double.NEGATIVE_INFINITY : 0.0;
			if (jumpAttempted) {
				moveName = moves[moveIndex].name(); if (!warmup) attempts[moveIndex]++;
				ReversibleJumpState jumpOrigin = state;
				double forwardSelection = selectionLogProbability(moveIndex, state, target, weights);
				ReversibleJumpProposal proposal = moves[moveIndex].propose(state, target, random);
				if (!proposal.valid()) { invalidProposal = true; if (!warmup) invalid[moveIndex]++; }
				else {
					ReversibleJumpState proposed = proposal.proposedState(); validateState(target, proposed);
					int reverse = moveIndex(proposal.reverseMove());
					if (reverse < 0 || !moves[reverse].applicable(proposed, target)) throw new IllegalStateException("reverse move is unavailable: " + proposal.reverseMove());
					double proposedLogJoint = target.logJoint(proposed);
					if (Double.isFinite(proposedLogJoint)) {
						double reverseSelection = selectionLogProbability(reverse, proposed, target, weights);
						logRatio = proposedLogJoint - logJoint + proposal.logReverseDensity() - proposal.logForwardDensity()
								+ reverseSelection - forwardSelection + proposal.logAbsJacobian();
						if (Double.isNaN(logRatio)) {
							invalidProposal = true; logRatio = Double.NEGATIVE_INFINITY; if (!warmup) invalid[moveIndex]++;
						} else {
							probability = Math.min(1.0, Math.exp(Math.min(0.0, logRatio)));
							if (Math.log(random.nextDouble()) < Math.min(0.0, logRatio)) {
								state = proposed; logJoint = proposedLogJoint; jumpAccepted = true; if (!warmup) accepts[moveIndex]++;
							}
						}
					}
				}
				if (warmup) {
					moves[moveIndex].warmupUpdate(jumpOrigin, proposal, target, jumpAccepted);
					if (options.adaptMoveWeights()) adaptWeight(weights, moveIndex, jumpAccepted, options, iteration);
				}
			}
			completed++;
			ReversibleJumpIterationStats iterationStats = new ReversibleJumpIterationStats(fromModel, state.modelId(), moveName,
					jumpAttempted, jumpAccepted, invalidProposal, probability, logRatio, withinAttempts, withinAccepts);
			options.progress(completed, total, warmup, iterationStats);
			if (options.cancelled()) return result(draws, values, stats, state, logJoint, priorCompleted + completed, random, weights,
					attempts, accepts, invalid, warnings, ReversibleJumpResult.Status.CANCELLED,
					resumed || completed >= options.warmupIterations());
			if (completed == options.warmupIterations()) {
				for (ReversibleJumpMove move : moves) move.freezeAdaptation();
				for (ReversibleJumpWithinModelKernel kernel : within) kernel.freezeAdaptation();
			}
			if (!warmup && (iteration - options.warmupIterations()) % options.thinning() == 0) {
				options.emit(retained++, state, logJoint, iterationStats);
				if (options.storeDraws()) { draws.add(state); values.add(Double.valueOf(logJoint)); stats.add(iterationStats); }
			}
			if (!Double.isFinite(logJoint)) { warnings.add("non-finite RJ state at iteration " + iteration);
				return result(draws, values, stats, state, logJoint, priorCompleted + completed, random, weights,
						attempts, accepts, invalid, warnings, ReversibleJumpResult.Status.NUMERICAL_FAILURE, false); }
		}
		return result(draws, values, stats, state, logJoint, priorCompleted + completed, random, weights,
				attempts, accepts, invalid, warnings, ReversibleJumpResult.Status.SUCCESS, true);
	}
	private ReversibleJumpResult result(List<ReversibleJumpState> draws, List<Double> values,
			List<ReversibleJumpIterationStats> statistics, ReversibleJumpState state, double logJoint, int completed,
			RandomEngine random, double[] weights, long[] attempts, long[] accepts, long[] invalid, List<String> warnings,
			ReversibleJumpResult.Status status, boolean warmupComplete) {
		String[] names = new String[moves.length]; for (int i = 0; i < names.length; i++) names[i] = moves[i].name();
		Map<String, double[]> adaptation = new LinkedHashMap<String, double[]>();
		for (ReversibleJumpMove move : moves) for (Map.Entry<String, double[]> entry : move.adaptationState().entrySet())
			adaptation.put("move/" + move.name() + "/" + entry.getKey(), entry.getValue());
		for (ReversibleJumpWithinModelKernel kernel : within) adaptation.putAll(kernel.adaptationState());
		ReversibleJumpCheckpoint checkpoint = new ReversibleJumpCheckpoint(state, logJoint, completed, random, names, weights, adaptation, warmupComplete);
		double[] logJoints = new double[values.size()]; for (int i = 0; i < values.size(); i++) logJoints[i] = values.get(i).doubleValue();
		return new ReversibleJumpResult(draws.toArray(new ReversibleJumpState[draws.size()]), logJoints,
				statistics.toArray(new ReversibleJumpIterationStats[statistics.size()]), checkpoint, status, warnings,
				names, attempts, accepts, invalid);
	}
	private int chooseMove(ReversibleJumpState state, ReversibleJumpTarget target, double[] weights, RandomEngine random) {
		double total = 0.0; int lastApplicable = -1; for (int i = 0; i < moves.length; i++) if (moves[i].applicable(state, target)) { total += weights[i]; lastApplicable = i; }
		if (!(total > 0.0)) return -1; double threshold = random.nextDouble() * total, cumulative = 0.0;
		for (int i = 0; i < moves.length; i++) if (moves[i].applicable(state, target)) { cumulative += weights[i]; if (threshold <= cumulative) return i; }
		return lastApplicable;
	}
	private double selectionLogProbability(int index, ReversibleJumpState state, ReversibleJumpTarget target, double[] weights) {
		double total = 0.0; for (int i = 0; i < moves.length; i++) if (moves[i].applicable(state, target)) total += weights[i];
		return Math.log(weights[index] / total);
	}
	private int moveIndex(String name) { for (int i = 0; i < moves.length; i++) if (moves[i].name().equals(name)) return i; return -1; }
	private static Map<String, double[]> componentState(Map<String, double[]> state, String prefix) {
		Map<String, double[]> result = new LinkedHashMap<String, double[]>();
		for (Map.Entry<String, double[]> entry : state.entrySet()) if (entry.getKey().startsWith(prefix)) result.put(entry.getKey().substring(prefix.length()), entry.getValue());
		return result;
	}
	private void adaptWeight(double[] weights, int index, boolean accepted, ReversibleJumpSamplingOptions options, int iteration) {
		double rate = 1.0 / Math.sqrt(iteration + 10.0); weights[index] *= Math.exp(rate * ((accepted ? 1.0 : 0.0) - options.targetJumpAcceptance()));
		weights[index] = Math.max(options.minimumMoveWeight(), Math.min(1e6, weights[index]));
	}
	private static void validateState(ReversibleJumpTarget target, ReversibleJumpState state) {
		ReversibleJumpModelSpace space = target.modelSpace(state.modelId());
		if (space == null || space.modelId() != state.modelId() || space.dimension() != state.dimension())
			throw new IllegalArgumentException("state does not match its RJ model space");
	}
}
