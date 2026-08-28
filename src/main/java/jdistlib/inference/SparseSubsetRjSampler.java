/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;

import jdistlib.rng.RandomEngine;

/** Allocation-conscious add/drop/swap RJMCMC for very large sparse candidate universes. */
public final class SparseSubsetRjSampler {
	private static final String[] MOVE_NAMES = {"add", "drop", "swap"};
	private final SparseCandidateProposal candidates; private final SparseCoefficientProposal coefficients;
	private final double initialWithinScale, targetWithinAcceptance; private final double[] initialMoveWeights;
	public SparseSubsetRjSampler(SparseCandidateProposal candidates, SparseCoefficientProposal coefficients,
			double initialWithinScale, double targetWithinAcceptance) {
		this(candidates, coefficients, initialWithinScale, targetWithinAcceptance, new double[] {1.0, 1.0, 0.5});
	}
	public SparseSubsetRjSampler(SparseCandidateProposal candidates, SparseCoefficientProposal coefficients,
			double initialWithinScale, double targetWithinAcceptance, double[] moveWeights) {
		if (candidates == null || coefficients == null || !(initialWithinScale > 0.0) || !Double.isFinite(initialWithinScale)
				|| !(targetWithinAcceptance > 0.0 && targetWithinAcceptance < 1.0) || moveWeights == null || moveWeights.length != 3)
			throw new IllegalArgumentException("candidate, coefficient, scale, acceptance, and three move weights required");
		this.candidates = candidates; this.coefficients = coefficients; this.initialWithinScale = initialWithinScale;
		this.targetWithinAcceptance = targetWithinAcceptance; this.initialMoveWeights = moveWeights.clone();
		for (double weight : this.initialMoveWeights) if (!(weight > 0.0) || !Double.isFinite(weight)) throw new IllegalArgumentException("positive finite move weights required");
	}
	public SparseSubsetResult sample(SparseSubsetTarget target, SparseSubsetState initialState,
			SparseSubsetSamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || options == null || random == null) throw new IllegalArgumentException("target, state, options, and random required");
		target.validate(initialState); int sizes = target.maximumActive() + 1, count = target.candidateCount();
		double[] logScales = new double[sizes]; for (int i = 0; i < sizes; i++) logScales[i] = Math.log(initialWithinScale);
		return run(target, initialState, target.logJoint(initialState), options, random, 0L, 0L,
				initialMoveWeights.clone(), logScales, new long[sizes], new long[3], new long[3], new long[3],
				new long[sizes], new long[count], new long[count], new double[count], new double[count],
				new double[target.commonDimension()], new double[target.commonDimension()]);
	}
	public SparseSubsetResult resume(SparseSubsetTarget target, SparseSubsetCheckpoint checkpoint,
			SparseSubsetSamplingOptions options) {
		if (target == null || checkpoint == null || options == null) throw new IllegalArgumentException("target, checkpoint, and options required");
		target.validate(checkpoint.state()); validateCheckpoint(target, checkpoint);
		if (checkpoint.warmupIterations() != options.warmupIterations())
			throw new IllegalArgumentException("sparse checkpoint warmup target does not match options");
		String[] names = checkpoint.moveNames(); for (int i = 0; i < names.length; i++) if (!MOVE_NAMES[i].equals(names[i])) throw new IllegalArgumentException("sparse checkpoint move schedule mismatch");
		return run(target, checkpoint.state(), checkpoint.logJoint(), options, checkpoint.random(), checkpoint.completedTransitions(), checkpoint.retainedDraws(),
				checkpoint.moveWeights(), checkpoint.logScales(), checkpoint.scaleUpdates(), checkpoint.moveAttempts(), checkpoint.moveAccepts(), checkpoint.invalidProposals(),
				checkpoint.modelSizeCounts(), checkpoint.inclusionCounts(), checkpoint.coefficientCounts(), checkpoint.coefficientSums(), checkpoint.coefficientSquareSums(),
				checkpoint.commonSums(), checkpoint.commonSquareSums());
	}
	private SparseSubsetResult run(SparseSubsetTarget target, SparseSubsetState initialState, double initialLogJoint,
			SparseSubsetSamplingOptions options, RandomEngine random, long priorCompleted, long priorRetained,
			double[] moveWeights, double[] logScales, long[] scaleUpdates, long[] attempts, long[] accepts,
			long[] invalid, long[] sizeCounts, long[] inclusion, long[] coefficientCounts,
			double[] coefficientSums, double[] coefficientSquares, double[] commonSums, double[] commonSquares) {
		List<SparseSubsetState> draws = new ArrayList<SparseSubsetState>(); List<Double> values = new ArrayList<Double>();
		List<SparseSubsetIterationStats> statistics = new ArrayList<SparseSubsetIterationStats>(); List<String> warnings = new ArrayList<String>();
		if (!Double.isFinite(initialLogJoint)) return result(draws, values, statistics, initialState, initialLogJoint,
				priorCompleted, priorRetained, random, moveWeights, logScales, scaleUpdates, attempts, accepts, invalid,
				sizeCounts, inclusion, coefficientCounts, coefficientSums, coefficientSquares, commonSums, commonSquares,
				options, SparseSubsetResult.Status.INVALID_INITIAL_STATE, warnings);
		SparseSubsetState state = initialState; double logJoint = initialLogJoint; int completed = 0;
		for (; completed < options.segmentTransitions(); completed++) {
			long global = priorCompleted + completed; boolean warmup = global < options.warmupIterations(); int size = state.size();
			WithinTransition within = within(state, logJoint, target, random, Math.exp(logScales[size]));
			state = within.state; logJoint = within.logJoint;
			if (warmup) adaptScale(logScales, scaleUpdates, size, within.accepted);
			int fromSize = state.size(), move = chooseMove(state, target, moveWeights, random); Proposal proposal = propose(move, state, target, random);
			double probability = 0.0, logRatio = Double.NEGATIVE_INFINITY; boolean jumpAccepted = false, invalidProposal = !proposal.valid;
			if (!warmup) attempts[move]++;
			if (proposal.valid) {
				double proposedLogJoint = target.logJoint(proposal.state);
				if (Double.isFinite(proposedLogJoint)) {
					double forwardSchedule = scheduleLogProbability(move, state, target, moveWeights);
					double reverseSchedule = scheduleLogProbability(reverse(move), proposal.state, target, moveWeights);
					logRatio = proposedLogJoint - logJoint + proposal.logReverse - proposal.logForward + reverseSchedule - forwardSchedule;
					if (!Double.isNaN(logRatio)) { probability = Math.min(1.0, Math.exp(Math.min(0.0, logRatio)));
						if (Math.log(random.nextDouble()) < Math.min(0.0, logRatio)) { state = proposal.state; logJoint = proposedLogJoint; jumpAccepted = true; if (!warmup) accepts[move]++; }
					} else invalidProposal = true;
				}
			}
			if (invalidProposal && !warmup) invalid[move]++;
			if (warmup && options.adaptMoveWeights()) adaptWeight(moveWeights, move, jumpAccepted, options, global);
			SparseSubsetIterationStats stats = new SparseSubsetIterationStats(fromSize, state.size(), MOVE_NAMES[move], jumpAccepted,
					invalidProposal, within.accepted, probability, logRatio);
			long totalCompleted = global + 1L; options.progress(completed + 1, totalCompleted, warmup, stats);
			if (!warmup) { long samplingIndex = global - options.warmupIterations(); if (samplingIndex % options.thinning() == 0L) {
				long retainedIndex = priorRetained + draws.size(); options.emit(retainedIndex, state, logJoint, stats); accumulate(state, sizeCounts, inclusion,
						coefficientCounts, coefficientSums, coefficientSquares, commonSums, commonSquares);
				if (options.storeDraws()) { draws.add(state); values.add(Double.valueOf(logJoint)); statistics.add(stats); }
				else priorRetained++;
			} }
			if (options.cancelled()) { completed++; return result(draws, values, statistics, state, logJoint, priorCompleted + completed,
					priorRetained + draws.size(), random, moveWeights, logScales, scaleUpdates, attempts, accepts, invalid,
					sizeCounts, inclusion, coefficientCounts, coefficientSums, coefficientSquares, commonSums, commonSquares,
					options, SparseSubsetResult.Status.CANCELLED, warnings); }
			if (!Double.isFinite(logJoint)) { warnings.add("non-finite sparse state at transition " + global); completed++;
				return result(draws, values, statistics, state, logJoint, priorCompleted + completed, priorRetained + draws.size(), random,
						moveWeights, logScales, scaleUpdates, attempts, accepts, invalid, sizeCounts, inclusion, coefficientCounts,
						coefficientSums, coefficientSquares, commonSums, commonSquares, options, SparseSubsetResult.Status.NUMERICAL_FAILURE, warnings); }
		}
		return result(draws, values, statistics, state, logJoint, priorCompleted + completed, priorRetained + draws.size(), random,
				moveWeights, logScales, scaleUpdates, attempts, accepts, invalid, sizeCounts, inclusion, coefficientCounts,
				coefficientSums, coefficientSquares, commonSums, commonSquares, options, SparseSubsetResult.Status.SUCCESS, warnings);
	}
	private static final class WithinTransition { final SparseSubsetState state; final double logJoint; final boolean accepted;
		WithinTransition(SparseSubsetState state, double logJoint, boolean accepted) { this.state = state; this.logJoint = logJoint; this.accepted = accepted; } }
	private static WithinTransition within(SparseSubsetState state, double logJoint, SparseSubsetTarget target, RandomEngine random, double scale) {
		double[] common = state.commonParameters(), beta = state.coefficients();
		for (int i = 0; i < common.length; i++) common[i] += scale * random.nextGaussian();
		for (int i = 0; i < beta.length; i++) beta[i] += scale * random.nextGaussian();
		SparseSubsetState proposed = target.state(state.activeCandidates(), common, beta); double proposedLogJoint = target.logJoint(proposed);
		if (Double.isFinite(proposedLogJoint) && Math.log(random.nextDouble()) < Math.min(0.0, proposedLogJoint - logJoint)) return new WithinTransition(proposed, proposedLogJoint, true);
		return new WithinTransition(state, logJoint, false);
	}
	private static final class Proposal { final SparseSubsetState state; final double logForward, logReverse; final boolean valid;
		Proposal(SparseSubsetState state, double logForward, double logReverse) { this.state = state; this.logForward = logForward; this.logReverse = logReverse; valid = true; } }
	private Proposal propose(int move, SparseSubsetState state, SparseSubsetTarget target, RandomEngine random) {
		if (move == 0) {
			SparseCandidateChoice choice = candidates.sample(state, target, random); double value = coefficients.sample(choice.candidate(), state, random);
			SparseSubsetState proposed = add(state, choice.candidate(), value, target);
			return new Proposal(proposed, choice.logProbability() + coefficients.logDensity(value, choice.candidate(), state), -Math.log(proposed.size()));
		}
		if (move == 1) {
			int position = random.nextInt(state.size()), candidate = state.activeCandidate(position); double value = state.coefficient(position);
			SparseSubsetState proposed = remove(state, position, target);
			return new Proposal(proposed, -Math.log(state.size()), candidates.logProbability(candidate, proposed, target)
					+ coefficients.logDensity(value, candidate, proposed));
		}
		int removedPosition = random.nextInt(state.size()), removed = state.activeCandidate(removedPosition); double removedValue = state.coefficient(removedPosition);
		SparseSubsetState base = remove(state, removedPosition, target); SparseCandidateChoice choice = candidates.sample(base, target, random);
		double addedValue = coefficients.sample(choice.candidate(), base, random); SparseSubsetState proposed = add(base, choice.candidate(), addedValue, target);
		double forward = -Math.log(state.size()) + choice.logProbability() + coefficients.logDensity(addedValue, choice.candidate(), base);
		double reverseDensity = -Math.log(proposed.size()) + candidates.logProbability(removed, base, target)
				+ coefficients.logDensity(removedValue, removed, base);
		return new Proposal(proposed, forward, reverseDensity);
	}
	private static SparseSubsetState add(SparseSubsetState state, int candidate, double value, SparseSubsetTarget target) {
		if (state.active(candidate)) throw new IllegalArgumentException("candidate already active"); int[] old = state.activeCandidates(), active = new int[old.length + 1];
		double[] oldBeta = state.coefficients(), beta = new double[active.length]; int source = 0;
		for (int i = 0; i < active.length; i++) if (source < old.length && old[source] < candidate) { active[i] = old[source]; beta[i] = oldBeta[source++]; }
		else { active[i] = candidate; beta[i] = value; candidate = Integer.MAX_VALUE; }
		return target.state(active, state.commonParameters(), beta);
	}
	private static SparseSubsetState remove(SparseSubsetState state, int position, SparseSubsetTarget target) {
		int[] active = new int[state.size() - 1]; double[] beta = new double[active.length];
		for (int source = 0, destination = 0; source < state.size(); source++) if (source != position) { active[destination] = state.activeCandidate(source); beta[destination++] = state.coefficient(source); }
		return target.state(active, state.commonParameters(), beta);
	}
	private static int chooseMove(SparseSubsetState state, SparseSubsetTarget target, double[] weights, RandomEngine random) {
		double total = 0.0; int last = -1; for (int move = 0; move < 3; move++) if (applicable(move, state, target)) { total += weights[move]; last = move; }
		double threshold = random.nextDouble() * total, cumulative = 0.0; for (int move = 0; move < 3; move++) if (applicable(move, state, target)) { cumulative += weights[move]; if (threshold <= cumulative) return move; }
		return last;
	}
	private static double scheduleLogProbability(int move, SparseSubsetState state, SparseSubsetTarget target, double[] weights) {
		double total = 0.0; for (int i = 0; i < 3; i++) if (applicable(i, state, target)) total += weights[i];
		if (!applicable(move, state, target)) return Double.NEGATIVE_INFINITY; return Math.log(weights[move] / total);
	}
	private static boolean applicable(int move, SparseSubsetState state, SparseSubsetTarget target) {
		if (move == 0) return state.size() < target.maximumActive() && state.size() < target.candidateCount();
		if (move == 1) return state.size() > 0; return state.size() > 0 && state.size() < target.candidateCount();
	}
	private static int reverse(int move) { return move == 0 ? 1 : move == 1 ? 0 : 2; }
	private void adaptScale(double[] logScales, long[] updates, int size, boolean accepted) {
		updates[size]++; double rate = 1.0 / Math.sqrt(updates[size] + 10.0);
		logScales[size] += rate * ((accepted ? 1.0 : 0.0) - targetWithinAcceptance);
		logScales[size] = Math.max(Math.log(1e-8), Math.min(Math.log(1e4), logScales[size]));
	}
	private static void adaptWeight(double[] weights, int move, boolean accepted, SparseSubsetSamplingOptions options, long iteration) {
		double rate = 1.0 / Math.sqrt(iteration + 10.0); weights[move] *= Math.exp(rate * ((accepted ? 1.0 : 0.0) - options.targetJumpAcceptance()));
		weights[move] = Math.max(options.minimumMoveWeight(), Math.min(1e6, weights[move]));
	}
	private static void accumulate(SparseSubsetState state, long[] sizeCounts, long[] inclusion, long[] coefficientCounts,
			double[] coefficientSums, double[] coefficientSquares, double[] commonSums, double[] commonSquares) {
		sizeCounts[state.size()]++; for (int i = 0; i < state.commonDimension(); i++) { double value = state.commonParameter(i); commonSums[i] += value; commonSquares[i] += value * value; }
		for (int i = 0; i < state.size(); i++) { int candidate = state.activeCandidate(i); double value = state.coefficient(i); inclusion[candidate]++; coefficientCounts[candidate]++;
			coefficientSums[candidate] += value; coefficientSquares[candidate] += value * value; }
	}
	private static void validateCheckpoint(SparseSubsetTarget target, SparseSubsetCheckpoint checkpoint) {
		if (checkpoint.moveNames().length != 3 || checkpoint.logScales().length != target.maximumActive() + 1
				|| checkpoint.inclusionCounts().length != target.candidateCount() || checkpoint.commonSums().length != target.commonDimension())
			throw new IllegalArgumentException("sparse checkpoint dimensions do not match target");
	}
	private static SparseSubsetResult result(List<SparseSubsetState> draws, List<Double> values,
			List<SparseSubsetIterationStats> statistics, SparseSubsetState state, double logJoint, long completed, long retained,
			RandomEngine random, double[] weights, double[] logScales, long[] scaleUpdates, long[] attempts, long[] accepts,
			long[] invalid, long[] sizeCounts, long[] inclusion, long[] coefficientCounts, double[] coefficientSums,
			double[] coefficientSquares, double[] commonSums, double[] commonSquares, SparseSubsetSamplingOptions options,
			SparseSubsetResult.Status status, List<String> warnings) {
		double[] logJoints = new double[values.size()]; for (int i = 0; i < values.size(); i++) logJoints[i] = values.get(i).doubleValue();
		SparseSubsetCheckpoint checkpoint = new SparseSubsetCheckpoint(state, logJoint, completed, retained, random, MOVE_NAMES,
				weights, logScales, scaleUpdates, attempts, accepts, invalid, sizeCounts, inclusion, coefficientCounts,
				coefficientSums, coefficientSquares, commonSums, commonSquares, options.warmupIterations(), completed >= options.warmupIterations());
		return new SparseSubsetResult(draws.toArray(new SparseSubsetState[draws.size()]), logJoints,
				statistics.toArray(new SparseSubsetIterationStats[statistics.size()]), checkpoint, status, warnings);
	}
}
