/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Multinomial-candidate NUTS with dual averaging and diagonal/dense metrics. */
public final class NoUTurnSampler implements Sampler {
	private static final class Tree {
		HamiltonianSupport.Point left;
		HamiltonianSupport.Point right;
		HamiltonianSupport.Point candidate;
		double logWeight;
		int leapfrogs;
		boolean continuing;
		boolean divergent;
		double acceptanceSum;
		int acceptanceCount;
		double maximumEnergyError;
	}

	@Override public ChainResult sample(LogDensity target, double[] initialState,
			SamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || initialState.length == 0
				|| options == null || random == null)
			throw new IllegalArgumentException("target, state, options and random are required");
		DifferentiableLogDensity differentiable = HamiltonianSupport.gradientTarget(target, options);
		HamiltonianSupport.Point state = HamiltonianSupport.at(differentiable, initialState);
		ChainAccumulator output = new ChainAccumulator();
		if (!finite(state)) {
			output.warn("initial log density or gradient is not finite");
			return output.result(initialState, state.logDensity, 0, random, null,
					ChainResult.Status.INVALID_INITIAL_STATE);
		}
		HamiltonianSupport.MassMatrix mass = new HamiltonianSupport.MassMatrix(initialState.length);
		double initialStep = options.adaptStepSize()
				? HamiltonianSupport.findReasonableStep(differentiable,
						state, mass, random, options.stepSize()) : options.stepSize();
		double step = initialStep;
		HamiltonianSupport.DualAveraging adaptation =
				new HamiltonianSupport.DualAveraging(step, options.targetAcceptance());
		HamiltonianSupport.RunningCovariance covariance =
				new HamiltonianSupport.RunningCovariance(initialState.length);
		double warmupAcceptance = 0.0;
		int completed = 0;
		int total = options.warmupIterations()
				+ options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) return finish(output, state, completed, random,
					options, initialStep, step, mass, warmupAcceptance,
					ChainResult.Status.CANCELLED);
			Transition transition = transition(differentiable, state, step, mass,
					options, random);
			state = transition.state;
			completed++;
			if (iteration < options.warmupIterations()) {
				warmupAcceptance += transition.acceptanceProbability;
				covariance.add(state.q);
				if (options.adaptStepSize())
					step = adaptation.update(transition.acceptanceProbability);
				if (options.adaptMassMatrix() && iteration > 50
						&& (iteration + 1) % 50 == 0) {
					mass.update(covariance.covariance(), options.denseMassMatrix());
				}
				if (iteration + 1 == options.warmupIterations()
						&& options.adaptStepSize()) step = adaptation.averaged();
			} else if ((iteration - options.warmupIterations() + 1)
					% options.thinning() == 0) {
				output.add(state.q, state.logDensity, new IterationStats(
						transition.moved, transition.acceptanceProbability, step,
						transition.energy, transition.maximumEnergyError,
						transition.divergent, transition.depth,
						transition.depth >= options.maximumTreeDepth(),
						transition.leapfrogs));
			}
		}
		return finish(output, state, completed, random, options, initialStep, step,
				mass, warmupAcceptance, ChainResult.Status.SUCCESS);
	}

	private static final class Transition {
		HamiltonianSupport.Point state;
		boolean moved;
		double acceptanceProbability;
		double energy;
		double maximumEnergyError;
		boolean divergent;
		int depth;
		int leapfrogs;
	}

	private static Transition transition(DifferentiableLogDensity target,
			HamiltonianSupport.Point current, double step,
			HamiltonianSupport.MassMatrix mass, SamplingOptions options,
			RandomEngine random) {
		double[] momentum = mass.momentum(random);
		HamiltonianSupport.Point start = new HamiltonianSupport.Point(current.q,
				momentum, current.gradient, current.logDensity);
		double initialJoint = current.logDensity - mass.kinetic(momentum);
		HamiltonianSupport.Point left = start;
		HamiltonianSupport.Point right = start;
		HamiltonianSupport.Point candidate = current;
		double logWeight = initialJoint;
		int depth = 0;
		int leapfrogs = 0;
		boolean continuing = true;
		boolean divergent = false;
		double acceptanceSum = 0.0;
		int acceptanceCount = 0;
		double maximumError = 0.0;
		while (continuing && depth < options.maximumTreeDepth()) {
			int direction = random.nextDouble() < 0.5 ? -1 : 1;
			Tree tree = direction < 0
					? buildTree(target, left, direction, depth, step,
							initialJoint, mass, options, random)
					: buildTree(target, right, direction, depth, step,
							initialJoint, mass, options, random);
			if (direction < 0) left = tree.left; else right = tree.right;
			double combinedWeight = logAdd(logWeight, tree.logWeight);
			if (tree.continuing && Double.isFinite(tree.logWeight)
					&& random.nextDouble() < Math.exp(tree.logWeight - combinedWeight))
				candidate = tree.candidate;
			logWeight = combinedWeight;
			leapfrogs += tree.leapfrogs;
			acceptanceSum += tree.acceptanceSum;
			acceptanceCount += tree.acceptanceCount;
			divergent |= tree.divergent;
			maximumError = Math.max(maximumError, tree.maximumEnergyError);
			continuing = tree.continuing && noUTurn(left, right, mass);
			depth++;
		}
		Transition result = new Transition();
		result.state = new HamiltonianSupport.Point(candidate.q,
				new double[candidate.q.length], candidate.gradient, candidate.logDensity);
		result.moved = !same(current.q, candidate.q);
		result.acceptanceProbability = acceptanceCount == 0 ? 0.0
				: acceptanceSum / acceptanceCount;
		result.energy = -initialJoint;
		result.maximumEnergyError = maximumError;
		result.divergent = divergent;
		result.depth = depth;
		result.leapfrogs = leapfrogs;
		return result;
	}

	private static Tree buildTree(DifferentiableLogDensity target,
			HamiltonianSupport.Point start, int direction,
			int depth, double step, double initialJoint,
			HamiltonianSupport.MassMatrix mass, SamplingOptions options,
			RandomEngine random) {
		if (depth == 0) {
			HamiltonianSupport.Point next = HamiltonianSupport.leapfrog(target, start,
					direction * step, mass);
			double joint = next.logDensity - mass.kinetic(next.p);
			double energyError = initialJoint - joint;
			Tree tree = new Tree();
			tree.left = next; tree.right = next; tree.candidate = next;
			tree.divergent = !Double.isFinite(energyError)
					|| energyError > options.maximumEnergyError();
			tree.continuing = !tree.divergent;
			tree.logWeight = tree.continuing ? joint : Double.NEGATIVE_INFINITY;
			tree.leapfrogs = 1;
			tree.acceptanceSum = Double.isFinite(joint)
					? Math.min(1.0, Math.exp(joint - initialJoint)) : 0.0;
			tree.acceptanceCount = 1;
			tree.maximumEnergyError = Double.isFinite(energyError)
					? Math.max(0.0, energyError) : Double.POSITIVE_INFINITY;
			return tree;
		}
		Tree first = buildTree(target, start, direction, depth - 1,
				step, initialJoint, mass, options, random);
		if (!first.continuing) return first;
		HamiltonianSupport.Point edge = direction < 0 ? first.left : first.right;
		Tree second = buildTree(target, edge, direction, depth - 1,
				step, initialJoint, mass, options, random);
		Tree combined = new Tree();
		combined.left = direction < 0 ? second.left : first.left;
		combined.right = direction < 0 ? first.right : second.right;
		combined.candidate = first.candidate;
		combined.logWeight = logAdd(first.logWeight, second.logWeight);
		if (Double.isFinite(second.logWeight) && random.nextDouble()
				< Math.exp(second.logWeight - combined.logWeight))
			combined.candidate = second.candidate;
		combined.leapfrogs = first.leapfrogs + second.leapfrogs;
		combined.divergent = first.divergent || second.divergent;
		combined.continuing = second.continuing
				&& noUTurn(combined.left, combined.right, mass);
		combined.acceptanceSum = first.acceptanceSum + second.acceptanceSum;
		combined.acceptanceCount = first.acceptanceCount + second.acceptanceCount;
		combined.maximumEnergyError = Math.max(first.maximumEnergyError,
				second.maximumEnergyError);
		return combined;
	}

	private static double logAdd(double first, double second) {
		if (first == Double.NEGATIVE_INFINITY) return second;
		if (second == Double.NEGATIVE_INFINITY) return first;
		double maximum = Math.max(first, second);
		return maximum + Math.log1p(Math.exp(Math.min(first, second) - maximum));
	}

	private static boolean noUTurn(HamiltonianSupport.Point left,
			HamiltonianSupport.Point right, HamiltonianSupport.MassMatrix mass) {
		double[] difference = new double[left.q.length];
		for (int i = 0; i < difference.length; i++)
			difference[i] = right.q[i] - left.q[i];
		return mass.velocityDot(left.p, difference) >= 0.0
				&& mass.velocityDot(right.p, difference) >= 0.0;
	}

	private static boolean finite(HamiltonianSupport.Point point) {
		if (!Double.isFinite(point.logDensity)) return false;
		for (double value : point.gradient) if (!Double.isFinite(value)) return false;
		return true;
	}
	private static boolean same(double[] first, double[] second) {
		for (int i = 0; i < first.length; i++)
			if (Double.doubleToLongBits(first[i]) != Double.doubleToLongBits(second[i])) return false;
		return true;
	}
	private static ChainResult finish(ChainAccumulator output,
			HamiltonianSupport.Point state, int completed, RandomEngine random,
			SamplingOptions options, double initialStep, double finalStep,
			HamiltonianSupport.MassMatrix mass, double acceptanceSum,
			ChainResult.Status status) {
		WarmupResult warmup = WarmupResult.withInverseMassMatrix(
				options.warmupIterations(), initialStep,
				finalStep, mass.inverseMatrix(), options.warmupIterations() == 0
				? Double.NaN : acceptanceSum / options.warmupIterations());
		return output.result(state.q, state.logDensity, completed, random, warmup, status);
	}
}
