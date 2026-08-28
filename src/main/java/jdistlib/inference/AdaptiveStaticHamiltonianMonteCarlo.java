/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Coordinated many-chain static HMC with ChEES or SNAPER trajectory adaptation. */
public final class AdaptiveStaticHamiltonianMonteCarlo {
	private AdaptiveStaticHamiltonianMonteCarlo() {}
	public static AdaptiveStaticHmcResult sample(DifferentiableLogDensity target, double[][] initialStates,
			AdaptiveStaticHmcOptions options, long baseSeed) {
		if (target == null || initialStates == null || initialStates.length < 2 || options == null)
			throw new IllegalArgumentException("target, at least two states, and options are required");
		int dimension = initialStates[0].length, chainCount = initialStates.length;
		HamiltonianSupport.Point[] states = new HamiltonianSupport.Point[chainCount]; RandomEngine[] random = new RandomEngine[chainCount];
		ChainAccumulator[] output = new ChainAccumulator[chainCount];
		for (int chain = 0; chain < chainCount; chain++) { if (initialStates[chain].length != dimension) throw new IllegalArgumentException("state dimensions must match");
			states[chain] = HamiltonianSupport.at(target, initialStates[chain]); random[chain] = new MersenneTwister(mixSeed(baseSeed, chain)); output[chain] = new ChainAccumulator(); }
		HamiltonianSupport.MassMatrix mass = new HamiltonianSupport.MassMatrix(dimension, options.metric());
		double step = HamiltonianSupport.findReasonableStep(target, states[0], mass, random[0], options.stepSize());
		HamiltonianSupport.DualAveraging stepAdaptation = new HamiltonianSupport.DualAveraging(step, options.targetAcceptance());
		HamiltonianSupport.RunningCovariance covariance = new HamiltonianSupport.RunningCovariance(dimension);
		int[] candidates = candidates(options.maximumLeapfrogSteps()), visits = new int[candidates.length]; double[] scores = new double[candidates.length];
		double[] principal = unit(dimension); int total = options.warmupIterations() + options.sampleIterations() * options.thinning();
		for (int iteration = 0; iteration < total; iteration++) {
			boolean warmup = iteration < options.warmupIterations(); int candidate = warmup ? chooseCandidate(iteration, scores, visits) : best(scores, visits);
			int leapfrogs = candidates[candidate]; double acceptance = 0.0, reward = 0.0;
			double[] center = ensembleMean(states);
			HamiltonianSupport.Point[] proposals = new HamiltonianSupport.Point[chainCount];
			double[] transitionSteps = new double[chainCount], initialEnergies = new double[chainCount];
			for (int chain = 0; chain < chainCount; chain++) {
				transitionSteps[chain] = HamiltonianSupport.jitter(step, options.jitter(), random[chain]);
				double[] momentum = mass.momentum(random[chain]); proposals[chain] = new HamiltonianSupport.Point(states[chain].q, momentum, states[chain].gradient, states[chain].logDensity);
				initialEnergies[chain] = -states[chain].logDensity + mass.kinetic(momentum);
			}
			for (int leapfrog = 0; leapfrog < leapfrogs; leapfrog++)
				proposals = leapfrogBatch(target, proposals, transitionSteps, mass);
			for (int chain = 0; chain < chainCount; chain++) {
				HamiltonianSupport.Point proposal = proposals[chain];
				double proposedEnergy = -proposal.logDensity + mass.kinetic(proposal.p), error = proposedEnergy - initialEnergies[chain];
				double probability = finite(proposal) && Double.isFinite(error) ? Math.min(1.0, Math.exp(-error)) : 0.0;
				boolean divergent = !Double.isFinite(error) || Math.abs(error) > options.maximumEnergyError(); boolean accepted = !divergent && random[chain].nextDouble() < probability;
				double[] next = accepted ? proposal.q : states[chain].q;
				reward += objective(options.criterion(), states[chain].q, next, center, principal) / leapfrogs; acceptance += probability;
				if (accepted) states[chain] = new HamiltonianSupport.Point(proposal.q, new double[dimension], proposal.gradient, proposal.logDensity);
				IterationStats stats = new IterationStats(accepted, probability, step, proposedEnergy, error, divergent, 0, false, leapfrogs, mass.conditionNumber());
				if (!warmup && (iteration - options.warmupIterations() + 1) % options.thinning() == 0) output[chain].add(states[chain].q, states[chain].logDensity, stats);
				if (warmup) covariance.add(states[chain].q);
			}
			if (warmup) {
				visits[candidate]++; scores[candidate] += (reward / chainCount - scores[candidate]) / visits[candidate];
				step = stepAdaptation.update(acceptance / chainCount);
				if ((iteration + 1) % Math.max(25, options.warmupIterations() / 5) == 0 && covariance.count() > 2) {
					double[][] sampleCovariance = covariance.covariance(); mass.update(sampleCovariance, options.metric()); principal = principal(sampleCovariance, principal); covariance.reset();
				}
				if (iteration + 1 == options.warmupIterations()) step = stepAdaptation.averaged();
			}
		}
		int selected = best(scores, visits); ChainResult[] chains = new ChainResult[chainCount];
		for (int chain = 0; chain < chainCount; chain++) { WarmupResult warmup = WarmupResult.withInverseMassMatrix(options.warmupIterations(), options.stepSize(), step,
				mass.inverseMatrix(), Double.NaN); chains[chain] = output[chain].result(states[chain].q, states[chain].logDensity, total, random[chain], warmup, ChainResult.Status.SUCCESS); }
		return new AdaptiveStaticHmcResult(chains, candidates[selected], step, principal, scores);
	}
	private static HamiltonianSupport.Point[] leapfrogBatch(DifferentiableLogDensity target,
			HamiltonianSupport.Point[] points, double[] steps, HamiltonianSupport.MassMatrix mass) {
		int chains = points.length, dimension = points[0].q.length; double[][] positions = new double[chains][];
		double[][] momenta = new double[chains][];
		for (int chain = 0; chain < chains; chain++) { momenta[chain] = points[chain].p.clone();
			for (int d = 0; d < dimension; d++) momenta[chain][d] += 0.5 * steps[chain] * points[chain].gradient[d];
			positions[chain] = points[chain].q.clone(); mass.addScaledVelocity(momenta[chain], steps[chain], positions[chain]); }
		double[] values = new double[chains]; double[][] gradients = new double[chains][dimension];
		if (target instanceof BatchedDifferentiableLogDensity)
			((BatchedDifferentiableLogDensity) target).logDensityAndGradientBatch(positions, values, gradients);
		else for (int chain = 0; chain < chains; chain++) values[chain] = target.logDensityAndGradient(positions[chain], gradients[chain]);
		HamiltonianSupport.Point[] result = new HamiltonianSupport.Point[chains];
		for (int chain = 0; chain < chains; chain++) { for (int d = 0; d < dimension; d++) momenta[chain][d] += 0.5 * steps[chain] * gradients[chain][d];
			result[chain] = new HamiltonianSupport.Point(positions[chain], momenta[chain], gradients[chain], values[chain]); }
		return result;
	}
	private static int[] candidates(int maximum) { List<Integer> values = new ArrayList<Integer>(); for (int value = 1; value < maximum; value *= 2) values.add(value);
		if (values.isEmpty() || values.get(values.size() - 1) != maximum) values.add(maximum); int[] result = new int[values.size()]; for (int i = 0; i < result.length; i++) result[i] = values.get(i); return result; }
	private static int chooseCandidate(int iteration, double[] scores, int[] visits) { if (iteration < scores.length) return iteration;
		double log = Math.log(iteration + 1.0), bestScore = Double.NEGATIVE_INFINITY; int result = 0; for (int i = 0; i < scores.length; i++) {
			double ucb = scores[i] + Math.sqrt(2.0 * log / Math.max(1, visits[i])); if (ucb > bestScore) { bestScore = ucb; result = i; } } return result; }
	private static int best(double[] scores, int[] visits) { int result = 0; double value = Double.NEGATIVE_INFINITY; for (int i = 0; i < scores.length; i++) if (visits[i] > 0 && scores[i] > value) { value = scores[i]; result = i; } return result; }
	private static double objective(AdaptiveStaticHmcOptions.Criterion criterion, double[] before, double[] after, double[] center, double[] principal) {
		if (criterion == AdaptiveStaticHmcOptions.Criterion.CHEES) { double first = 0.0, second = 0.0; for (int i = 0; i < before.length; i++) { first += (before[i] - center[i]) * (before[i] - center[i]); second += (after[i] - center[i]) * (after[i] - center[i]); }
			double change = second - first; return change * change; }
		double projection = 0.0; for (int i = 0; i < before.length; i++) projection += (after[i] - before[i]) * principal[i]; return projection * projection; }
	private static double[] ensembleMean(HamiltonianSupport.Point[] points) { double[] result = new double[points[0].q.length];
		for (HamiltonianSupport.Point point : points) for (int d = 0; d < result.length; d++) result[d] += point.q[d];
		for (int d = 0; d < result.length; d++) result[d] /= points.length; return result; }
	private static double[] principal(double[][] covariance, double[] initial) { double[] vector = initial.clone(); for (int iteration = 0; iteration < 30; iteration++) { double[] next = new double[vector.length]; double norm = 0.0;
		for (int i = 0; i < vector.length; i++) for (int j = 0; j < vector.length; j++) next[i] += covariance[i][j] * vector[j]; for (double value : next) norm += value * value;
		norm = Math.sqrt(norm); if (!(norm > 0.0)) break; for (int i = 0; i < vector.length; i++) vector[i] = next[i] / norm; } return vector; }
	private static double[] unit(int dimension) { double[] result = new double[dimension]; double value = 1.0 / Math.sqrt(dimension); for (int i = 0; i < dimension; i++) result[i] = value; return result; }
	private static boolean finite(HamiltonianSupport.Point point) { if (!Double.isFinite(point.logDensity)) return false; for (double value : point.gradient) if (!Double.isFinite(value)) return false; return true; }
	private static long mixSeed(long base, int chain) { long value = base + 0x9e3779b97f4a7c15L * (chain + 1L); value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
		value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL; return value ^ (value >>> 31); }
}
