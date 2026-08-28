/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.RandomEngine;

/** Replica exchange using random-walk within-temperature transitions. */
public final class ParallelTempering {
	private ParallelTempering() {}
	public static ParallelTemperingResult sample(TemperedLogDensity target,
			double[] initialState, double[] inverseTemperatures,
			SamplingOptions options, RandomEngine random) {
		if (target == null || initialState == null || inverseTemperatures == null
				|| inverseTemperatures.length < 2 || options == null || random == null)
			throw new IllegalArgumentException("target, state, temperatures, options and random are required");
		if (inverseTemperatures[0] != 1.0)
			throw new IllegalArgumentException("first inverse temperature must be 1");
		int replicas = inverseTemperatures.length;
		double[][] states = new double[replicas][];
		double[] base = new double[replicas], likelihood = new double[replicas];
		for (int i = 0; i < replicas; i++) {
			if (!(inverseTemperatures[i] > 0.0) || inverseTemperatures[i] > 1.0
					|| (i > 0 && inverseTemperatures[i] >= inverseTemperatures[i - 1]))
				throw new IllegalArgumentException("inverse temperatures must strictly decrease from 1");
			states[i] = initialState.clone();
			base[i] = target.baseLogDensity(states[i]);
			likelihood[i] = target.temperedLogDensity(states[i]);
		}
		int[] attempted = new int[replicas - 1], accepted = new int[replicas - 1];
		ChainAccumulator output = new ChainAccumulator();
		int total = options.warmupIterations() + options.sampleIterations() * options.thinning();
		int retained = 0;
		for (int iteration = 0; iteration < total; iteration++) {
			if (options.cancelled()) break;
			for (int replica = 0; replica < replicas; replica++) {
				double[] proposal = states[replica].clone();
				for (int j = 0; j < proposal.length; j++) proposal[j] += options.stepSize() * random.nextGaussian();
				double proposedBase = target.baseLogDensity(proposal);
				double proposedLikelihood = target.temperedLogDensity(proposal);
				double ratio = proposedBase - base[replica]
						+ inverseTemperatures[replica] * (proposedLikelihood - likelihood[replica]);
				if (Math.log(Math.max(Double.MIN_VALUE, random.nextDouble())) < ratio) {
					states[replica] = proposal; base[replica] = proposedBase;
					likelihood[replica] = proposedLikelihood;
				}
			}
			int parity = iteration & 1;
			for (int pair = parity; pair < replicas - 1; pair += 2) {
				attempted[pair]++;
				double ratio = (inverseTemperatures[pair] - inverseTemperatures[pair + 1])
						* (likelihood[pair + 1] - likelihood[pair]);
				if (Math.log(Math.max(Double.MIN_VALUE, random.nextDouble())) < ratio) {
					double[] state = states[pair]; states[pair] = states[pair + 1]; states[pair + 1] = state;
					double value = base[pair]; base[pair] = base[pair + 1]; base[pair + 1] = value;
					value = likelihood[pair]; likelihood[pair] = likelihood[pair + 1]; likelihood[pair + 1] = value;
					accepted[pair]++;
				}
			}
			IterationStats stats = new IterationStats(true, Double.NaN,
					options.stepSize(), Double.NaN, Double.NaN, false, 0, 0);
			if (iteration >= options.warmupIterations()
					&& (iteration - options.warmupIterations() + 1) % options.thinning() == 0) {
				output.retain(options, states[0], base[0] + likelihood[0], stats); retained++;
			}
			options.progress(iteration + 1, total, iteration < options.warmupIterations(), stats);
		}
		ChainResult.Status status = options.cancelled() ? ChainResult.Status.CANCELLED
				: ChainResult.Status.SUCCESS;
		ChainResult cold = output.result(states[0], base[0] + likelihood[0], retained,
				random, null, status);
		return new ParallelTemperingResult(cold, attempted, accepted);
	}
}
