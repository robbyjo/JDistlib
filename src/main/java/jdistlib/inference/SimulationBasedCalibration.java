/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Seeded simulation-based calibration rank utility. */
public final class SimulationBasedCalibration {
	public interface Simulation {
		LogDensity target();
		double[] initialState();
		double[] trueParameters();
	}
	public interface Generator { Simulation generate(RandomEngine random); }
	private SimulationBasedCalibration() {}
	public static int[][] ranks(Generator generator, Sampler sampler,
			SamplingOptions options, int replications, long seed) {
		if (generator == null || sampler == null || options == null || replications < 1)
			throw new IllegalArgumentException("invalid SBC arguments");
		RandomEngine random = new MersenneTwister(seed);
		int[][] ranks = null;
		for (int replication = 0; replication < replications; replication++) {
			Simulation simulation = generator.generate(random);
			double[] truth = simulation.trueParameters();
			ChainResult chain = sampler.sample(simulation.target(), simulation.initialState(),
					options, random);
			if (ranks == null) ranks = new int[replications][truth.length];
			if (chain.dimension() != truth.length)
				throw new IllegalArgumentException("SBC truth and chain dimensions differ");
			for (int parameter = 0; parameter < truth.length; parameter++) {
				int rank = 0;
				for (int draw = 0; draw < chain.size(); draw++)
					if (chain.valueAt(draw, parameter) < truth[parameter]) rank++;
				ranks[replication][parameter] = rank;
			}
		}
		return ranks;
	}
}
