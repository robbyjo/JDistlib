/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Deterministic multi-chain execution and checkpoint continuation helpers. */
public final class Chains {
	private Chains() {}

	public static ChainResult[] parallel(final Sampler sampler,
			final LogDensity target, double[][] initialStates,
			final SamplingOptions options, long baseSeed, int parallelism) {
		if (sampler == null || target == null || initialStates == null
				|| initialStates.length == 0 || options == null || parallelism < 1)
			throw new IllegalArgumentException("sampler, target, states, options and parallelism are required");
		final double[][] states = new double[initialStates.length][];
		for (int i = 0; i < states.length; i++) {
			if (initialStates[i] == null) throw new IllegalArgumentException("initial state must not be null");
			states[i] = initialStates[i].clone();
		}
		ExecutorService executor = Executors.newFixedThreadPool(
				Math.min(parallelism, states.length));
		try {
			List<Future<ChainResult>> futures = new ArrayList<Future<ChainResult>>();
			for (int chain = 0; chain < states.length; chain++) {
				final int index = chain;
				final long seed = mixSeed(baseSeed, chain);
				futures.add(executor.submit(new Callable<ChainResult>() {
					@Override public ChainResult call() {
						return sampler.sample(target, states[index], options,
								new MersenneTwister(seed));
					}
				}));
			}
			ChainResult[] results = new ChainResult[states.length];
			for (int i = 0; i < results.length; i++) results[i] = futures.get(i).get();
			return results;
		} catch (InterruptedException exception) {
			Thread.currentThread().interrupt();
			throw new IllegalStateException("parallel chains were interrupted", exception);
		} catch (ExecutionException exception) {
			throw new IllegalStateException("parallel chain failed", exception.getCause());
		} finally {
			executor.shutdownNow();
		}
	}

	/** Restarts from the state and cloned stream stored by an in-memory checkpoint. */
	public static ChainResult resume(Sampler sampler, LogDensity target,
			ChainCheckpoint checkpoint, SamplingOptions options) {
		if (sampler == null || target == null || checkpoint == null || options == null)
			throw new IllegalArgumentException("sampler, target, checkpoint and options are required");
		return sampler.sample(target, checkpoint.state(), options, checkpoint.random());
	}

	private static long mixSeed(long base, int chain) {
		long value = base + 0x9e3779b97f4a7c15L * (chain + 1L);
		value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
		value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
		return value ^ (value >>> 31);
	}
}
