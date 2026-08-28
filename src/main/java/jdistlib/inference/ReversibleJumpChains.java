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

/** Deterministic parallel execution with one independently adaptive RJ sampler per chain. */
public final class ReversibleJumpChains {
	private ReversibleJumpChains() {}
	public static ReversibleJumpResult[] parallel(final ReversibleJumpSamplerFactory factory,
			final ReversibleJumpTarget target, ReversibleJumpState[] initialStates,
			final ReversibleJumpSamplingOptions options, long baseSeed, int parallelism) {
		if (factory == null || target == null || initialStates == null || initialStates.length == 0
				|| options == null || parallelism < 1) throw new IllegalArgumentException("factory, target, states, options, and parallelism required");
		final ReversibleJumpState[] states = initialStates.clone();
		for (ReversibleJumpState state : states) if (state == null) throw new IllegalArgumentException("initial states must not contain null");
		ExecutorService executor = Executors.newFixedThreadPool(Math.min(parallelism, states.length));
		try {
			List<Future<ReversibleJumpResult>> futures = new ArrayList<Future<ReversibleJumpResult>>();
			for (int chain = 0; chain < states.length; chain++) {
				final int index = chain; final long seed = mixSeed(baseSeed, chain);
				futures.add(executor.submit(new Callable<ReversibleJumpResult>() {
					@Override public ReversibleJumpResult call() {
						ReversibleJumpSampler sampler = factory.create(); if (sampler == null) throw new IllegalStateException("RJ sampler factory returned null");
						return sampler.sample(target, states[index], options, new MersenneTwister(seed));
					}
				}));
			}
			ReversibleJumpResult[] result = new ReversibleJumpResult[states.length];
			for (int i = 0; i < result.length; i++) result[i] = futures.get(i).get(); return result;
		} catch (InterruptedException exception) {
			Thread.currentThread().interrupt(); throw new IllegalStateException("parallel RJ chains were interrupted", exception);
		} catch (ExecutionException exception) { throw new IllegalStateException("parallel RJ chain failed", exception.getCause()); }
		finally { executor.shutdownNow(); }
	}
	private static long mixSeed(long base, int chain) {
		long value = base + 0x9e3779b97f4a7c15L * (chain + 1L);
		value = (value ^ value >>> 30) * 0xbf58476d1ce4e5b9L; value = (value ^ value >>> 27) * 0x94d049bb133111ebL;
		return value ^ value >>> 31;
	}
}
