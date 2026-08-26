/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.disttest.online;

import java.util.ArrayList;
import java.util.List;

/**
 * Stateful constant-candidate-threshold SAFFRON online-FDR controller.
 *
 * <p>A p-value is a candidate when it is at most {@code lambda}. The supplied
 * finite gamma sequence is padded with zeros. FDR control requires independent
 * null p-values; under conditional super-uniformity the procedure controls
 * modified FDR.</p>
 *
 * @see <a href="https://proceedings.mlr.press/v80/ramdas18a.html">Ramdas et
 * al. (2018)</a>
 */
public final class Saffron implements OnlineFdrController {
	private final double alpha;
	private final double initialWealth;
	private final double lambda;
	private final double[] gamma;
	private final List<Long> rejectionTimes = new ArrayList<Long>();
	private final List<Long> candidatesAtRejection = new ArrayList<Long>();
	private long tests;
	private long candidates;

	public Saffron(double alpha, double initialWealth, double lambda,
			double[] gamma) {
		OnlineFdr.validateProbability(alpha, "alpha");
		if (alpha <= 0.0)
			throw new IllegalArgumentException("alpha must be positive");
		OnlineFdr.validateProbability(lambda, "lambda");
		if (lambda <= 0.0 || lambda >= 1.0)
			throw new IllegalArgumentException("lambda must be in (0, 1)");
		OnlineFdr.validateProbability(initialWealth, "initial wealth");
		if (initialWealth >= (1.0 - lambda) * alpha)
			throw new IllegalArgumentException(
					"initial wealth must be less than (1-lambda)*alpha");
		this.alpha = alpha;
		this.initialWealth = initialWealth;
		this.lambda = lambda;
		this.gamma = OnlineFdr.validateGamma(gamma);
	}

	@Override
	public synchronized OnlineFdrDecision test(double pValue) {
		OnlineFdr.validateProbability(pValue, "p-value");
		long time = tests + 1;
		double level = initialWealth * OnlineFdr.gamma(gamma,
				time - candidates);
		for (int i = 0; i < rejectionTimes.size(); i++) {
			long nonCandidatesSince = time - rejectionTimes.get(i)
					- (candidates - candidatesAtRejection.get(i));
			double reward = i == 0
					? (1.0 - lambda) * alpha - initialWealth
					: (1.0 - lambda) * alpha;
			level += reward * OnlineFdr.gamma(gamma, nonCandidatesSince);
		}
		level = Math.min(lambda, Math.max(0.0, level));
		boolean candidate = pValue <= lambda;
		boolean rejected = pValue <= level;
		tests = time;
		if (candidate) candidates++;
		if (rejected) {
			rejectionTimes.add(time);
			candidatesAtRejection.add(candidates);
		}
		return new OnlineFdrDecision(time, pValue, level, rejected);
	}

	/** Returns the number of candidate p-values seen so far. */
	public synchronized long getCandidateCount() { return candidates; }
	public double getCandidateThreshold() { return lambda; }
	@Override public synchronized long getTestCount() { return tests; }
	@Override public synchronized int getRejectionCount() {
		return rejectionTimes.size();
	}
	@Override public synchronized void reset() {
		tests = 0;
		candidates = 0;
		rejectionTimes.clear();
		candidatesAtRejection.clear();
	}
}
