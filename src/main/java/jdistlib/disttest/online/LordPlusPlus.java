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
 * Stateful LORD++ online-FDR controller for a prespecified hypothesis order.
 *
 * <p>The supplied finite gamma sequence is interpreted as an infinite sequence
 * padded with zeros. It must be nonnegative, nonincreasing, and sum to at most
 * one. The standard FDR guarantee requires independent null p-values and a
 * monotone testing rule.</p>
 *
 * @see <a href="https://papers.nips.cc/paper_files/paper/2017/hash/7f018eb7b301a66658931cb8a93fd6e8-Abstract.html">Ramdas et al. (2017)</a>
 */
public final class LordPlusPlus implements OnlineFdrController {
	private final double alpha;
	private final double initialWealth;
	private final double[] gamma;
	private final List<Long> rejectionTimes = new ArrayList<Long>();
	private long tests;

	public LordPlusPlus(double alpha, double initialWealth, double[] gamma) {
		OnlineFdr.validateProbability(alpha, "alpha");
		if (alpha <= 0.0)
			throw new IllegalArgumentException("alpha must be positive");
		OnlineFdr.validateProbability(initialWealth, "initial wealth");
		if (initialWealth > alpha)
			throw new IllegalArgumentException(
					"initial wealth must not exceed alpha");
		this.alpha = alpha;
		this.initialWealth = initialWealth;
		this.gamma = OnlineFdr.validateGamma(gamma);
	}

	@Override
	public synchronized OnlineFdrDecision test(double pValue) {
		OnlineFdr.validateProbability(pValue, "p-value");
		long time = tests + 1;
		double level = initialWealth * OnlineFdr.gamma(gamma, time);
		for (int i = 0; i < rejectionTimes.size(); i++) {
			double reward = i == 0 ? alpha - initialWealth : alpha;
			level += reward * OnlineFdr.gamma(gamma,
					time - rejectionTimes.get(i));
		}
		level = Math.min(alpha, Math.max(0.0, level));
		boolean rejected = pValue <= level;
		tests = time;
		if (rejected) rejectionTimes.add(time);
		return new OnlineFdrDecision(time, pValue, level, rejected);
	}

	@Override public synchronized long getTestCount() { return tests; }
	@Override public synchronized int getRejectionCount() {
		return rejectionTimes.size();
	}
	@Override public synchronized void reset() {
		tests = 0;
		rejectionTimes.clear();
	}
}
