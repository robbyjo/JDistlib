/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.disttest.online;

/** Immutable record of one online-FDR test. */
public final class OnlineFdrDecision {
	private final long index;
	private final double pValue;
	private final double testLevel;
	private final boolean rejected;

	OnlineFdrDecision(long index, double pValue, double testLevel,
			boolean rejected) {
		this.index = index;
		this.pValue = pValue;
		this.testLevel = testLevel;
		this.rejected = rejected;
	}

	/** Returns the one-based arrival index. */
	public long getIndex() { return index; }
	/** Returns the submitted p-value. */
	public double getPValue() { return pValue; }
	/** Returns the level chosen before this p-value was observed. */
	public double getTestLevel() { return testLevel; }
	/** Returns whether the hypothesis was rejected. */
	public boolean isRejected() { return rejected; }
}
