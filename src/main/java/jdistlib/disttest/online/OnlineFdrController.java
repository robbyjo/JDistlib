/*
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.disttest.online;

/** A stateful controller for hypotheses arriving in a fixed sequential order. */
public interface OnlineFdrController {
	/** Tests the next p-value and advances the controller exactly once. */
	OnlineFdrDecision test(double pValue);
	/** Returns the number of p-values processed. */
	long getTestCount();
	/** Returns the number of rejections so far. */
	int getRejectionCount();
	/** Restores the controller to its newly constructed state. */
	void reset();
}
