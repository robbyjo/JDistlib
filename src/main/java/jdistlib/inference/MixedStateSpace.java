/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Coordinate-by-coordinate support declaration for hybrid MCMC. */
public final class MixedStateSpace {
	private final CoordinateSupport[] supports;
	public MixedStateSpace(CoordinateSupport... supports) {
		if (supports == null || supports.length == 0) throw new IllegalArgumentException("at least one support required");
		this.supports = supports.clone();
		for (CoordinateSupport support : this.supports) if (support == null) throw new IllegalArgumentException("supports must not contain null");
	}
	public int dimension() { return supports.length; }
	public CoordinateSupport support(int coordinate) { return supports[coordinate]; }
	public CoordinateSupport[] supports() { return supports.clone(); }
	public boolean contains(double[] state) {
		if (state == null || state.length != supports.length) return false;
		for (int i = 0; i < supports.length; i++) if (!supports[i].contains(state[i])) return false;
		return true;
	}
	public int[] continuousCoordinates() { return coordinates(false); }
	public int[] discreteCoordinates() { return coordinates(true); }
	private int[] coordinates(boolean discrete) {
		int count = 0; for (CoordinateSupport support : supports) if (support.discrete() == discrete) count++;
		int[] result = new int[count]; int offset = 0;
		for (int i = 0; i < supports.length; i++) if (supports[i].discrete() == discrete) result[offset++] = i;
		return result;
	}
}
