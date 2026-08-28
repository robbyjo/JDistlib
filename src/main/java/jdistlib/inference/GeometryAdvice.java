/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Coordinate associated with divergences and a scale/reparameterization suggestion. */
public final class GeometryAdvice {
	private final int coordinate; private final double standardizedSeparation; private final String suggestion;
	GeometryAdvice(int coordinate, double standardizedSeparation, String suggestion) { this.coordinate = coordinate; this.standardizedSeparation = standardizedSeparation; this.suggestion = suggestion; }
	public int coordinate() { return coordinate; } public double standardizedSeparation() { return standardizedSeparation; } public String suggestion() { return suggestion; }
}
