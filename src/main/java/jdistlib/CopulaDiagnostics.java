/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Immutable classification of a proposed copula evaluation point. */
public final class CopulaDiagnostics {
	/** Location of the point relative to the unit hypercube. */
	public enum Classification { INTERIOR, BOUNDARY, INVALID }

	private final Classification classification;
	private final int lowerBoundaryCoordinates;
	private final int upperBoundaryCoordinates;
	private final boolean densityDefined;
	private final String message;

	private CopulaDiagnostics(Classification classification, int lower,
			int upper, boolean densityDefined, String message) {
		this.classification = classification;
		this.lowerBoundaryCoordinates = lower;
		this.upperBoundaryCoordinates = upper;
		this.densityDefined = densityDefined;
		this.message = message;
	}

	static CopulaDiagnostics inspect(double[] u, int dimension) {
		return inspect(u, dimension, false);
	}

	static CopulaDiagnostics inspect(double[] u, int dimension,
			boolean boundaryDensityDefined) {
		if (u == null || u.length != dimension) {
			return new CopulaDiagnostics(Classification.INVALID, 0, 0, false,
					"point dimension does not match the copula");
		}
		int lower = 0;
		int upper = 0;
		for (double value : u) {
			if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
				return new CopulaDiagnostics(Classification.INVALID, lower, upper, false,
						"coordinates must be finite values in [0, 1]");
			}
			if (value == 0.0) lower++;
			if (value == 1.0) upper++;
		}
		if (lower + upper == 0) {
			return new CopulaDiagnostics(Classification.INTERIOR, 0, 0, true,
					"CDF and density evaluation are well-defined");
		}
		return new CopulaDiagnostics(Classification.BOUNDARY, lower, upper,
				boundaryDensityDefined, boundaryDensityDefined
				? "CDF and density evaluation are well-defined"
				: "the CDF has its usual boundary value; a density limit may be "
				+ "singular or path-dependent");
	}

	public Classification getClassification() { return classification; }
	public int getLowerBoundaryCoordinates() { return lowerBoundaryCoordinates; }
	public int getUpperBoundaryCoordinates() { return upperBoundaryCoordinates; }
	public String getMessage() { return message; }
	public boolean isValid() { return classification != Classification.INVALID; }
	public boolean isInterior() { return classification == Classification.INTERIOR; }
	public boolean isBoundary() { return classification == Classification.BOUNDARY; }
	public boolean isDensityDefined() { return densityDefined; }
}
