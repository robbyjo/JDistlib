/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP64 eigendecomposition of a real symmetric matrix. */
public final class SymmetricEigenDecomposition {
	private final int dimension;
	private final double[] eigenvalues, eigenvectors;

	/** Creates a decomposition with ascending eigenvalues and eigenvectors in columns. */
	public SymmetricEigenDecomposition(int dimension, double[] eigenvalues,
			double[] eigenvectors) {
		if (dimension < 1 || eigenvalues == null || eigenvalues.length != dimension
				|| eigenvectors == null || eigenvectors.length != dimension * dimension)
			throw new IllegalArgumentException("invalid symmetric eigendecomposition dimensions");
		this.dimension = dimension; this.eigenvalues = eigenvalues.clone();
		this.eigenvectors = eigenvectors.clone();
	}

	public int dimension() { return dimension; }
	/** Returns eigenvalues in ascending order. */
	public double[] eigenvalues() { return eigenvalues.clone(); }
	/** Returns a row-major orthogonal matrix whose columns are eigenvectors. */
	public double[] eigenvectors() { return eigenvectors.clone(); }
}
