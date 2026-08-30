/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable FP32 eigendecomposition of a real symmetric matrix. */
public final class FloatSymmetricEigenDecomposition {
	private final int dimension;
	private final float[] eigenvalues, eigenvectors;

	/** Creates a decomposition with ascending eigenvalues and eigenvectors in columns. */
	public FloatSymmetricEigenDecomposition(int dimension, float[] eigenvalues,
			float[] eigenvectors) {
		if (dimension < 1 || eigenvalues == null || eigenvalues.length != dimension
				|| eigenvectors == null || eigenvectors.length != dimension * dimension)
			throw new IllegalArgumentException("invalid FP32 symmetric eigendecomposition dimensions");
		this.dimension = dimension; this.eigenvalues = eigenvalues.clone();
		this.eigenvectors = eigenvectors.clone();
	}

	public int dimension() { return dimension; }
	/** Returns eigenvalues in ascending order. */
	public float[] eigenvalues() { return eigenvalues.clone(); }
	/** Returns a row-major orthogonal matrix whose columns are eigenvectors. */
	public float[] eigenvectors() { return eigenvectors.clone(); }
}
