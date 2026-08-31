/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** FP32 pivoted {@code P*A*P' = L*D*L'} factorization with 1x1/2x2 D blocks. */
public final class FloatSymmetricIndefiniteFactor {
	private final SymmetricIndefiniteFactor delegate;
	public FloatSymmetricIndefiniteFactor(int dimension, float[] lower, float[] diagonal,
			int[] permutation, int[] blockSizes) {
		if (lower == null || diagonal == null) throw new IllegalArgumentException("FP32 LDL factors are required");
		double[] l = new double[lower.length], d = new double[diagonal.length];
		for (int i = 0; i < l.length; i++) l[i] = lower[i];
		for (int i = 0; i < d.length; i++) d[i] = diagonal[i];
		delegate = new SymmetricIndefiniteFactor(dimension, l, d, permutation, blockSizes);
	}
	public int dimension() { return delegate.dimension(); }
	public float[] lower() { double[] source = delegate.lower(); float[] result = new float[source.length];
		for (int i = 0; i < source.length; i++) result[i] = (float) source[i]; return result; }
	public float[] diagonalBlocks() { double[] source = delegate.diagonalBlocks(); float[] result = new float[source.length];
		for (int i = 0; i < source.length; i++) result[i] = (float) source[i]; return result; }
	public int[] permutation() { return delegate.permutation(); }
	public int[] blockSizes() { return delegate.blockSizes(); }
	public int determinantSign() { return delegate.determinantSign(); }
	public float logAbsDeterminant() { return (float) delegate.logAbsDeterminant(); }
	public float[] solve(float[] right) { return solve(right, 1); }
	public float[] solve(float[] right, int columns) {
		if (right == null) throw new IllegalArgumentException("FP32 LDL right side is required");
		double[] input = new double[right.length]; for (int i = 0; i < right.length; i++) input[i] = right[i];
		double[] output = delegate.solve(input, columns); float[] result = new float[output.length];
		for (int i = 0; i < output.length; i++) result[i] = (float) output[i]; return result;
	}
}
