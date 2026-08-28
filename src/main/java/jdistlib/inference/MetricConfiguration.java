/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable Euclidean metric selection for HMC-family samplers. */
public final class MetricConfiguration {
	public enum Type { UNIT, DIAGONAL, DENSE, BLOCK_DIAGONAL, LOW_RANK_DIAGONAL }
	private final Type type;
	private final int[][] blocks;
	private final int lowRank;
	private final double[][] initialInverseMass;

	private MetricConfiguration(Type type, int[][] blocks, int lowRank,
			double[][] initialInverseMass) {
		if (type == null || lowRank < 0) throw new IllegalArgumentException("invalid metric");
		this.type = type; this.blocks = copy(blocks); this.lowRank = lowRank;
		this.initialInverseMass = copy(initialInverseMass);
	}
	public static MetricConfiguration unit() { return new MetricConfiguration(Type.UNIT, null, 0, null); }
	public static MetricConfiguration diagonal() { return new MetricConfiguration(Type.DIAGONAL, null, 0, null); }
	public static MetricConfiguration dense() { return new MetricConfiguration(Type.DENSE, null, 0, null); }
	public static MetricConfiguration blockDiagonal(int[]... blocks) {
		if (blocks == null || blocks.length == 0) throw new IllegalArgumentException("blocks required");
		return new MetricConfiguration(Type.BLOCK_DIAGONAL, blocks, 0, null);
	}
	public static MetricConfiguration lowRankDiagonal(int rank) {
		if (rank < 1) throw new IllegalArgumentException("positive rank required");
		return new MetricConfiguration(Type.LOW_RANK_DIAGONAL, null, rank, null);
	}
	/** Dense, caller-supplied inverse mass matrix; disable adaptation to keep it fixed. */
	public static MetricConfiguration supplied(double[][] inverseMassMatrix) {
		if (inverseMassMatrix == null) throw new IllegalArgumentException("inverse mass matrix required");
		return new MetricConfiguration(Type.DENSE, null, 0, inverseMassMatrix);
	}
	public MetricConfiguration withInitialInverseMassMatrix(double[][] matrix) {
		return new MetricConfiguration(type, blocks, lowRank, matrix);
	}
	public Type type() { return type; }
	public int[][] blocks() { return copy(blocks); }
	public int lowRank() { return lowRank; }
	public double[][] initialInverseMassMatrix() { return copy(initialInverseMass); }
	private static int[][] copy(int[][] values) {
		if (values == null) return null;
		int[][] result = new int[values.length][];
		for (int i = 0; i < values.length; i++) result[i] = values[i].clone();
		return result;
	}
	private static double[][] copy(double[][] values) {
		if (values == null) return null;
		double[][] result = new double[values.length][];
		for (int i = 0; i < values.length; i++) result[i] = values[i].clone();
		return result;
	}
}
