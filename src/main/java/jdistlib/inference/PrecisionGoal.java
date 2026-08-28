/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.function.ToDoubleFunction;

/** Stopping goal for one posterior coordinate, guarded by minimum draws and chunk count. */
public final class PrecisionGoal {
	private final int coordinate, minimumDraws, maximumChunks; private final double absoluteMcse, relativeMcse;
	private final ToDoubleFunction<double[]> quantity;
	private PrecisionGoal(Builder builder) { coordinate = builder.coordinate; minimumDraws = builder.minimumDraws;
		maximumChunks = builder.maximumChunks; absoluteMcse = builder.absoluteMcse; relativeMcse = builder.relativeMcse; quantity = builder.quantity; }
	public static Builder builder(int coordinate) { return new Builder(coordinate); }
	public static Builder builder(ToDoubleFunction<double[]> quantity) { return new Builder(quantity); }
	public int coordinate() { return coordinate; } public int minimumDraws() { return minimumDraws; }
	public int maximumChunks() { return maximumChunks; } public double absoluteMcse() { return absoluteMcse; }
	public double relativeMcse() { return relativeMcse; }
	double[] evaluate(ChainResult chain) { double[] result = new double[chain.size()]; for (int draw = 0; draw < result.length; draw++)
		result[draw] = quantity == null ? chain.valueAt(draw, coordinate) : quantity.applyAsDouble(chain.sample(draw)); return result; }
	public static final class Builder {
		private final int coordinate; private final ToDoubleFunction<double[]> quantity; private int minimumDraws = 1000, maximumChunks = 20;
		private double absoluteMcse = Double.NaN, relativeMcse = 0.01;
		private Builder(int coordinate) { this.coordinate = coordinate; this.quantity = null; }
		private Builder(ToDoubleFunction<double[]> quantity) { if (quantity == null) throw new IllegalArgumentException("quantity is required"); this.coordinate = -1; this.quantity = quantity; }
		public Builder minimumDraws(int value) { minimumDraws = value; return this; }
		public Builder maximumChunks(int value) { maximumChunks = value; return this; }
		public Builder absoluteMcse(double value) { absoluteMcse = value; return this; }
		public Builder relativeMcse(double value) { relativeMcse = value; return this; }
		public PrecisionGoal build() { if ((coordinate < 0 && quantity == null) || minimumDraws < 4 || maximumChunks < 1
				|| (Double.isNaN(absoluteMcse) && Double.isNaN(relativeMcse))
				|| (!Double.isNaN(absoluteMcse) && !(absoluteMcse > 0.0))
				|| (!Double.isNaN(relativeMcse) && !(relativeMcse > 0.0))) throw new IllegalArgumentException("invalid precision goal"); return new PrecisionGoal(this); }
	}
}
