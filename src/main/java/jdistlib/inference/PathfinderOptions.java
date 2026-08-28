/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Immutable controls for multi-path Pathfinder initialization and sampling. */
public final class PathfinderOptions {
	private final int paths, drawsPerPath, resampledDraws, maximumIterations, historySize;
	private final double tolerance;
	private PathfinderOptions(Builder builder) {
		paths = builder.paths; drawsPerPath = builder.drawsPerPath;
		resampledDraws = builder.resampledDraws; maximumIterations = builder.maximumIterations;
		historySize = builder.historySize; tolerance = builder.tolerance;
	}
	public static Builder builder() { return new Builder(); }
	public int paths() { return paths; }
	public int drawsPerPath() { return drawsPerPath; }
	public int resampledDraws() { return resampledDraws; }
	public int maximumIterations() { return maximumIterations; }
	public int historySize() { return historySize; }
	public double tolerance() { return tolerance; }
	public static final class Builder {
		private int paths = 4, drawsPerPath = 250, resampledDraws = 1000;
		private int maximumIterations = 1000, historySize = 8;
		private double tolerance = 1e-6;
		public Builder paths(int value) { paths = value; return this; }
		public Builder drawsPerPath(int value) { drawsPerPath = value; return this; }
		public Builder resampledDraws(int value) { resampledDraws = value; return this; }
		public Builder maximumIterations(int value) { maximumIterations = value; return this; }
		public Builder historySize(int value) { historySize = value; return this; }
		public Builder tolerance(double value) { tolerance = value; return this; }
		public PathfinderOptions build() {
			if (paths < 1 || drawsPerPath < 1 || resampledDraws < 1 || maximumIterations < 1
					|| historySize < 1 || !(tolerance > 0.0))
				throw new IllegalArgumentException("invalid Pathfinder options");
			return new PathfinderOptions(this);
		}
	}
}
