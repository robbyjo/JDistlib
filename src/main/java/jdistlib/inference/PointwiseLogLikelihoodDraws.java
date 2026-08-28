/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Pointwise log likelihoods with retained chain boundaries and observation metadata. */
public final class PointwiseLogLikelihoodDraws {
	private final ObservationMetadata metadata;
	private final double[][] values;
	private final int[] chainStarts;

	public PointwiseLogLikelihoodDraws(ObservationMetadata metadata, double[][] values,
			int[] chainStarts) {
		if (metadata == null || values == null || chainStarts == null || chainStarts.length < 2
				|| chainStarts[0] != 0 || chainStarts[chainStarts.length - 1] != values.length)
			throw new IllegalArgumentException("metadata, draws, and valid chain boundaries are required");
		this.metadata = metadata; this.values = new double[values.length][];
		for (int draw = 0; draw < values.length; draw++) {
			if (values[draw] == null || values[draw].length != metadata.size())
				throw new IllegalArgumentException("pointwise draw width does not match observations");
			this.values[draw] = values[draw].clone();
		}
		this.chainStarts = chainStarts.clone();
		for (int i = 1; i < this.chainStarts.length; i++)
			if (this.chainStarts[i] < this.chainStarts[i - 1])
				throw new IllegalArgumentException("chain boundaries must be ordered");
	}

	public static PointwiseLogLikelihoodDraws extract(PointwiseLogLikelihood model,
			ChainResult... chains) {
		if (model == null || chains == null || chains.length == 0)
			throw new IllegalArgumentException("a model and at least one chain are required");
		int draws = 0;
		for (ChainResult chain : chains) {
			if (chain == null) throw new IllegalArgumentException("chains must not contain null");
			draws += chain.size();
		}
		double[][] values = new double[draws][]; int[] starts = new int[chains.length + 1];
		int offset = 0;
		for (int chain = 0; chain < chains.length; chain++) {
			starts[chain] = offset;
			for (int draw = 0; draw < chains[chain].size(); draw++)
				values[offset++] = model.pointwiseLogLikelihood(chains[chain].sample(draw));
		}
		starts[chains.length] = offset;
		return new PointwiseLogLikelihoodDraws(model.observationMetadata(), values, starts);
	}

	public ObservationMetadata metadata() { return metadata; }
	public int draws() { return values.length; }
	public int observations() { return metadata.size(); }
	public int chains() { return chainStarts.length - 1; }
	public double valueAt(int draw, int observation) { return values[draw][observation]; }
	public double[][] values() {
		double[][] copy = new double[values.length][];
		for (int i = 0; i < values.length; i++) copy[i] = values[i].clone();
		return copy;
	}
	public int[] chainStarts() { return chainStarts.clone(); }
}
