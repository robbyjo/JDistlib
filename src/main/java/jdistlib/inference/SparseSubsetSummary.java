/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Online occupancy and conditional-coefficient summary stored in a sparse checkpoint. */
public final class SparseSubsetSummary {
	private final SparseSubsetTarget target; private final SparseSubsetCheckpoint checkpoint;
	public SparseSubsetSummary(SparseSubsetTarget target, SparseSubsetCheckpoint checkpoint) {
		if (target == null || checkpoint == null || checkpoint.inclusionCounts().length != target.candidateCount()
				|| checkpoint.modelSizeCounts().length != target.maximumActive() + 1) throw new IllegalArgumentException("matching target and sparse checkpoint required");
		this.target = target; this.checkpoint = checkpoint;
	}
	public long retainedDraws() { return checkpoint.retainedDraws(); }
	public double inclusionProbability(int candidate) { return retainedDraws() == 0L ? Double.NaN : checkpoint.inclusionCounts()[candidate] / (double) retainedDraws(); }
	public double conditionalCoefficientMean(int candidate) { long count = checkpoint.coefficientCounts()[candidate]; return count == 0L ? Double.NaN : checkpoint.coefficientSums()[candidate] / count; }
	public double conditionalCoefficientStandardDeviation(int candidate) {
		long count = checkpoint.coefficientCounts()[candidate]; if (count < 2L) return Double.NaN;
		double sum = checkpoint.coefficientSums()[candidate], squares = checkpoint.coefficientSquareSums()[candidate];
		return Math.sqrt(Math.max(0.0, (squares - sum * sum / count) / (count - 1.0)));
	}
	public double modelSizeProbability(int size) { return retainedDraws() == 0L ? Double.NaN : checkpoint.modelSizeCounts()[size] / (double) retainedDraws(); }
	public double commonMean(int parameter) { return retainedDraws() == 0L ? Double.NaN : checkpoint.commonSums()[parameter] / retainedDraws(); }
	public double commonStandardDeviation(int parameter) {
		long count = retainedDraws(); if (count < 2L) return Double.NaN; double sum = checkpoint.commonSums()[parameter], squares = checkpoint.commonSquareSums()[parameter];
		return Math.sqrt(Math.max(0.0, (squares - sum * sum / count) / (count - 1.0)));
	}
	public String toJson(double minimumInclusionProbability) {
		if (!(minimumInclusionProbability >= 0.0 && minimumInclusionProbability <= 1.0)) throw new IllegalArgumentException("valid inclusion threshold required");
		StringBuilder out = new StringBuilder("{\"schema\":\"jdistlib.sparse-rjmcmc-summary/1\",\"retainedDraws\":").append(retainedDraws()).append(",\"modelSize\":[");
		for (int size = 0; size <= target.maximumActive(); size++) { if (size > 0) out.append(','); out.append("{\"size\":").append(size).append(",\"probability\":").append(number(modelSizeProbability(size))).append('}'); }
		out.append("],\"candidates\":["); boolean first = true;
		for (int candidate = 0; candidate < target.candidateCount(); candidate++) { double probability = inclusionProbability(candidate); if (!(probability >= minimumInclusionProbability)) continue;
			if (!first) out.append(','); first = false; out.append("{\"candidate\":\"").append(escape(target.candidateName(candidate))).append("\",\"index\":").append(candidate)
					.append(",\"probability\":").append(number(probability)).append(",\"conditionalMean\":").append(number(conditionalCoefficientMean(candidate)))
					.append(",\"conditionalSd\":").append(number(conditionalCoefficientStandardDeviation(candidate))).append('}'); }
		return out.append("]}").toString();
	}
	private static String number(double value) { return Double.isFinite(value) ? Double.toString(value) : "null"; }
	private static String escape(String value) { return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r"); }
}
