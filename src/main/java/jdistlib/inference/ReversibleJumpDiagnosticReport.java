/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Model occupancy, movement, inclusion, conditional-parameter, and reliability diagnostics. */
public final class ReversibleJumpDiagnosticReport {
	private final long[] modelIds, modelVisits; private final double[] modelProbabilities, modelEss, modelRHat, modelMcse; private final long[][] transitions;
	private final String[] candidateNames; private final double[] inclusionProbabilities, inclusionEss, inclusionRHat, inclusionMcse;
	private final String[] moveNames; private final long[] moveAttempts, moveAccepts, invalidProposals;
	private final ReversibleJumpParameterSummary[] parameterSummaries; private final long modelChanges; private final int roundTrips;
	private final List<String> warnings;
	ReversibleJumpDiagnosticReport(long[] modelIds, long[] modelVisits, double[] modelProbabilities,
			double[] modelEss, double[] modelRHat, double[] modelMcse, long[][] transitions,
			String[] candidateNames, double[] inclusionProbabilities, double[] inclusionEss, double[] inclusionRHat,
			double[] inclusionMcse, String[] moveNames, long[] moveAttempts, long[] moveAccepts,
			long[] invalidProposals, ReversibleJumpParameterSummary[] parameterSummaries,
			long modelChanges, int roundTrips, List<String> warnings) {
		this.modelIds = modelIds; this.modelVisits = modelVisits; this.modelProbabilities = modelProbabilities;
		this.modelEss = modelEss; this.modelRHat = modelRHat; this.modelMcse = modelMcse; this.transitions = transitions;
		this.candidateNames = candidateNames; this.inclusionProbabilities = inclusionProbabilities; this.inclusionEss = inclusionEss;
		this.inclusionRHat = inclusionRHat; this.inclusionMcse = inclusionMcse; this.moveNames = moveNames;
		this.moveAttempts = moveAttempts; this.moveAccepts = moveAccepts; this.invalidProposals = invalidProposals;
		this.parameterSummaries = parameterSummaries; this.modelChanges = modelChanges; this.roundTrips = roundTrips;
		this.warnings = Collections.unmodifiableList(new ArrayList<String>(warnings));
	}
	public int modelCount() { return modelIds.length; }
	public long[] modelIds() { return modelIds.clone(); }
	public long[] modelVisits() { return modelVisits.clone(); }
	public double[] modelProbabilities() { return modelProbabilities.clone(); }
	public double[] modelEffectiveSampleSizes() { return modelEss.clone(); }
	public double[] modelRHats() { return modelRHat.clone(); }
	public double[] modelMcses() { return modelMcse.clone(); }
	public long[][] transitionCounts() { long[][] copy = new long[transitions.length][]; for (int i = 0; i < copy.length; i++) copy[i] = transitions[i].clone(); return copy; }
	public long modelChanges() { return modelChanges; }
	public int roundTrips() { return roundTrips; }
	public String[] candidateNames() { return candidateNames.clone(); }
	public double[] inclusionProbabilities() { return inclusionProbabilities.clone(); }
	public double[] inclusionEffectiveSampleSizes() { return inclusionEss.clone(); }
	public double[] inclusionRHats() { return inclusionRHat.clone(); }
	public double[] inclusionMcses() { return inclusionMcse.clone(); }
	public int moveCount() { return moveNames.length; }
	public String moveName(int index) { return moveNames[index]; }
	public long moveAttempts(int index) { return moveAttempts[index]; }
	public long moveAccepts(int index) { return moveAccepts[index]; }
	public long invalidProposals(int index) { return invalidProposals[index]; }
	public double moveAcceptanceRate(int index) { return moveAttempts[index] == 0L ? Double.NaN : moveAccepts[index] / (double) moveAttempts[index]; }
	public ReversibleJumpParameterSummary[] parameterSummaries() { return parameterSummaries.clone(); }
	public List<String> warnings() { return warnings; }
	public boolean reliable() { return warnings.isEmpty(); }
	public String toJson() {
		StringBuilder out = new StringBuilder("{\"schema\":\"jdistlib.rjmcmc-diagnostics/1\",\"models\":[");
		for (int i = 0; i < modelIds.length; i++) { if (i > 0) out.append(','); out.append("{\"id\":").append(modelIds[i])
				.append(",\"visits\":").append(modelVisits[i]).append(",\"probability\":").append(number(modelProbabilities[i]))
				.append(",\"ess\":").append(number(modelEss[i])).append(",\"rhat\":").append(number(modelRHat[i]))
				.append(",\"mcse\":").append(number(modelMcse[i])).append('}'); }
		out.append("],\"transitions\":["); boolean firstTransition = true;
		for (int from = 0; from < transitions.length; from++) for (int to = 0; to < transitions[from].length; to++) if (transitions[from][to] > 0L) {
			if (!firstTransition) out.append(','); firstTransition = false; out.append("{\"from\":").append(modelIds[from])
					.append(",\"to\":").append(modelIds[to]).append(",\"count\":").append(transitions[from][to]).append('}');
		}
		out.append("],\"inclusion\":[");
		for (int i = 0; i < candidateNames.length; i++) { if (i > 0) out.append(','); out.append("{\"candidate\":\"").append(escape(candidateNames[i]))
				.append("\",\"probability\":").append(number(inclusionProbabilities[i])).append(",\"ess\":").append(number(inclusionEss[i]))
				.append(",\"rhat\":").append(number(inclusionRHat[i])).append(",\"mcse\":").append(number(inclusionMcse[i])).append('}'); }
		out.append("],\"moves\":[");
		for (int i = 0; i < moveNames.length; i++) { if (i > 0) out.append(','); out.append("{\"name\":\"").append(escape(moveNames[i]))
				.append("\",\"attempts\":").append(moveAttempts[i]).append(",\"accepts\":").append(moveAccepts[i])
				.append(",\"invalid\":").append(invalidProposals[i]).append('}'); }
		out.append("],\"parameters\":[");
		for (int i = 0; i < parameterSummaries.length; i++) { if (i > 0) out.append(','); ReversibleJumpParameterSummary summary = parameterSummaries[i];
			out.append("{\"name\":\"").append(escape(summary.name())).append("\",\"draws\":").append(summary.draws())
					.append(",\"mean\":").append(number(summary.mean())).append(",\"sd\":").append(number(summary.standardDeviation())).append('}'); }
		out.append("],\"modelChanges\":").append(modelChanges).append(",\"roundTrips\":").append(roundTrips).append(",\"warnings\":[");
		for (int i = 0; i < warnings.size(); i++) { if (i > 0) out.append(','); out.append('"').append(escape(warnings.get(i))).append('"'); }
		return out.append("]}").toString();
	}
	private static String number(double value) { return Double.isFinite(value) ? Double.toString(value) : "null"; }
	private static String escape(String value) { return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r"); }
}
