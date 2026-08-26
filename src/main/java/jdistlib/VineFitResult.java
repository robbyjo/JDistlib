/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

/** Result of sequential pair-family selection for a simplified vine. */
public final class VineFitResult {
	public enum Status { SUCCESS, INVALID_DATA, PAIR_FIT_FAILED, NUMERICAL_FAILURE }

	private final VineStructure structure;
	private final VineCopula copula;
	private final List<CopulaSelectionResult> pairSelections;
	private final double logLikelihood;
	private final int parameters;
	private final int observations;
	private final CopulaLikelihoodDiagnostics diagnostics;
	private final Status status;
	private final String message;

	VineFitResult(VineStructure structure, VineCopula copula,
			List<CopulaSelectionResult> pairSelections, double logLikelihood,
			int parameters, int observations, Status status, String message,
			CopulaLikelihoodDiagnostics diagnostics) {
		this.structure = structure;
		this.copula = copula;
		this.pairSelections = Collections.unmodifiableList(new ArrayList<>(pairSelections));
		this.logLikelihood = logLikelihood;
		this.parameters = parameters;
		this.observations = observations;
		this.diagnostics = diagnostics;
		this.status = status;
		this.message = message;
	}

	public VineStructure getStructure() { return structure; }
	public VineCopula getCopula() { return copula; }
	public List<CopulaSelectionResult> getPairSelections() { return pairSelections; }
	public double getLogLikelihood() { return logLikelihood; }
	public int getParameters() { return parameters; }
	public int getObservations() { return observations; }
	public CopulaLikelihoodDiagnostics getDiagnostics() { return diagnostics; }
	public Status getStatus() { return status; }
	public String getMessage() { return message; }
	public String message() { return message; }
	public boolean isSuccess() { return status == Status.SUCCESS; }
	public double aic() {
		return isSuccess() ? 2.0 * parameters - 2.0 * logLikelihood
				: Double.POSITIVE_INFINITY;
	}
	public double bic() {
		return isSuccess() ? Math.log(observations) * parameters
				- 2.0 * logLikelihood : Double.POSITIVE_INFINITY;
	}
}
