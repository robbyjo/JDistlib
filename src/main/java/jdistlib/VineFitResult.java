/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

/** Result of sequential pair-family selection for a simplified vine. */
public final class VineFitResult {
	public enum Status { SUCCESS, INVALID_DATA, PAIR_FIT_FAILED }

	private final VineStructure structure;
	private final VineCopula copula;
	private final List<CopulaSelectionResult> pairSelections;
	private final double logLikelihood;
	private final int parameters;
	private final Status status;
	private final String message;

	VineFitResult(VineStructure structure, VineCopula copula,
			List<CopulaSelectionResult> pairSelections, double logLikelihood,
			int parameters, Status status, String message) {
		this.structure = structure;
		this.copula = copula;
		this.pairSelections = Collections.unmodifiableList(new ArrayList<>(pairSelections));
		this.logLikelihood = logLikelihood;
		this.parameters = parameters;
		this.status = status;
		this.message = message;
	}

	public VineStructure getStructure() { return structure; }
	public VineCopula getCopula() { return copula; }
	public List<CopulaSelectionResult> getPairSelections() { return pairSelections; }
	public double getLogLikelihood() { return logLikelihood; }
	public int getParameters() { return parameters; }
	public Status getStatus() { return status; }
	public String message() { return message; }
	public boolean isSuccess() { return status == Status.SUCCESS; }
}
