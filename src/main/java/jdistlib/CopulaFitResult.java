/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Result of fitting one copula family to pseudo-observations. */
public final class CopulaFitResult {
	public enum Status {
		SUCCESS,
		INVALID_DATA,
		INCOMPATIBLE_DEPENDENCE,
		NUMERICAL_FAILURE
	}

	private final CopulaFamily family;
	private final Copula copula;
	private final double logLikelihood;
	private final int observations;
	private final int parameters;
	private final Status status;
	private final String message;

	CopulaFitResult(CopulaFamily family, Copula copula, double logLikelihood,
			int observations, int parameters, Status status, String message) {
		this.family = family;
		this.copula = copula;
		this.logLikelihood = logLikelihood;
		this.observations = observations;
		this.parameters = parameters;
		this.status = status;
		this.message = message;
	}

	public CopulaFamily getFamily() { return family; }
	public Copula getCopula() { return copula; }
	public double getLogLikelihood() { return logLikelihood; }
	public int getObservations() { return observations; }
	public int getParameters() { return parameters; }
	public Status getStatus() { return status; }
	public String message() { return message; }
	public boolean isSuccess() { return status == Status.SUCCESS; }
	public double aic() {
		return isSuccess() ? 2.0 * parameters - 2.0 * logLikelihood : Double.POSITIVE_INFINITY;
	}
	public double bic() {
		return isSuccess() ? Math.log(observations) * parameters
				- 2.0 * logLikelihood : Double.POSITIVE_INFINITY;
	}
}
