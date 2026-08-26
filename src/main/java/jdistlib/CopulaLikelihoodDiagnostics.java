/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/** Row-level log-density and unit-cube boundary diagnostics for a copula model. */
public final class CopulaLikelihoodDiagnostics {
	public enum Status { SUCCESS, INVALID_DATA, NONFINITE_CONTRIBUTION }

	private final double logLikelihood;
	private final double[] logContributions;
	private final double[] boundaryDistances;
	private final int finiteContributions;
	private final int firstProblemIndex;
	private final double minimumLogContribution;
	private final double maximumLogContribution;
	private final double meanLogContribution;
	private final double minimumBoundaryDistance;
	private final Status status;
	private final String message;

	private CopulaLikelihoodDiagnostics(double logLikelihood,
			double[] logContributions, double[] boundaryDistances,
			int finiteContributions, int firstProblemIndex,
			double minimumLogContribution, double maximumLogContribution,
			double meanLogContribution, double minimumBoundaryDistance,
			Status status, String message) {
		this.logLikelihood = logLikelihood;
		this.logContributions = logContributions.clone();
		this.boundaryDistances = boundaryDistances.clone();
		this.finiteContributions = finiteContributions;
		this.firstProblemIndex = firstProblemIndex;
		this.minimumLogContribution = minimumLogContribution;
		this.maximumLogContribution = maximumLogContribution;
		this.meanLogContribution = meanLogContribution;
		this.minimumBoundaryDistance = minimumBoundaryDistance;
		this.status = status;
		this.message = message;
	}

	/** Evaluates every row without modifying the supplied observations. */
	public static CopulaLikelihoodDiagnostics assess(Copula copula,
			double[][] uniforms) {
		if (copula == null || uniforms == null || uniforms.length == 0)
			return invalid(uniforms == null ? 0 : uniforms.length,
					"copula and at least one observation are required");
		double[] contributions = new double[uniforms.length];
		double[] distances = new double[uniforms.length];
		double sum = 0.0;
		double correction = 0.0;
		double minimum = Double.POSITIVE_INFINITY;
		double maximum = Double.NEGATIVE_INFINITY;
		double minimumDistance = Double.POSITIVE_INFINITY;
		int finite = 0;
		int firstProblem = -1;
		for (int row = 0; row < uniforms.length; row++) {
			double[] point = uniforms[row];
			if (point == null || point.length != copula.dimension())
				return invalid(uniforms.length,
						"observation " + row + " does not match the copula dimension");
			double distance = 0.5;
			for (double value : point) {
				if (!Double.isFinite(value) || value < 0.0 || value > 1.0)
					return invalid(uniforms.length,
							"observation " + row + " is outside the closed unit cube");
				distance = Math.min(distance, Math.min(value, 1.0 - value));
			}
			distances[row] = distance;
			minimumDistance = Math.min(minimumDistance, distance);
			double contribution = copula.logDensity(point);
			contributions[row] = contribution;
			if (Double.isFinite(contribution)) {
				finite++;
				minimum = Math.min(minimum, contribution);
				maximum = Math.max(maximum, contribution);
				double adjusted = contribution - correction;
				double next = sum + adjusted;
				correction = (next - sum) - adjusted;
				sum = next;
			} else if (firstProblem < 0) {
				firstProblem = row;
			}
		}
		if (firstProblem >= 0) {
			return new CopulaLikelihoodDiagnostics(Double.NaN, contributions,
					distances, finite, firstProblem,
					finite == 0 ? Double.NaN : minimum,
					finite == 0 ? Double.NaN : maximum,
					finite == 0 ? Double.NaN : sum / finite,
					minimumDistance, Status.NONFINITE_CONTRIBUTION,
					"copula log density is non-finite at observation " + firstProblem);
		}
		return new CopulaLikelihoodDiagnostics(sum, contributions, distances,
				finite, -1, minimum, maximum, sum / finite, minimumDistance,
				Status.SUCCESS, "all copula log-density contributions are finite");
	}

	static CopulaLikelihoodDiagnostics invalid(int observations, String message) {
		double[] contributions = new double[Math.max(0, observations)];
		double[] distances = new double[contributions.length];
		for (int i = 0; i < contributions.length; i++) {
			contributions[i] = Double.NaN;
			distances[i] = Double.NaN;
		}
		return new CopulaLikelihoodDiagnostics(Double.NaN, contributions,
				distances, 0, -1, Double.NaN, Double.NaN, Double.NaN,
				Double.NaN, Status.INVALID_DATA, message);
	}

	public double getLogLikelihood() { return logLikelihood; }
	public int getObservations() { return logContributions.length; }
	public int getFiniteContributions() { return finiteContributions; }
	public int getNonFiniteContributions() {
		return logContributions.length - finiteContributions;
	}
	public int getFirstProblemIndex() { return firstProblemIndex; }
	public double getMinimumLogContribution() { return minimumLogContribution; }
	public double getMaximumLogContribution() { return maximumLogContribution; }
	public double getMeanLogContribution() { return meanLogContribution; }
	public double getMinimumBoundaryDistance() { return minimumBoundaryDistance; }
	public double[] getLogContributions() { return logContributions.clone(); }
	public double[] getBoundaryDistances() { return boundaryDistances.clone(); }
	public int countNearBoundary(double threshold) {
		if (!Double.isFinite(threshold) || threshold < 0.0 || threshold > 0.5)
			throw new IllegalArgumentException("boundary threshold must be in [0, 0.5]");
		int count = 0;
		for (double distance : boundaryDistances)
			if (Double.isFinite(distance) && distance <= threshold) count++;
		return count;
	}
	public Status getStatus() { return status; }
	public String getMessage() { return message; }
	public String message() { return message; }
	public boolean isSuccess() { return status == Status.SUCCESS; }
}
