/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

/** Joint distribution with continuous, discrete, or mixed scalar marginals. */
public final class MixedCopulaDistribution {
	private final Copula copula;
	private final CopulaMarginal[] marginals;
	private final CopulaMeasureOptions options;

	public MixedCopulaDistribution(Copula copula, CopulaMarginal... marginals) {
		this(copula, new CopulaMeasureOptions(), marginals);
	}

	public MixedCopulaDistribution(Copula copula, CopulaMeasureOptions options,
			CopulaMarginal... marginals) {
		if (copula == null || options == null)
			throw new IllegalArgumentException("copula and options must not be null");
		if (marginals == null || marginals.length != copula.dimension())
			throw new IllegalArgumentException("one marginal is required per copula coordinate");
		for (CopulaMarginal marginal : marginals)
			if (marginal == null) throw new IllegalArgumentException("marginals must not contain null");
		this.copula = copula;
		this.options = options;
		this.marginals = marginals.clone();
	}

	public int dimension() { return marginals.length; }
	public Copula getCopula() { return copula; }
	public CopulaMeasureOptions getOptions() { return options; }
	public CopulaMarginal getMarginal(int coordinate) {
		if (coordinate < 0 || coordinate >= dimension())
			throw new IndexOutOfBoundsException("marginal coordinate out of range");
		return marginals[coordinate];
	}

	public double cumulative(double[] x) {
		if (!validObservation(x)) return Double.NaN;
		double[] u = new double[dimension()];
		for (int i = 0; i < u.length; i++) u[i] = marginals[i].cumulative(x[i]);
		return copula.cumulative(u);
	}

	/** Evaluates density, probability mass, or mixed product-measure density. */
	public CopulaMeasureResult measure(double[] x) {
		if (!validObservation(x)) return invalid("observation dimension does not match");
		int continuous = 0;
		for (CopulaMarginal marginal : marginals) if (marginal.isContinuous()) continuous++;
		if (continuous == dimension()) return continuousMeasure(x);
		long corners = 1L << Math.min(dimension(), 62);
		if (dimension() >= 63 || corners > options.getMaxCdfEvaluations()) {
			return new CopulaMeasureResult(Double.NaN, Double.NaN, 0,
					CopulaMeasureResult.Status.EVALUATION_BUDGET_EXCEEDED,
					"mixed rectangle requires more CDF corners than the configured budget");
		}
		if (continuous == 0) return discreteMeasure(x);
		return mixedMeasure(x, continuous);
	}

	public double logLikelihood(double[][] observations) {
		if (observations == null) return Double.NaN;
		double result = 0.0;
		for (double[] observation : observations) {
			CopulaMeasureResult contribution = measure(observation);
			if (!contribution.hasEstimate()) return Double.NaN;
			result += contribution.logValue;
		}
		return result;
	}

	public double[] random(RandomEngine random) {
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[] result = copula.random(random);
		for (int i = 0; i < result.length; i++) result[i] = marginals[i].quantile(result[i]);
		return result;
	}

	public double[] random(long seed) { return random(new MersenneTwister(seed)); }
	public double[][] random(int count, RandomEngine random) {
		if (count < 0) throw new IllegalArgumentException("sample size must be nonnegative");
		if (random == null) throw new IllegalArgumentException("random engine must not be null");
		double[][] result = new double[count][];
		for (int i = 0; i < count; i++) result[i] = random(random);
		return result;
	}
	public double[][] random(int count, long seed) {
		return random(count, new MersenneTwister(seed));
	}

	private CopulaMeasureResult continuousMeasure(double[] x) {
		double[] u = new double[dimension()];
		double logValue = 0.0;
		for (int i = 0; i < dimension(); i++) {
			u[i] = marginals[i].cumulative(x[i]);
			logValue += marginals[i].logDensityOrMass(x[i]);
		}
		logValue += copula.logDensity(u);
		double value = Math.exp(logValue);
		if (Double.isNaN(logValue)) return invalid("continuous density is undefined at this point");
		return new CopulaMeasureResult(value, logValue, 0.0, 0,
				CopulaMeasureResult.Status.EXACT_CONTINUOUS,
				"analytic copula density times marginal densities");
	}

	private CopulaMeasureResult discreteMeasure(double[] x) {
		double[][] endpoints = new double[dimension()][2];
		for (int i = 0; i < dimension(); i++) {
			endpoints[i][0] = marginals[i].leftCumulative(x[i]);
			endpoints[i][1] = marginals[i].cumulative(x[i]);
			if (!validInterval(endpoints[i])) return invalid("invalid marginal CDF interval");
			if (endpoints[i][0] == endpoints[i][1]) {
				return new CopulaMeasureResult(0.0, 0.0, 0,
						CopulaMeasureResult.Status.RECTANGLE_DIFFERENCE,
						"observation has zero marginal mass");
			}
		}
		double value = cornerSum(endpoints);
		return nonnegative(value, 0.0, 1 << dimension(),
				CopulaMeasureResult.Status.RECTANGLE_DIFFERENCE,
				"copula CDF rectangle difference");
	}

	private CopulaMeasureResult mixedMeasure(double[] x, int continuous) {
		Evaluation coarse = mixedEvaluation(x, options.getDerivativeStep());
		Evaluation fine = mixedEvaluation(x,
				Math.max(options.getMinimumStep(), options.getDerivativeStep() / 2.0));
		if (!Double.isFinite(coarse.value) || !Double.isFinite(fine.value))
			return invalid("mixed numerical derivative produced a non-finite value");
		double marginalLogDensity = 0.0;
		for (int i = 0; i < dimension(); i++)
			if (marginals[i].isContinuous()) marginalLogDensity += marginals[i].logDensityOrMass(x[i]);
		double scale = Math.exp(marginalLogDensity);
		double value = fine.value * scale;
		double error = Math.abs(fine.value - coarse.value) * scale;
		return nonnegative(value, error, coarse.evaluations + fine.evaluations,
				CopulaMeasureResult.Status.NUMERICAL_MIXED_DERIVATIVE,
				"finite CDF differences with a Richardson step comparison over "
				+ continuous + " continuous coordinates");
	}

	private Evaluation mixedEvaluation(double[] x, double requestedStep) {
		double[][] endpoints = new double[dimension()][2];
		double denominator = 1.0;
		for (int i = 0; i < dimension(); i++) {
			double upper = marginals[i].cumulative(x[i]);
			if (marginals[i].isDiscrete()) {
				endpoints[i][0] = marginals[i].leftCumulative(x[i]);
				endpoints[i][1] = upper;
			} else {
				double step = Math.max(options.getMinimumStep(),
						Math.min(requestedStep, Math.max(upper, 1.0 - upper) * requestedStep));
				double lowerPoint = Math.max(0.0, upper - step);
				double upperPoint = Math.min(1.0, upper + step);
				endpoints[i][0] = lowerPoint;
				endpoints[i][1] = upperPoint;
				denominator *= upperPoint - lowerPoint;
			}
			if (!validInterval(endpoints[i])) return new Evaluation(Double.NaN, 0);
		}
		return new Evaluation(cornerSum(endpoints) / denominator, 1 << dimension());
	}

	private double cornerSum(double[][] endpoints) {
		int count = 1 << dimension();
		double sum = 0.0;
		double correction = 0.0;
		double[] point = new double[dimension()];
		for (int mask = 0; mask < count; mask++) {
			int lowerCount = 0;
			for (int i = 0; i < dimension(); i++) {
				boolean upper = (mask & (1 << i)) != 0;
				point[i] = endpoints[i][upper ? 1 : 0];
				if (!upper) lowerCount++;
			}
			double term = ((lowerCount & 1) == 0 ? 1.0 : -1.0)
					* copula.cumulative(point);
			double adjusted = term - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
		}
		return sum;
	}

	private CopulaMeasureResult nonnegative(double value, double error,
			int evaluations, CopulaMeasureResult.Status successStatus, String message) {
		if (!Double.isFinite(value)) return invalid("copula measure is not finite");
		if (value < -options.getNegativeTolerance()) {
			return new CopulaMeasureResult(0.0, Math.max(error, -value), evaluations,
					CopulaMeasureResult.Status.NUMERICAL_WARNING,
					message + "; cancellation produced a negative estimate");
		}
		return new CopulaMeasureResult(Math.max(0.0, value), error, evaluations,
				successStatus, message);
	}

	private static boolean validInterval(double[] interval) {
		return Double.isFinite(interval[0]) && Double.isFinite(interval[1])
				&& interval[0] >= 0.0 && interval[1] <= 1.0
				&& interval[0] <= interval[1];
	}

	private boolean validObservation(double[] x) {
		if (x == null || x.length != dimension()) return false;
		for (double value : x) if (Double.isNaN(value)) return false;
		return true;
	}

	private static CopulaMeasureResult invalid(String message) {
		return new CopulaMeasureResult(Double.NaN, Double.NaN, 0,
				CopulaMeasureResult.Status.INVALID_INPUT, message);
	}

	private static final class Evaluation {
		final double value;
		final int evaluations;
		Evaluation(double value, int evaluations) {
			this.value = value;
			this.evaluations = evaluations;
		}
	}
}
