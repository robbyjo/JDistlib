/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

/**
 * Bivariate copula adapter exposing conditional CDFs and their inverses.
 * Built-in families use analytic conditional CDFs; custom copulas retain the
 * finite-difference fallback selected by {@code derivativeStep}.
 */
public final class PairCopula {
	private final Copula copula;
	private final double derivativeStep;

	public PairCopula(Copula copula) { this(copula, 1e-5); }

	public PairCopula(Copula copula, double derivativeStep) {
		if (copula == null || copula.dimension() != 2)
			throw new IllegalArgumentException("a pair copula must have dimension two");
		if (!(derivativeStep > 0.0 && derivativeStep < 0.1)
				|| !Double.isFinite(derivativeStep))
			throw new IllegalArgumentException("conditional derivative step must be in (0, 0.1)");
		this.copula = copula;
		this.derivativeStep = derivativeStep;
	}

	public Copula getCopula() { return copula; }
	public double getDerivativeStep() { return derivativeStep; }
	public double logDensity(double first, double second) {
		return copula.logDensity(new double[] {first, second});
	}

	/** Returns {@code P[U2 <= second | U1 = first]}. */
	public double conditionalSecondGivenFirst(double first, double second) {
		if (!unit(first) || !unit(second)) return Double.NaN;
		if (second == 0.0 || second == 1.0) return second;
		if (copula instanceof IndependenceCopula) return second;
		if (copula instanceof GaussianCopula)
			return gaussianSecondGivenFirst((GaussianCopula) copula, first, second);
		if (copula instanceof StudentTCopula)
			return studentSecondGivenFirst((StudentTCopula) copula, first, second);
		if (copula instanceof ClaytonCopula)
			return claytonSecondGivenFirst((ClaytonCopula) copula, first, second);
		if (copula instanceof GumbelCopula)
			return gumbelSecondGivenFirst((GumbelCopula) copula, first, second);
		if (copula instanceof FrankCopula)
			return frankSecondGivenFirst((FrankCopula) copula, first, second);
		return partial(first, second, true);
	}

	/** Returns {@code P[U1 <= first | U2 = second]}. */
	public double conditionalFirstGivenSecond(double first, double second) {
		if (!unit(first) || !unit(second)) return Double.NaN;
		if (first == 0.0 || first == 1.0) return first;
		if (copula instanceof IndependenceCopula) return first;
		if (copula instanceof GaussianCopula)
			return gaussianSecondGivenFirst((GaussianCopula) copula, second, first);
		if (copula instanceof StudentTCopula)
			return studentSecondGivenFirst((StudentTCopula) copula, second, first);
		if (copula instanceof ClaytonCopula)
			return claytonSecondGivenFirst((ClaytonCopula) copula, second, first);
		if (copula instanceof GumbelCopula)
			return gumbelSecondGivenFirst((GumbelCopula) copula, second, first);
		if (copula instanceof FrankCopula)
			return frankSecondGivenFirst((FrankCopula) copula, second, first);
		return partial(first, second, false);
	}

	/** Inverts {@link #conditionalSecondGivenFirst(double, double)} in its second argument. */
	public double inverseSecondGivenFirst(double first, double probability) {
		if (!unit(first) || !unit(probability)) return Double.NaN;
		if (probability == 0.0 || probability == 1.0) return probability;
		if (copula instanceof IndependenceCopula) return probability;
		if (copula instanceof GaussianCopula)
			return inverseGaussianSecond((GaussianCopula) copula, first, probability);
		if (copula instanceof StudentTCopula)
			return inverseStudentSecond((StudentTCopula) copula, first, probability);
		if (copula instanceof ClaytonCopula)
			return inverseClaytonSecond((ClaytonCopula) copula, first, probability);
		if (copula instanceof FrankCopula)
			return inverseFrankSecond((FrankCopula) copula, first, probability);
		return invert(first, probability, true);
	}

	/** Inverts {@link #conditionalFirstGivenSecond(double, double)} in its first argument. */
	public double inverseFirstGivenSecond(double second, double probability) {
		if (!unit(second) || !unit(probability)) return Double.NaN;
		if (probability == 0.0 || probability == 1.0) return probability;
		if (copula instanceof IndependenceCopula) return probability;
		if (copula instanceof GaussianCopula)
			return inverseGaussianSecond((GaussianCopula) copula, second, probability);
		if (copula instanceof StudentTCopula)
			return inverseStudentSecond((StudentTCopula) copula, second, probability);
		if (copula instanceof ClaytonCopula)
			return inverseClaytonSecond((ClaytonCopula) copula, second, probability);
		if (copula instanceof FrankCopula)
			return inverseFrankSecond((FrankCopula) copula, second, probability);
		return invert(second, probability, false);
	}

	private double partial(double first, double second, boolean varyFirst) {
		double coordinate = varyFirst ? first : second;
		double low = Math.max(0.0, coordinate - derivativeStep);
		double high = Math.min(1.0, coordinate + derivativeStep);
		double lower = varyFirst
				? copula.cumulative(new double[] {low, second})
				: copula.cumulative(new double[] {first, low});
		double upper = varyFirst
				? copula.cumulative(new double[] {high, second})
				: copula.cumulative(new double[] {first, high});
		return clampProbability((upper - lower) / (high - low));
	}

	private double invert(double conditioning, double probability,
			boolean secondGivenFirst) {
		double low = 0.0;
		double high = 1.0;
		for (int i = 0; i < 52; i++) {
			double middle = low + (high - low) / 2.0;
			double value = secondGivenFirst
					? conditionalSecondGivenFirst(conditioning, middle)
					: conditionalFirstGivenSecond(middle, conditioning);
			if (value < probability) low = middle;
			else high = middle;
		}
		return CopulaUtil.clampOpen(low + (high - low) / 2.0);
	}

	private static double gaussianSecondGivenFirst(GaussianCopula gaussian,
			double first, double second) {
		double rho = gaussian.getCorrelation()[0][1];
		double firstNormal = Normal.quantile(first, 0.0, 1.0, true, false);
		double secondNormal = Normal.quantile(second, 0.0, 1.0, true, false);
		return Normal.cumulative((secondNormal - rho * firstNormal)
				/ Math.sqrt(1.0 - rho * rho), 0.0, 1.0, true, false);
	}

	private static double inverseGaussianSecond(GaussianCopula gaussian,
			double first, double probability) {
		double rho = gaussian.getCorrelation()[0][1];
		double firstNormal = Normal.quantile(first, 0.0, 1.0, true, false);
		double conditional = Normal.quantile(probability, 0.0, 1.0, true, false);
		double secondNormal = rho * firstNormal
				+ Math.sqrt(1.0 - rho * rho) * conditional;
		return CopulaUtil.clampOpen(
				Normal.cumulative(secondNormal, 0.0, 1.0, true, false));
	}

	private static double studentSecondGivenFirst(StudentTCopula student,
			double first, double second) {
		double rho = student.getCorrelation()[0][1];
		double df = student.getDegreesOfFreedom();
		double firstT = T.quantile(first, df, true, false);
		double secondT = T.quantile(second, df, true, false);
		double scale = Math.sqrt((df + firstT * firstT) * (1.0 - rho * rho)
				/ (df + 1.0));
		return T.cumulative((secondT - rho * firstT) / scale,
				df + 1.0, true, false);
	}

	private static double inverseStudentSecond(StudentTCopula student,
			double first, double probability) {
		double rho = student.getCorrelation()[0][1];
		double df = student.getDegreesOfFreedom();
		double firstT = T.quantile(first, df, true, false);
		double scale = Math.sqrt((df + firstT * firstT) * (1.0 - rho * rho)
				/ (df + 1.0));
		double secondT = rho * firstT + scale
				* T.quantile(probability, df + 1.0, true, false);
		return CopulaUtil.clampOpen(T.cumulative(secondT, df, true, false));
	}

	private static double claytonSecondGivenFirst(ClaytonCopula clayton,
			double first, double second) {
		double theta = clayton.getTheta();
		if (theta == 0.0) return second;
		if (first == 0.0) return second == 0.0 ? 0.0 : 1.0;
		double scaledExcess = Math.exp(theta * Math.log(first))
				* Math.expm1(-theta * Math.log(second));
		return clampProbability(Math.exp(-(theta + 1.0) / theta
				* Math.log1p(scaledExcess)));
	}

	private static double inverseClaytonSecond(ClaytonCopula clayton,
			double first, double probability) {
		double theta = clayton.getTheta();
		if (theta == 0.0) return probability;
		double exponent = -theta / (theta + 1.0) * Math.log(probability);
		double termLog = -theta * Math.log(first) + Math.log(Math.expm1(exponent));
		double logInversePower = termLog > 0.0
				? termLog + Math.log1p(Math.exp(-termLog))
				: Math.log1p(Math.exp(termLog));
		return CopulaUtil.clampOpen(Math.exp(-logInversePower / theta));
	}

	private static double gumbelSecondGivenFirst(GumbelCopula gumbel,
			double first, double second) {
		double theta = gumbel.getTheta();
		if (theta == 1.0) return second;
		if (first == 0.0) return second == 0.0 ? 0.0 : 1.0;
		double firstLog = -Math.log(first);
		double secondLog = -Math.log(second);
		double sum = Math.pow(firstLog, theta) + Math.pow(secondLog, theta);
		double copula = Math.exp(-Math.pow(sum, 1.0 / theta));
		return clampProbability(copula * Math.pow(sum, 1.0 / theta - 1.0)
				* Math.pow(firstLog, theta - 1.0) / first);
	}

	private static double frankSecondGivenFirst(FrankCopula frank,
			double first, double second) {
		double theta = frank.getTheta();
		if (theta == 0.0) return second;
		double firstExponential = Math.exp(-theta * first);
		double secondMinusOne = Math.expm1(-theta * second);
		double denominator = Math.expm1(-theta)
				+ Math.expm1(-theta * first) * secondMinusOne;
		return clampProbability(firstExponential * secondMinusOne / denominator);
	}

	private static double inverseFrankSecond(FrankCopula frank,
			double first, double probability) {
		double theta = frank.getTheta();
		if (theta == 0.0) return probability;
		double firstExponential = Math.exp(-theta * first);
		double secondExponential = 1.0 + probability * Math.expm1(-theta)
				/ (firstExponential * (1.0 - probability) + probability);
		return CopulaUtil.clampOpen(-Math.log(secondExponential) / theta);
	}

	private static double clampProbability(double value) {
		if (Double.isNaN(value)) return value;
		return Math.max(0.0, Math.min(1.0, value));
	}

	private static boolean unit(double value) {
		return Double.isFinite(value) && value >= 0.0 && value <= 1.0;
	}
}
