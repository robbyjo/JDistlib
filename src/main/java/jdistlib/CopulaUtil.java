/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.rng.RandomEngine;

/** Package-private copula validation and numerical helpers. */
final class CopulaUtil {
	private CopulaUtil() {}

	static double logAdd(double a, double b) {
		double maximum = Math.max(a, b);
		return Double.isInfinite(maximum) ? maximum
				: maximum + Math.log1p(Math.exp(Math.min(a, b) - maximum));
	}

	static double logExpm1(double x) {
		return x > 40.0 ? x + Math.log1p(-Math.exp(-x)) : Math.log(Math.expm1(x));
	}

	static double softplus(double x) { return logAdd(0.0, x); }

	/** Adaptive conditional integration for bivariate elliptical copulas. */
	static double ellipticalCumulative(double[] u, double rho, double df) {
		double upper = Math.min(u[0], u[1]), other = Math.max(u[0], u[1]);
		if (other == 1.0) return upper;
		if (Double.isInfinite(df) && rho == 0.0) return upper * other;
		double threshold = Double.isInfinite(df) ? Normal.quantile(other, 0, 1, true, false)
				: T.quantile(other, df, true, false);
		java.util.function.DoubleUnaryOperator conditional = p -> {
			if (p == 0.0) return Double.isInfinite(df) ? (rho > 0 ? 1.0 : rho < 0 ? 0.0 : other)
					: T.cumulative(rho * Math.sqrt((df + 1.0) / (1.0-rho*rho)), df+1, true, false);
			double x = Double.isInfinite(df) ? Normal.quantile(p, 0, 1, true, false) : T.quantile(p, df, true, false);
			double scale = Math.sqrt((1.0-rho*rho) * (Double.isInfinite(df) ? 1.0 : (df+x*x)/(df+1.0)));
			return Double.isInfinite(df) ? Normal.cumulative((threshold-rho*x)/scale,0,1,true,false)
					: T.cumulative((threshold-rho*x)/scale,df+1,true,false);
		};
		double a=conditional.applyAsDouble(0),b=conditional.applyAsDouble(upper/2),c=conditional.applyAsDouble(upper);
		double value=adaptiveSimpson(conditional,0,upper,a,b,c,upper*(a+4*b+c)/6,2e-11,22);
		return Math.max(Math.max(0.0,u[0]+u[1]-1),Math.min(upper,value));
	}

	private static double adaptiveSimpson(java.util.function.DoubleUnaryOperator f,
			double low,double high,double a,double b,double c,double whole,double tolerance,int depth) {
		double middle=(low+high)/2,left=f.applyAsDouble((low+middle)/2),right=f.applyAsDouble((middle+high)/2);
		double first=(middle-low)*(a+4*left+b)/6,second=(high-middle)*(b+4*right+c)/6;
		double error=first+second-whole;
		if(depth==0||Math.abs(error)<=15*tolerance)return first+second+error/15;
		return adaptiveSimpson(f,low,middle,a,left,b,first,tolerance/2,depth-1)
				+adaptiveSimpson(f,middle,high,b,right,c,second,tolerance/2,depth-1);
	}

	static void requireDimension(int dimension) {
		if (dimension < 1) throw new IllegalArgumentException("dimension must be positive");
	}

	static void requireMultivariateDimension(int dimension) {
		if (dimension < 2) throw new IllegalArgumentException("copula dimension must be at least two");
	}

	static boolean validPoint(double[] u, int dimension) {
		return CopulaDiagnostics.inspect(u, dimension).isValid();
	}

	static boolean interiorPoint(double[] u, int dimension) {
		return CopulaDiagnostics.inspect(u, dimension).isInterior();
	}

	static boolean hasZero(double[] u) {
		for (double value : u) if (value == 0.0) return true;
		return false;
	}

	static void requirePair(int first, int second, int dimension) {
		if (first < 0 || second < 0 || first >= dimension || second >= dimension)
			throw new IndexOutOfBoundsException("copula coordinate index out of range");
	}

	static double uniformOpen(RandomEngine random) {
		return clampOpen(random.nextDouble());
	}

	static double clampOpen(double value) {
		if (value <= 0.0) return Math.nextUp(0.0);
		if (value >= 1.0) return Math.nextDown(1.0);
		return value;
	}

	static double[][] copyMatrix(double[][] matrix) {
		double[][] result = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) result[i] = matrix[i].clone();
		return result;
	}

	static double[] zeros(int dimension) { return new double[dimension]; }

	static double logSumExp(double[] values) {
		double maximum = Double.NEGATIVE_INFINITY;
		for (double value : values) maximum = Math.max(maximum, value);
		if (Double.isInfinite(maximum)) return maximum;
		double sum = 0.0;
		for (double value : values) sum += Math.exp(value - maximum);
		return maximum + Math.log(sum);
	}

	static void validateCorrelation(double[][] correlation) {
		MultivariateDistributionUtil.Factor factor =
				MultivariateDistributionUtil.factor(correlation);
		if (factor == null) {
			throw new IllegalArgumentException(
					"correlation must be a finite symmetric positive-definite matrix");
		}
		for (int i = 0; i < correlation.length; i++) {
			double tolerance = 1e-12 * Math.max(1.0, Math.abs(correlation[i][i]));
			if (Math.abs(correlation[i][i] - 1.0) > tolerance) {
				throw new IllegalArgumentException("correlation diagonal must equal one");
			}
		}
	}
}
