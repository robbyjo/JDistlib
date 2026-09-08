/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 3 of the License.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 */
package jdistlib.disttest;

import java.util.HashSet;
import java.util.Set;

import jdistlib.ChiSquare;
import jdistlib.Normal;
import jdistlib.math.VectorMath;
import jdistlib.exception.PrecisionException;
import jdistlib.util.Debug;

import static java.lang.Math.abs;
import static java.lang.Math.asin;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.signum;
import static java.lang.Math.sqrt;
import static jdistlib.disttest.Utils.calculate_ecdf;
import static jdistlib.math.Constants.M_1_SQRT_2;
import static jdistlib.util.Utilities.sort;

/** 
 * A package about normality testing. Taken from various places.
 * @author Roby Joehanes
 */
public class NormalityTest {
	/**
	 * Invoke shapiro_wilk_statistic with sort = false
	 * @param X pre-sorted array of numbers
	 */
	public static final double shapiro_wilk_statistic(double[] X) {
		return shapiro_wilk_statistic(X, false);
	}

	/**
	 * Compute Shapiro-Wilk statistic
	 * @param x array of number
	 * @param sort if true, then this function will attempt to sort the X first.
	 */
	@SuppressWarnings("unused")
	public static final double shapiro_wilk_statistic(double[] x, boolean sort) {
		// constant for Shapiro-wilk
		final double[]
			c1 = {0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056},
			c2 = {0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633};
		int	n = x.length, n2 = n/2; // yes, integer division
		if (n < 3) {
			if (Debug.warningAsError)
				throw new PrecisionException("Shapiro Wilks error: n < 3", n);
			return 0;
		}
		double[] X = x;
		if (sort) {
			X = new double[n];
			System.arraycopy(x, 0, X, 0, n);
			sort(X);
		}
		double[] a = new double[n2];
		if (n == 3)
			a[0] = M_1_SQRT_2;
		else {
			double an25 = n + 0.25, sum2 = 0, sqrtSum2;
			for (int i = 0; i < n2; i++) {
				double val = a[i] = Normal.quantile((i + 0.625) / an25, 0, 1, true, false);
				sum2 += val * val;
			}
			sum2 *= 2;
			sqrtSum2 = sqrt(sum2);
			double rsn = 1 / sqrt(n), a1 = poly(c1, rsn) - a[0] / sqrtSum2, fac, a2;
			int i1 = 1;
			if (n > 5) {
				i1 = 2;
				a2 = -a[1] / sqrtSum2 + poly(c2, rsn);
				fac = -sqrt( (sum2 - 2 * a[0] * a[0] - 2 * a[1] * a[1]) / (1 - 2 * a1 * a1 - 2 * a2 * a2) );
				a[1] = a2;
			} else
				fac = -sqrt( (sum2 - 2 * a[0] * a[0]) / (1 - 2 * a1 * a1) );
			a[0] = a1;
			for (int i = i1; i < n2; i++)
				a[i] /= fac;
		}

		double range = X[n - 1] - X[0];
		if (range < 1e-19)
			throw new PrecisionException("Shapiro-Wilks error: Range too small!", range);
		double xx = X[0] / range, sx = xx, sa = -a[0], xi;
		int j = n - 1;
		for (int i = 1; i < n; j--) {
			xi = X[i] / range;
			sx += xi;
			i++;
			if (i != j)
				sa += signum(i - j) * a[min(i,j) - 1];
			xx = xi;
		}

		// W statistic is a squared correlation between data and coefficients
		double ssa = 0, ssx = 0, sax = 0;
		sa /= n; sx /= n; j = n - 1;
		for (int i = 0; i < n; ++i, --j) {
			double
			asa = i != j ? signum(i - j) * a[min(i,j)] - sa : -sa,
				xsx = X[i] / range - sx;
			ssa += asa * asa;
			ssx += xsx * xsx;
			sax += asa * xsx;
		}

		// W1 = (1-W) calculated to avoid excessive rounding error for W
		// very near 1 (a potential problem in very large samples)

		double
			ssassx = sqrt(ssa * ssx),
			w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);

		if (n > 5000 && Debug.warningAsError)
			throw new PrecisionException("Shapiro-Wilks error: n > 5000", n);
		return 1 - w1;
	}

	/**
	 * 
	 * @param w
	 * @param n The length of the array
	 * @return p value
	 */
	public static final double shapiro_wilk_pvalue(double w, int n) {
		final double kVerySmallValue = 1e-99;

		// constant for Shapiro-wilk
		final double[]
			c3 = {0.544, -0.39978, 0.025054, -6.714e-4},
			c4 = {1.3822, -0.77857, 0.062767, -0.0020322},
			c5 = {-1.5861, -0.31082, -0.083751, 0.0038915},
			c6 = {-0.4803, -0.082676, 0.0030302},
			g = {-2.273, 0.459};

		if (n < 3) {
			if (Debug.warningAsError)
				throw new PrecisionException("Shapiro Wilks error: n < 3", n);
			return 1;
		}
		if (n == 3) // exact P value :
			return max(0, 1.90985931710274 * (asin(sqrt(w)) - 1.04719755119660));

		double
		y = log(1 - w),
		xx = log(n),
		gamma,
		m, s;

		if (n <= 11) {
			gamma = poly(g, n);
			if (y >= gamma)
				return kVerySmallValue; // rather use an even smaller value, or NA ?
			y = -log(gamma - y);
			m = poly(c3, n);
			s = exp(poly(c4, n));
		} else {
			m = poly(c5, xx);
			s = exp(poly(c6, xx));
		}
		return Normal.cumulative(y, m, s, false, false);
	}

	public static final double anderson_darling_statistic(double[] X) {
		int n = X.length;
		double[] y = standardizedSample(X);
		sort(y);
		double sum = 0;
		for (int i = 0; i < n; i++)
			sum += (2.0 * i + 1) * (Normal.cumulative(y[i], 0, 1, true, true)
				+ Normal.cumulative(y[n - 1 - i], 0, 1, false, true));
		return -n - sum / n;
	}

	/**
	 * 
	 * @param value
	 * @param n The length of the array
	 * @return p value
	 */
	public static final double anderson_darling_pvalue(double value, int n) {
		if (Double.isNaN(value) || value < 0 || n < 1) return Double.NaN;
		double
			aa = value * (1 + 0.75/n + 2.25 / ((double)n*n)),
			aasq = aa * aa;
		if (aa < 0.2)
			return 1 - exp(-13.436 + 101.14 * aa - 223.73 * aasq);
		else if (aa < 0.34)
			return 1 - exp(-8.318 + 42.796 * aa - 59.938 * aasq);
		else if (aa < 0.6)
			return exp(0.9177 - 4.279 * aa - 1.38 * aasq);
		return aa < 10 ? exp(1.2937 - 5.709 * aa + 0.0186 * aasq) : 3.7e-24;
	}

	public static final double cramer_vonmises_statistic(double[] X) {
		int
			n = X.length,
			n2 = n * 2;
		double[] sortedZ = standardizedSample(X);

		// Standardize X
		for (int i = 0; i < n; i++)
			sortedZ[i] = Normal.cumulative(sortedZ[i], 0, 1, true, false);
		sort(sortedZ);

		double w = 1.0 / (12.0 * n);
		for (int i = 0; i < n; i++) {
			double val = (2 * i + 1.0) / n2 - sortedZ[i];
			w += val * val;
		}
		return w;
	}

	/**
	 * 
	 * @param w
	 * @param n The length of the array
	 * @return p value
	 */
	public static final double cramer_vonmises_pvalue(double w, int n) {
		if (Double.isNaN(w) || w < 0 || n < 1) return Double.NaN;
		double
			ww = (1 + 0.5/n) * w,
			ww2 = ww * ww;
	    if (ww < 0.0275)
	        return 1 - exp(-13.953 + 775.5 * ww - 12542.61 * ww2);
	    else if (ww < 0.051)
	        return 1 - exp(-5.903 + 179.546 * ww - 1515.29 * ww2);
	    else if (ww < 0.092)
	        return exp(0.886 - 31.62 * ww + 10.897 * ww2);
	    // The fitted approximation is valid only below 1.1; retain the
	    // published numerical floor instead of extrapolating it upward.
	    return ww < 1.1 ? exp(1.111 - 34.242 * ww + 12.832 * ww2) : 7.37e-10;
	}

	/**
	 * Calculate D'Agostino-Pearson test for normality. Follows Chi^2 distribution with df = 2.
	 * This is the infamous omnibus test. Note: It breaks down for n &lt;= 7. See:<br>
	 * Doornik, Hansen, An Omnibus Test for Univariate and Multivariate Normality, 1994
	 * 
	 * @param X
	 * @return test statistic
	 */
	public static final double dagostino_pearson_statistic(double[] X) {
		double n = X.length;
		if (n < 8) return Double.NaN;
		double[] moments = standardizedMoments(X);
		// D'Agostino skewness and Anscombe-Glynn kurtosis transforms.
		double y = moments[0] * sqrt((n + 1) * (n + 3) / (6 * (n - 2)));
		double beta = 3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3)
			/ ((n - 2) * (n + 5) * (n + 7) * (n + 9));
		double w2 = -1 + sqrt(2 * (beta - 1));
		double a = sqrt(2 / (w2 - 1)), t = abs(y / a);
		double z1 = Math.copySign(log(t + Math.hypot(t, 1)), y) / sqrt(0.5 * log(w2));
		double expected = 3 * (n - 1) / (n + 1);
		double variance = 24 * n * (n - 2) * (n - 3)
			/ ((n + 1) * (n + 1) * (n + 3) * (n + 5));
		double rootBeta = 6 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9))
			* sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)));
		double A = 6 + 8 / rootBeta * (2 / rootBeta + sqrt(1 + 4 / (rootBeta * rootBeta)));
		double denominator = 1 + (moments[1] - expected) / sqrt(variance) * sqrt(2 / (A - 4));
		double z2 = (1 - 2 / (9 * A) - Math.cbrt((1 - 2 / A) / denominator)) / sqrt(2 / (9 * A));
		return z1 * z1 + z2 * z2;
	}

	public static final double dagostino_pearson_pvalue(double value)
	{	return ChiSquare.cumulative(value, 2, false, false); }

	/**
	 * Calculate Jarque-Bera Normality Test. Follows Chi^2 distribution with df = 2<br>
	 * http://en.wikipedia.org/wiki/Jarque-Bera_test
	 * 
	 * @param X
	 * @return test statistic
	 */
	public static final double jarque_bera_statistic(double[] X) {
		double[] moments = standardizedMoments(X);
		double excess = moments[1] - 3;
		return X.length * (moments[0] * moments[0] + excess * excess / 4) / 6;
	}

	/** Scale central moments before taking powers, preserving affine invariance. */
	private static double[] standardizedMoments(double[] x) {
		double m2 = 0, m3 = 0, m4 = 0;
		for (double z : standardizedSample(x)) {
			double z2 = z * z;
			m2 += z2; m3 += z2 * z; m4 += z2 * z2;
		}
		m2 /= x.length; m3 /= x.length; m4 /= x.length;
		return new double[] {m3 / (m2 * sqrt(m2)), m4 / (m2 * m2)};
	}

	private static double[] standardizedSample(double[] x) {
		double mean = VectorMath.mean(x), sd = VectorMath.sd(x);
		double[] z = x.clone();
		if (Double.isInfinite(sd)) {
			// A sample SD can exceed MAX_VALUE even when every datum is finite.
			double scale = 0;
			for (double value : x) scale = max(scale, abs(value));
			for (int i = 0; i < z.length; i++) z[i] /= scale;
			mean = VectorMath.mean(z); sd = VectorMath.sd(z);
		}
		for (int i = 0; i < z.length; i++) {
			double delta = z[i] - mean;
			z[i] = Double.isInfinite(delta) && Double.isFinite(z[i]) && Double.isFinite(mean)
				? z[i] / sd - mean / sd : delta / sd;
		}
		return z;
	}

	public static final double jarque_bera_pvalue(double value)
	{	return ChiSquare.cumulative(value, 2, false, false); }

	/**
	 * Perform Kolmogorov-Smirnov two-sided normality test.
	 * @param X
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X) {
		return DistributionTest.kolmogorov_smirnov_test(X, new Normal());
	}

	/**
	 * Computation of Kolmogorov-Smirnov statistics. Deprecated. Please use kolmogorov_smirnov_test instead.
	 * @deprecated
	 * @param X
	 * @return statistic
	 */
  @Deprecated
	public static final double kolmogorov_smirnov_statistic(double[] X) {
		return DistributionTest.kolmogorov_smirnov_statistic(X, new Normal(), TestKind.TWO_SIDED);
	}

	/**
	 * Computation of Kolmogorov-Smirnov p-value. Deprecated. Please use kolmogorov_smirnov_test instead.
	 * @deprecated
	 * @param d
	 * @param X
	 * @return p-value
	 */
  @Deprecated
	public static final double kolmogorov_smirnov_pvalue(double d, double[] X) {
		int nX = X.length;
		Set<Double> set = new HashSet<Double>(nX);
		for (double x: X)
			set.add(x);
		if (set.size() < nX)
			return DistributionTest.kolmogorov_smirnov_pvalue_inexact(d, nX);
		return DistributionTest.kolmogorov_smirnov_pvalue_exact(d, nX);
	}

	/**
	 * Exactly identical as kolmogorov_smirnov_statistic
	 * @param X
	 * @return test statistic
	 */
	public static final double kolmogorov_lilliefors_statistic(double[] X) {
		int n = X.length;
		double[] sortedZ = standardizedSample(X);
		if (n == 0 || !Double.isFinite(sortedZ[0])) return Double.NaN;
		sort(sortedZ);
	
		double
			cdfZ[] = calculate_ecdf(sortedZ),
			max = 0;
	
		for (int i = 0; i < n; i++) {
			double
				cdfNormal = Normal.cumulative(sortedZ[i], 0, 1, true, false),
				M = abs(cdfNormal - cdfZ[i]),
				MPrime = i > 0 ? abs(cdfNormal - cdfZ[i-1]) : abs(cdfNormal),
				curMax = max(M, MPrime);
			if (curMax > max)
				max = curMax;
		}
		return max;
	}

	/**
	 * 
	 * @param k
	 * @param n The length of the array
	 * @return p-value
	 */
	public static final double kolmogorov_lilliefors_pvalue(double k, int n) {
		double
			Kd = k,
			nd = n;
		if (n > 100) {
			Kd = k * pow((n/100.0),0.49);
			nd = 100;
		}
		double pvalue = exp(-7.01256 * Kd * Kd * (nd + 2.78019) + 2.99587 * Kd * sqrt(nd + 2.78019) - 0.122119 + 0.974598/sqrt(nd) + 1.67997/nd);
		if (pvalue > 0.1) {
			double
				KK = (sqrt(n) - 0.01 + 0.85/sqrt(n)) * k,
				KK2 = KK * KK,
				KK3 = KK2 * KK,
				KK4 = KK2 * KK2;
			if (KK <= 0.302)
				return 1;
			else if (KK <= 0.5)
				return 2.76773 - 19.828315 * KK + 80.709644 * KK2 - 138.55152 * KK3 + 81.218052 * KK4;
			else if (KK <= 0.9)
				return -4.901232 + 40.662806 * KK - 97.490286 * KK2 + 94.029866 * KK3 - 32.355711 * KK4;
			else if (KK <= 1.31)
				return 6.198765 - 19.558097 * KK + 23.186922 * KK2 - 12.234627 * KK3 + 2.423045 * KK4;
			return 0;
		}
		return pvalue; 
	}

	/**
	 * Shapiro-Francia normality test
	 * @param X a sorted array of values
	 * @return test statistic
	 */
	public static final double shapiro_francia_statistic(double[] X) {
		int n = X.length;
		double
			denum = n + 0.25,
			sumX = 0,
			sumY = 0,
			sumXSq = 0,
			sumYSq = 0,
			sumXY = 0,
			x,
			y;

		for (int i = 0; i < n; i++) {
			y = Normal.quantile((i + 0.625) / denum, 0, 1, true, false);
			x = X[i];
			sumX += x;
			sumY += y;
			sumXSq += x * x;
			sumYSq += y * y;
			sumXY += x * y;
		}
		double cor = (n * sumXY - sumX * sumY) / (sqrt(n * sumXSq - sumX * sumX) * sqrt(n * sumYSq - sumY * sumY));
		return cor * cor;
	}

	/**
	 * P-value of Shapiro-Francia normality test
	 * @param w the result from ShapiroFrancia's statistic
	 * @param n the length of the original array
	 * @return p-value
	 */
	public static final double shapiro_francia_pvalue(double w, int n) {
		double
			a = log(n),
			b = log(a),
			mu = -1.2725 + 1.0521 * (b - a),
			sigma = 1.0308 - 0.26758 * (b + 2/a);
		return Normal.cumulative(log(1 - w), mu, sigma, false, false);
	}

	/**
	 * Helper function to calculate polynomials (for Shapiro-Wilk)
	 * @param coeff
	 * @param x
	 * @return polynomial value
	 */
	private static final double poly(double[] coeff, double x) {
		int n = coeff.length;
		double result = coeff[0];
		if (n > 1) {
			double p = x * coeff[n - 1];
			for (int i = n - 2; i > 0; i--)
				p = (p + coeff[i]) * x;
			result += p;
		}
		return result;
	}

}
