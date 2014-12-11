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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import jdistlib.Ansari;
import jdistlib.Binomial;
import jdistlib.ChiSquare;
import jdistlib.F;
import jdistlib.Normal;
import jdistlib.Poisson;
import jdistlib.SignRank;
import jdistlib.T;
import jdistlib.Wilcoxon;
import jdistlib.generic.GenericDistribution;
import static java.lang.Math.*;
import static jdistlib.math.Constants.M_1_SQRT_2PI;
import static jdistlib.math.Constants.M_PISQ_8;
import static jdistlib.math.MathFunctions.lchoose;
import static jdistlib.util.Utilities.*;
import static jdistlib.math.VectorMath.*;

/**
 * Comparing two distributions
 * @author Roby Joehanes
 *
 */
public class DistributionTest {
	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution, two-sided, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y) {
		return kolmogorov_smirnov_test(X, Y, TestKind.TWO_SIDED, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution, two-sided.
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y, boolean isExact) {
		return kolmogorov_smirnov_test(X, Y, TestKind.TWO_SIDED, isExact);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y, TestKind kind) {
		return kolmogorov_smirnov_test(X, Y, kind, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution.
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, double[] Y, TestKind kind, boolean isExact) {
		int
			nX = X.length,
			nY = Y.length,
			n_total = nX + nY;
		Set<Double> set = new HashSet<Double>();
		double w[] = c(X, Y);
		int[] ow = order(w);
		double z[] = new double[n_total], cs = 0;
		for (int i = 0; i < n_total; i++) {
			cs += (ow[i] <= nX ? 1.0 / nX : -1.0/nY);
			z[i] = cs;
			set.add(w[i]);
		}
		int new_n = set.size();
		// Has ties
		if (new_n < n_total) {
			double[] dz = new double[n_total], new_z = new double[new_n];
			System.arraycopy(w, 0, dz, 0, n_total);
			sort(dz);
			dz = diff(dz);
			for (int i = 0, j = 0; i < dz.length; i++) {
				if (dz[i] != 0)
					new_z[j++] = z[i];
			}
			new_z[new_n-1] = z[n_total-1];
			z = new_z;
		}
		double maxDiv = Double.NaN, pval = maxDiv;
		switch(kind) {
			case TWO_SIDED: maxDiv = max(vabs(z)); break;
			case LOWER: maxDiv = -min(z); break;
			case GREATER: maxDiv = max(z); break;
		}

		// Has ties
		if (new_n < n_total || !isExact) {
			double n = nX * (nY * 1.0 / (nX + nY));
			if (kind != TestKind.TWO_SIDED) {
				pval = exp(-2 * n * maxDiv * maxDiv);
			} else {
				double tol = 1e-10;
				double d = sqrt(n) * maxDiv;
				if (d < 1) {
					int k_max = (int) sqrt(2 - log(tol));
					double
						z_star = -M_PISQ_8 / (d * d),
						w_star = log(d),
						s = 0;
					for (int k = 1; k < k_max; k += 2)
						s += exp(k * k * z_star - w_star);
					pval = 1 - s / M_1_SQRT_2PI;
				} else {
					int k = 1;
					double
						z_star = -2 * d * d,
						s = -1,
						oldval = 0,
						newval = 1;
					while (abs(oldval - newval) > tol) {
						oldval = newval;
						newval += 2 * s * exp(z_star * k * k);
						s *= -1;
						k++;
					}
					pval = 1 - newval;
				}
			}
		} else {
			if (nX > nY) {
				int temp = nY;
				nY = nX;
				nX = temp;
			}

			double
				q = (0.5 + floor(maxDiv * nX * nY - 1e-7)) / (nX * nY),
				u[] = new double[nY + 1],
				dx = 1.0 / nX, dy = 1.0 / nY;

			for (int j = 0; j <= nY; j++)
				u[j] = (j / nY) > q ? 0: 1;
			for(int i = 1; i <= nX; i++) {
				double w_star = (double)(i) / ((double)(i + nY));
				u[0] = (i * dx) > q ? 0 : w_star * u[0];
				for(int j = 1; j <= nY; j++)
					u[j] = abs(i * dx - j * dy) > q ? 0 : w_star * u[j] + u[j - 1];
			}
			pval = 1 - u[nY];
		}
		return new double[] { maxDiv, pval };
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution, two-sided, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist) {
		return kolmogorov_smirnov_test(X, dist, TestKind.TWO_SIDED, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution, exact p-value.
	 * If there are ties, then p-values will be inexact!
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist, TestKind kind) {
		return kolmogorov_smirnov_test(X, dist, kind, true);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution, two-sided.
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist, boolean isExact) {
		return kolmogorov_smirnov_test(X, dist, TestKind.TWO_SIDED, isExact);
	}

	/**
	 * Compute the Kolmogorov-Smirnov test to test between X and a known reference distribution.
	 * 
	 * @param X an array with length of nX
	 * @param dist reference distribution
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @param isExact whether the p-value should be computed with the exact method or not (takes a long time). If there are ties, this option is ignored.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kolmogorov_smirnov_test(double[] X, GenericDistribution dist, TestKind kind, boolean isExact) {
		int n = X.length;
		double[] sortedX = new double[n];
		System.arraycopy(X, 0, sortedX, 0, n);
		sort(sortedX);
		boolean hasTies = false;
		double maxX = Long.MIN_VALUE, minX = Long.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			if (i > 0 && sortedX[i] == sortedX[i - 1])
				hasTies = true;
			double val = dist.cumulative(sortedX[i]) - (i * 1.0 / n);
			if (maxX < val)
				maxX = val;
			else if (minX > val)
				minX = val;
		}
		double maxDiv = Double.NaN, pval = Double.NaN;
		switch(kind) {
			case TWO_SIDED: maxDiv = max(maxX, 1.0/n - minX); break;
			case LOWER: maxDiv = maxX; break;
			case GREATER: maxDiv = 1.0/n - minX; break;
		}
		pval = hasTies || !isExact ? kolmogorov_smirnov_pvalue_inexact(maxDiv, n) : kolmogorov_smirnov_pvalue_exact(maxDiv, n);
		return new double[] { maxDiv, pval };
	}

	static final double kolmogorov_smirnov_statistic(double[] X, GenericDistribution dist, TestKind kind) {
		int n = X.length;
		double[] sortedX = new double[n];
		System.arraycopy(X, 0, sortedX, 0, n);
		sort(sortedX);
		double maxX = Long.MIN_VALUE, minX = Long.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			double val = dist.cumulative(sortedX[i]) - (i * 1.0 / n);
			if (maxX < val)
				maxX = val;
			else if (minX > val)
				minX = val;
		}
		switch(kind) {
			case TWO_SIDED: return max(maxX, 1.0/n - minX);
			case LOWER: return maxX;
			case GREATER: return 1.0/n - minX;
		}
		return Double.NaN;
	}

	static final double kolmogorov_smirnov_pvalue_inexact(double d, int n) {
		double pval;
		if (d >= 1)
			pval = 0;
		else if (d <= 0)
			pval = 1;
		else {
			int max_j = (int) floor(n * (1 - d));
			double s = 0, invN = 1.0 / n;
			for (int j = 0; j <= max_j; j++)
				s += exp(lchoose(n, j) + (n - j) * log(1 - d - j * invN) + (j - 1) * log(d + j * invN));
			pval = d * s;
		}
		return pval;
	}

	static final double kolmogorov_smirnov_pvalue_exact(double d, int n) {
		int
			k = (int) (n * d) + 1,
			m = 2 * k - 1,
			mm = m * m;
		double
			h = k - n * d,
			H[] = new double[mm],
			Q[] = new double[mm];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < m; j++)
				H[i * m + j] = i - j + 1 < 0 ? 0 : 1;

		for(int i = 0; i < m; i++) {
			H[i * m] -= pow(h, i + 1);
			H[(m - 1) * m + i] -= pow(h, (m - i));
		}
		H[(m - 1) * m] += ((2 * h - 1 > 0) ? pow(2 * h - 1, m) : 0);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < m; j++)
				if(i - j + 1 > 0)
					for(int g = 1; g <= i - j + 1; g++)
						H[i * m + j] /= g;
		int eH = 0;
		double eQ = m_power(H, eH, Q, 0, m, n);
		double s = Q[(k - 1) * m + k - 1];
		for(int i = 1; i <= n; i++) {
			s = s * i / n;
			if(s < 1e-140) {
				s *= 1e140;
				eQ -= 140;
			}
		}
		s *= pow(10., eQ);
		return 1 - s;
	}

	/**
	 * Helper function for Kolmogorov-Smirnov
	 * @param A
	 * @param eA
	 * @param V
	 * @param eV
	 * @param m
	 * @param n
	 * @return result of power series
	 */
	private static final int m_power(double[] A, int eA, double[] V, int eV, int m, int n) {
		double[] B = new double[m * m];
		int eB;

		if(n == 1) {
			for (int i = 0; i < m * m; i++)
				V[i] = A[i];
			return eA;
		}
		eV = m_power(A, eA, V, eV, m, n / 2);
		m_multiply(V, V, B, m);
		eB = 2 * eV;
		if((n & 1) == 0) {
			for (int i = 0; i < m * m; i++)
				V[i] = B[i];
			eV = eB;
		} else {
			m_multiply(A, B, V, m);
			eV = eA + eB;
	    }
		int mdiv2 = m / 2;
		if(V[mdiv2 * m + mdiv2] > 1e140) {
			for (int i = 0; i < m * m; i++)
				V[i] = V[i] * 1e-140;
			eV += 140;
		}
		return eV;
	}

	/**
	 * Helper function for Kolmogorov-Smirnov
	 * @param A
	 * @param B
	 * @param C
	 * @param m multiplication of a vector-shaped matrix
	 */
	private static final void m_multiply(double[] A, double[] B, double[] C, int m) {
		for (int i = 0; i < m; i++)
			for (int j = 0; j < m; j++) {
				double s = 0.0;
				for (int k = 0; k < m; k++)
					s+= A[i * m + k] * B[k * m + j];
				C[i * m + j] = s;
			}
	}

	/**
	 * Return the two-sided test of Ansari-Bradley.
	 * @param x the original x
	 * @param y the original y
	 * @param force_exact Set to true if you want exact answer. The default behavior is that
	 * if there are ties or either the length of x or the length of y is at least 50.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] ansari_bradley_test(double[] x, double[] y, boolean force_exact) {
		return ansari_bradley_test(x, y, force_exact, TestKind.TWO_SIDED);
	}

	/**
	 * Ansari-Bradley test.
	 * @param x the original x
	 * @param y the original y
	 * @param force_exact Set to true if you want exact answer. The default behavior is that
	 * if there are ties or either the length of x or the length of y is at least 50.
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] ansari_bradley_test(double[] x, double[] y, boolean force_exact, TestKind kind) {
		int nx = x.length, ny = y.length;
		double N = nx + ny;
		double[] r = rank(c(x, y));
		boolean has_ties = unique(r).length != r.length;
		boolean large_xy = nx >= 50 || ny >= 50;
		double h = 0;
		r = pmin(r, vmin(N + 1, r));
		for (int i = 0; i < nx; i++)
			h += r[i];
		if (force_exact || (!has_ties && !large_xy)) {
			switch(kind) {
				case TWO_SIDED:
					double limit = (nx + 1) * (nx + 1) / 4 + (nx * ny / 2) / 2.0;
					double p = h > limit ? 1 - Ansari.cumulative((int) h - 1, nx, ny)
						: Ansari.cumulative((int) h, nx, ny);
					p = min(2 * p, 1);
					return new double[] {h, p};
				case LOWER:
					p = Ansari.cumulative((int) (h - 1), nx, ny, false);
					return new double[] {h, p};
				case GREATER:
					p = Ansari.cumulative((int) h, nx, ny, true);
					return new double[] {h, p};
			}
			throw new RuntimeException(); // Should never happen
		}
		// Has ties or large xy: Inexact match
		sort(r);
		double[][] rle = rle(r);
		double denom = 16 * N * (N - 1), Np1Sq = N + 1, Np2 = (N+2);
		Np1Sq *= Np1Sq;
		double sigma = 16 * sum(vtimes(vtimes(rle[0], rle[0]), rle[1]));
		sigma = N % 2 == 0 ? sqrt(nx * ny * (sigma - N*Np2*Np2) / denom)
			: sqrt(nx * ny * (sigma*N - Np1Sq*Np1Sq) / (denom * N));
		double z = N % 2 == 0 ? h - nx * Np2 / 4 : h - nx * Np1Sq / (4 * N);
		double p = Normal.cumulative(z/sigma, 0, 1, true, false);
		p = 2 * min(p, 1-p);
		return new double[] {h, p};
	}

	/**
	 * Performs Mood's two-sample test for a difference in scale parameters. Two-sided test. 
	 * @param x
	 * @param y
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] mood_test(double[] x, double[] y) {
		return mood_test(x, y, TestKind.TWO_SIDED);
	}
	/**
	 * Performs Mood's two-sample test for a difference in scale parameters. 
	 * @param x
	 * @param y
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] mood_test(double[] x, double[] y, TestKind kind) {
		int nx = x.length, ny = y.length;
		double N = nx + ny;
		if (N < 3)
			throw new RuntimeException("Not enough observations");
		double E = nx * (N * N - 1.) / 12.;
		double v = (1./180.) * nx * ny * (N + 1) * (N + 2) * (N - 2);
		double[] z = c(x, y);
		boolean has_ties = unique(z).length != z.length;
		double T = 0;
		double con = (N + 1.0)/ 2.;
		if (!has_ties) {
			double[] r = rank(z);
			for (int i = 0; i < nx; i++) {
				double val = r[i] - con;
				T += val * val;
			}
		} else {
			double[] u = unique(z);
			sort(u);
			int[] a = tabulate(match(x, u), u.length);
			int[] t = tabulate(match(z, u), u.length);
			double[] p = vmin(colon(1., z.length), con);
			p = cumsum(vsq(p));
			double sum = 0;
			for (int i = 0; i < u.length; i++) {
				double ti = t[i], NmtiSq = N - t[i], tsq = ti * ti;
				NmtiSq *= NmtiSq;
				sum += ti * (tsq - 1) * (tsq - 4 + 15 * NmtiSq);
			}
			v = v - (nx * ny) / (180. * N * (N - 1)) * sum;
			double[] temp = diff(c(new double[] {0.}, index_min1(p, cumsum(t))));
			for (int i = 0; i < t.length; i++)
				T += a[i] * temp[i] / t[i];
		}
		double zz = (T - E) / sqrt(v);
		double p = Normal.cumulative(zz, 0, 1, kind != TestKind.GREATER, false);
		return new double[] {zz, kind == TestKind.TWO_SIDED ? 2 * min(p, 1 - p) : p};
	}

	/**
	 * Performs an F test to compare the variances of two samples from normal populations. Ratio is set to 1.0. 
	 * @param x
	 * @param y
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] var_test(double[] x, double[] y, TestKind kind) {
		return var_test(x, y, 1, kind);
	}

	/**
	 * Performs an F test to compare the variances of two samples from normal populations. 
	 * @param x
	 * @param y
	 * @param ratio the hypothesized ratio of the population variances of x and y.
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] var_test(double[] x, double[] y, double ratio, TestKind kind) {
		double stat = (var(x) / var(y)) / ratio;
		double p = Double.NaN;
		switch (kind) {
			case TWO_SIDED:
				p = F.cumulative(stat, x.length - 1, y.length - 1, true, false);
				p = 2 * min(p, 1 - p);
				break;
			case GREATER:
				p = F.cumulative(stat, x.length - 1, y.length - 1, false, false);
				break;
			case LOWER:
				p = F.cumulative(stat, x.length - 1, y.length - 1, true, false);
				break;
		}
		return new double[] {stat, p};
	}

	/**
	 * One-sample Wilcoxon test. Test whether the vector of x is != mu
	 * @param x
	 * @param mu
	 * @param correction set to true if continuity correction is desired. Only matters
	 * if x has zeroes or ties
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] wilcoxon_test(double[] x, double mu, boolean correction, TestKind kind) {
		boolean has_zeroes = false;
		int n_nonzero = 0;
		Set<Double> seen_set = new HashSet<Double>();
		for (int i = 0; i < x.length; i++) {
			if (x[i] == mu) {
				has_zeroes = true;
			} else {
				n_nonzero++;
				seen_set.add(x[i]);
			}
		}
		double[] new_x = new double[n_nonzero];
		for (int i = 0, j = 0; i < x.length; i++)
			if (x[i] != mu)
				new_x[j++] = x[i] - mu;
		x = new_x;
		boolean has_ties = seen_set.size() != x.length;
		double[] r = rank(vabs(x));
		int n = r.length;
		double limit = n * (n + 1) / 4.0, v = 0, p = Double.NaN;
		for (int i = 0; i < n; i++)
			v += r[i];
		if (!has_ties && !has_zeroes) {
			SignRank sr = new SignRank(n);
			switch (kind) {
				case TWO_SIDED:
					p = v > limit ? sr.cumulative(v - 1, false, false) : sr.cumulative(v);
					p = min(2*p, 1);
					break;
				case GREATER:
					p = sr.cumulative(v - 1, false, false);
					break;
				case LOWER:
					p = sr.cumulative(v);
					break;
			}
		} else {
			double sigma = 0;
			for (int nties : table(r).values())
				sigma += (nties * nties * nties - nties);
			sigma = sqrt(limit * (2 * n + 1) / 6 - sigma / 48);
			v = v - limit;
			double cor = 0;
			if (correction) {
				switch (kind) {
					case TWO_SIDED:
						cor = signum(v) * 0.5; break;
					case GREATER:
						cor = 0.5; break;
					case LOWER:
						cor = -0.5; break;
				}
			}
			v = (v - cor) / sigma;
			switch (kind) {
				case TWO_SIDED:
					p = 2 * min (Normal.cumulative_standard(v),
						Normal.cumulative(v, 0, 1, false, false));
					break;
				case GREATER:
					p = Normal.cumulative(v, 0, 1, false, false); break;
				case LOWER:
					p = Normal.cumulative_standard(v); break;
			}
		}
		return new double[] {v, p};
	}

	/**
	 * Mann-Whitney-U test
	 * @param x
	 * @param y
	 * @param mu
	 * @param correction set to true if continuity correction is desired. Only matters
	 * then there are ties
	 * @param paired set to true for paired test (which reduces to Wilcoxon test)
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] mann_whitney_u_test(double[] x, double[] y, double mu, boolean correction,
		boolean paired, TestKind kind) {
		int nx = x.length, ny = y.length, n = nx + ny;
		if (paired)
			return wilcoxon_test(vmin(x, y), mu, correction, kind);
		double[] r = mu == 0 ? rank(c(x, y)) : rank(c(vmin(x, mu), y));
		double w = -nx * (nx + 1) / 2, p = Double.NaN, limit = nx * ny / 2.0;
		for (int i = 0; i < nx; i++)
			w += r[i];
		boolean has_ties = unique(r).length != r.length;
		if (!has_ties) {
			Wilcoxon wilcox = new Wilcoxon(nx, ny);
			switch (kind) {
				case TWO_SIDED:
					p = w > limit ? wilcox.cumulative(w - 1, false, false) : wilcox.cumulative(w);
					p = min(2*p, 1);
					break;
				case GREATER:
					p = wilcox.cumulative(w - 1, false, false);
					break;
				case LOWER:
					p = wilcox.cumulative(w);
					break;
			}
		} else {
			double sigma = 0;
			for (int nties : table(r).values())
				sigma += (nties * nties * nties - nties);
			sigma = sqrt((limit / 6) * ((n + 1) - sigma / (n * (n + 1))));
			w = w - limit;
			double cor = 0;
			if (correction) {
				switch (kind) {
					case TWO_SIDED:
						cor = signum(w) * 0.5; break;
					case GREATER:
						cor = 0.5; break;
					case LOWER:
						cor = -0.5; break;
				}
			}
			w = (w - cor) / sigma;
			switch (kind) {
				case TWO_SIDED:
					p = 2 * min (Normal.cumulative_standard(w),
						Normal.cumulative(w, 0, 1, false, false));
					break;
				case GREATER:
					p = Normal.cumulative(w, 0, 1, false, false); break;
				case LOWER:
					p = Normal.cumulative_standard(w); break;
			}
		}
		return new double[] {w, p};
	}

	/**
	 * One-sample t-test
	 * @param x
	 * @param mu
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] t_test(double[] x, double mu, TestKind kind) {
		int nx = x.length;
		double stderr = sqrt(var(x)/nx);
		nx--;
		double t = (mean(x) - mu) / stderr, p = Double.NaN;
		switch (kind) {
			case TWO_SIDED: p = 2 * T.cumulative(-abs(t), nx, true, false); break;
			case GREATER: p = T.cumulative(t, nx, false, false); break;
			case LOWER: p = T.cumulative(t, nx, true, false); break;
		}
		return new double[] {t, p};
	}

	/**
	 * Paired t-test
	 * @param x
	 * @param y
	 * @param mu
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] t_test_paired(double[] x, double[] y, double mu, TestKind kind) {
		return t_test(vmin(x, y), mu, kind);
	}

	/**
	 * Two sample t-test
	 * @param x
	 * @param y
	 * @param mu
	 * @param pool_var set to true if the variance should be pooled. Only matters when paired == false
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] t_test(double[] x, double[] y, double mu, boolean pool_var, TestKind kind) {
		int nx = x.length, ny = y.length;
		double stderr, df;
		if (pool_var) {
			df = nx + ny - 2;
			stderr = 0;
			if (nx > 1) stderr += (nx - 1) * var(x);
			if (ny > 1) stderr += (ny - 1) * var(y);
			stderr = sqrt((stderr/df) * (1.0/nx + 1.0/ny));
		} else {
			double sx = var(x)/nx, sy = var(y)/ny;
			stderr = sx + sy;
			df = stderr*stderr / (sx*sx/(nx-1) + sy*sy/(ny-1));
			stderr = sqrt(stderr);
		}
		double t = (mean(x) - mean(y) - mu) / stderr, p = Double.NaN;
		switch (kind) {
			case TWO_SIDED: p = 2 * T.cumulative(-abs(t), df, true, false); break;
			case GREATER: p = T.cumulative(t, df, false, false); break;
			case LOWER: p = T.cumulative(t, df, true, false); break;
		}
		return new double[] {t, p};
	}

	/**
	 * Binomial test
	 * @param n_success The number of successes
	 * @param n The total number of trials
	 * @param p Expected probability
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] binomial_test(int n_success, int n, double p, TestKind kind) {
		switch (kind) {
			case TWO_SIDED:
				if (p == 0) {
					p = n_success == 0 ? 1 : 0;
				} else if (p == 1) {
					p = n_success == n ? 1 : 0;
				} else {
					double d = Binomial.density(n_success, n, p, false) * (1 + 1e-7),
						m = n * p;
					if (n_success == m) {
						p = 1;
					} else if (n_success < m){
						int y = 0;
						for (int i = (int) ceil(m); i <= n; i++)
							if (Binomial.density(i, n, p, false) <= d) y++;
						p = Binomial.cumulative(n_success, n, p, true, false) +
							Binomial.cumulative(n - y, n, p, false, false);
					} else {
						int y = 0, mlo = (int) floor(m);
						for (int i = 0; i <= mlo; i++)
							if (Binomial.density(i, n, p, false) <= d) y++;
						p = Binomial.cumulative(y - 1, n, p, true, false) +
							Binomial.cumulative(n_success - 1, n, p, false, false);
					}
				}
				break;
			case GREATER: p = Binomial.cumulative(n_success - 1, n, p, false, false); break;
			case LOWER: p = Binomial.cumulative(n_success, n, p, true, false); break;
		}
		return new double[] {n_success, p};
	}

	/**
	 * Bartlett's test
	 * @param x
	 * @param group an array of group indices. Observation in x that belongs in the same group must have the same index.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] bartlett_test(double[] x, int[] group) {
		int n = x.length;
		if (n != group.length)
			throw new RuntimeException();
		Map<Integer, List<Double>> map = new HashMap<Integer, List<Double>>();
		for (int i = 0; i < n; i++) {
			List<Double> ll = map.get(group[i]);
			if (ll == null) {
				ll = new ArrayList<Double>();
				map.put(group[i], ll);
			}
			ll.add(x[i]);
		}
		int[] unique_group = to_int_array(map.keySet());
		int k = unique_group.length;
		double v_total = 0, sum_recip = 0, sum_n_vlog = 0;
		for (int i = 0; i < k; i++) {
			double[] dbl = to_double_array(map.get(unique_group[i]));
			double var_group = var(dbl);
			int ni = dbl.length - 1;
			v_total += ni * var_group / (n - k); 
			sum_recip += 1.0/ni;
			sum_n_vlog += ni * log(var_group);
		}
		double stat = (((n - k) * log(v_total) - sum_n_vlog) / (1 + (sum_recip - 1.0/(n-k)) / (3*(k-1))));
		double p = ChiSquare.cumulative(stat, k - 1, false, false);
		return new double[] {stat, p};
	}

	/**
	 * Fligner-Killeen test
	 * @param x
	 * @param group an array of group indices. Observation in x that belongs in the same group must have the same index.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] fligner_test(double[] x, int[] group) {
		int n = x.length;
		if (n != group.length)
			throw new RuntimeException();
		Map<Integer, List<Double>> map = new HashMap<Integer, List<Double>>();
		for (int i = 0; i < n; i++) {
			List<Double> ll = map.get(group[i]);
			if (ll == null) {
				ll = new ArrayList<Double>();
				map.put(group[i], ll);
			}
			ll.add(x[i]);
		}
		int[] unique_group = to_int_array(map.keySet());
		int k = unique_group.length;
		int[] cumsum_n_group = new int[k], n_group = new int[k];
		double[] new_x = new double[n];
		for (int i = 0; i < k; i++) {
			double[] dbl = to_double_array(map.get(unique_group[i]));
			int ni = dbl.length;
			cumsum_n_group[i] = ni + (i > 0 ? cumsum_n_group[i-1] : 0);
			n_group[i] = ni;
			double med_group = median(dbl);
			for (int j = 0; j < ni; j++)
				dbl[j] -= med_group;
			System.arraycopy(dbl, 0, new_x, i == 0 ? 0 : cumsum_n_group[i-1], ni);
		}
		new_x = rank(vabs(new_x));
		for (int i = 0; i < new_x.length; i++)
			new_x[i] = Normal.quantile((1 + new_x[i] / (n + 1)) / 2.0, 0, 1, true, false);
		double stat = 0;
		for (int i = 0; i < k; i++) {
			int
				from = i == 0 ? 0 : cumsum_n_group[i - 1],
				ni = n_group[i],
				to = from + ni;
			double sum = 0;
			for (int j = from; j < to; j++)
				sum += new_x[j];
			stat += sum * sum / ni;
		}
		double ma = mean(new_x);
		stat = (stat - n * ma * ma) / var(new_x);
		double p = ChiSquare.cumulative(stat, k - 1, false, false);
		return new double[] {stat, p};
	}

	/**
	 * Kruskal-Wallis test
	 * @param x
	 * @param group an array of group indices. Observation in x that belongs in the same group must have the same index.
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] kruskal_wallis_test(double[] x, int[] group) {
		int n = x.length;
		if (n != group.length)
			throw new RuntimeException();
		Map<Integer, List<Double>> map = new HashMap<Integer, List<Double>>();
		for (int i = 0; i < n; i++) {
			List<Double> ll = map.get(group[i]);
			if (ll == null) {
				ll = new ArrayList<Double>();
				map.put(group[i], ll);
			}
			ll.add(x[i]);
		}
		int[] unique_group = to_int_array(map.keySet());
		int k = unique_group.length;
		int[] cumsum_n_group = new int[k], n_group = new int[k];
		double[] new_x = new double[n];
		for (int i = 0; i < k; i++) {
			double[] dbl = to_double_array(map.get(unique_group[i]));
			int ni = dbl.length;
			cumsum_n_group[i] = ni + (i > 0 ? cumsum_n_group[i-1] : 0);
			n_group[i] = ni;
			System.arraycopy(dbl, 0, new_x, i == 0 ? 0 : cumsum_n_group[i-1], ni);
		}
		double[] r = rank(new_x);
		double stat = 0;
		for (int i = 0; i < k; i++) {
			int
				from = i == 0 ? 0 : cumsum_n_group[i - 1],
				ni = n_group[i],
				to = from + ni;
			double sum = 0;
			for (int j = from; j < to; j++)
				sum += r[j];
			stat += sum * sum / ni;
		}
		double sigma = 0;
		for (int nties : table(r).values())
			sigma += (nties * nties * nties - nties);
		stat = ((12 * stat / (n * (n + 1)) - 3 * (n + 1)) / (1 - sigma / (n*n*n - n)));
		double p = ChiSquare.cumulative(stat, k - 1, false, false);
		return new double[] {stat, p};
	}

	/**
	 * Performs an exact test of a simple null hypothesis about the rate parameter in Poisson distribution
	 * @param num_events number of events.
	 * @param time time base for event count.
	 * @param rate hypothesized rate
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] poisson_test(int num_events, double time, double rate, TestKind kind) {
		if (time < 0 || rate < 0 || num_events < 0) throw new RuntimeException();
		double m = rate * time, p = Double.NaN;
		switch (kind) {
			case TWO_SIDED:
				if (m == 0) {
					p = num_events == 0 ? 1 : 0; break;
				}
				if (num_events == m) {
					p = 1; break;
				}
				double d = Poisson.density(num_events, m, false), fuzz = (1 + 1e-7) * d, y = 0, N;
				if (num_events < m) {
					N = (int) ceil(2 * m - num_events);
					while (Poisson.density(N, m, false) > d) N *= 2;
					for (int i = (int) ceil(m); i <= N; i++)
						if (Poisson.density(i, m, false) <= fuzz)
							y++;
					p = Poisson.cumulative(num_events, m, true, false) +
						Poisson.cumulative(N - y, m, false, false);
				} else {
					N = (int) floor(m);
					for (int i = 0; i < N; i++)
						if (Poisson.density(i, m, false) <= fuzz)
							y++;
					p = Poisson.cumulative(y - 1, m, true, false) +
						Poisson.cumulative(num_events - 1, m, false, false);
				}
				break;
			case GREATER: p = Poisson.cumulative(num_events-1, m, false, false); break;
			case LOWER: p = Poisson.cumulative(num_events, m, true, false); break;
		}
		return new double[] {num_events, p};
	}

	/**
	 * Comparison of Poisson rates
	 * @param num_events1 number of events for the treatment.
	 * @param num_events2 number of events for control.
	 * @param time1 time base for event count for treatment.
	 * @param time2 time base for event count for control.
	 * @param kind the kind of test {LOWER, GREATER, TWO_SIDED}
	 * @return an array of two elements: The first is the test statistic, the second is the p-value
	 */
	public static final double[] poisson_test(int num_events1, int num_events2, double time1, double time2, double r, TestKind kind) {
		if (time1 < 0 || time2 < 0 || r < 0 || num_events1 < 0 || num_events2 < 0) throw new RuntimeException();
		return binomial_test(num_events1, num_events1+num_events2, r * time1 / (r * time1 + time2), kind);
	}

	/**
	 * Two-sample Cramer-Von Mises test
	 * @param X
	 * @param Y
	 * @return statistic
	 */
	static final double cramer_vonmises_statistic(double[] X, double[] Y) {
		int
			nX = X.length,
			nY = Y.length,
			nXY = nX * nY,
			nXPY = nX + nY;
		double[] rank = rank(c(X, Y)),
			rankX = rank(X),
			rankY = rank(Y);
		double sumX = 0, sumY = 0, val;
		for (int i = 0; i < nX; i++) {
			val = rank[i] - rankX[i];
			sumX += val * val;
		}
		for (int i = nX; i < nXPY; i++) {
			val = rank[i] - rankY[i - nX];
			sumY += val * val;
		}
		val = (nX * sumX + nY * sumY) / (nXY * nXPY) - (4*nXY - 1) / (6 * nXPY); // T statistic

		/*
		int gcd = jdistlib.math.MathFunctions.gcd(nX, nY);
		int nL = nX / gcd * nY, nP = nL / nX, nQ = nL / nY;
		double coef = (((1.0/nP) * (1.0/nQ)) / nXPY) / nXPY;
		sumX = sumY = 0;
		for (int i = 0; i < nXPY; i++) {
			sumX += rank[i] < nX ? nP : -nQ;
			sumY += sumX*sumX;
		}
		val = sumY * coef;
		//*/
		return val;
	}

	static final void kstest_example() {
		double x[] = { 1.16004168838821208886714, -1.05547634926559497081655, -1.32072420295666459466588,  0.23915046399456202363965,
			 0.12803724906074620548679,  0.05569133699728501252224, -0.81250026875197078890523, -0.25270560923205648284906,
			 -0.18064235632836450617944, -1.68851736711207101038212,  0.45201765941273730486927, -0.82239514187785234256012,
			 -1.28431746020543480213405, -1.11673394115548996197163,  0.35484674303354840629865,  0.69469717363334693160937,
			  0.80814838465816862811408, -0.21328821452795371227396, -1.27614688227021422228802,  0.70701146687565052939561,
			  0.50751733987764702238366,  0.03272845258879651664241,  0.35783108099085625397606,  0.44208900297507758292426,
			 -0.52069082447125036861024, -0.55763158717776695194601, -0.41633151696193471114071,  0.26769602419784399582880,
			 -0.74987234035877792237557, -0.41904535934255054963060,  0.89257906144225818145799, -0.07190597453573960295969,
			  0.73302279756488808448722, -0.10734082387514076728507,  0.69406629605031111562852, -0.15263137425121245382975,
			 -1.19674895163987238255743,  0.66757167860376609436202, -0.81494857018266397830075,  0.80040931005785931340313,
			  0.80754967242376574088070, -1.56916928352228968179816, -1.35167386137606548857093,  1.16318789818012624515120,
			  0.28936206000057662635072,  0.52290491081517120885991, -0.05762605570118027598081, -0.14176370966120629968366,
			 -0.37619927990739526757480, -1.06736558034971595887441};
		double y[] = { 0.40490121906623244285583, 0.24850796000100672245026, 0.10222866409458220005035, 0.62853105599060654640198,
				0.61955126281827688217163, 0.28701157541945576667786, 0.95149635546840727329254, 0.01087409863248467445374,
				0.63707210076972842216492, 0.61038276972249150276184, 0.09210657258518040180206, 0.64758135029114782810211,
				0.48832037649117410182953, 0.10307366680353879928589, 0.86465166741982102394104, 0.68774600140750408172607,
				0.76253556110896170139313, 0.52018100558780133724213, 0.61665696790441870689392, 0.77783631114289164543152,
				0.89877208555117249488831, 0.83583764336071908473969, 0.92252740426920354366302, 0.83699792949482798576355,
				0.35809992859140038490295, 0.59004115150310099124908, 0.60853263596072793006897, 0.13569264346733689308167,
				0.38345616171136498451233, 0.91171105671674013137817};
		double[] p = kolmogorov_smirnov_test(x, y);
		System.out.println(p[0]);
		System.out.println(p[1]);
		p = kolmogorov_smirnov_test(x, new Normal());
		System.out.println(p[0]);
		System.out.println(p[1]);
	}

	public static final void main(String[] args) {
		kstest_example();
		System.exit(0);
		double[] x = { -1.2315764307891696738295, 0.1076666048919862200828, -0.2507677102611699515577,	0.1865730243313593050836,
			0.7674721840239807635342, -0.1874640529241502207025, 0.1376975996921310230192, 0.3722658431557314684390,
			1.8257862598243677076937, -1.4691239378183402752853
		};
		double[] y = { 2.633833206002905935605, -1.041337574910569774289, -1.081121838223072728624, 2.702460192243479220053,
			1.626548966201278201282, 1.336642538096019183769, 1.075145021293279601338, 1.543056949670002397923,
			-0.085039987328253241472, 1.357930215887039437916
		};
		// Correct answer: T = 0.405, P-value: 0.07656584901166944845397
		System.out.println(cramer_vonmises_statistic(x, y));
	}
}
