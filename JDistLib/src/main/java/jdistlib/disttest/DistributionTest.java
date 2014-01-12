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

import static java.lang.Math.*;
import static jdistlib.util.Utilities.*;
import static jdistlib.math.VectorMath.*;
import static jdistlib.disttest.Utils.calculate_ecdf;

import java.util.HashSet;
import java.util.Set;

import jdistlib.Ansari;
import jdistlib.F;
import jdistlib.Normal;
import jdistlib.SignRank;
import jdistlib.T;
import jdistlib.Wilcoxon;


/**
 * Comparing two distributions
 * @author Roby Joehanes
 *
 */
public class DistributionTest {
	/**
	 * Compute the Kolmogorov-Smirnov test to test between two distribution.
	 * Note: I don't multiply the D score with sqrt(nX*nY / (nX + nY)), which
	 * is needed for P-value computation
	 * 
	 * @param X an array with length of nX
	 * @param Y an array with length of nY
	 * @return K-S statistics
	 */
	public static final double kolmogorov_smirnov_statistic(double[] X, double[] Y) {
		int
			nX = X.length,
			nY = Y.length,
			idxX = 0,
			idxY = 0;
		double
			sortedX[] = new double[nX],
			sortedY[] = new double[nY],
			maxDiv = 0;
		System.arraycopy(X, 0, sortedX, 0, nX);
		System.arraycopy(Y, 0, sortedY, 0, nY);
		sort(sortedX);
		sort(sortedY);
	
		// Pathological case
		if (sortedX[nX - 1] < sortedY[0] || sortedY[nY - 1] < sortedX[0])
			return 1.0;
		// Scan for duplicate values
		double
			cdfX[] = calculate_ecdf(sortedX),
			cdfY[] = calculate_ecdf(sortedY),
			pX = 0,
			pY = 0,
			div = 0;
		while (idxX < nX && idxY < nY) {
			double
				x = sortedX[idxX],
				y = sortedY[idxY];
			if (y < x) {
				pY = cdfY[idxY];
				idxY++;
			} else if (y > x) {
				pX = cdfX[idxX];
				idxX++;
			} else {
				pX = cdfX[idxX];
				pY = cdfY[idxY];
				idxX++; idxY++;
			}
			div = abs(pX - pY);
			//div = abs(idxX / ((double) nX) - (idxY - 1.0) / nY);
			if (div > maxDiv)
				maxDiv = div;
		}
		return maxDiv;
	}

	/**
	 * Compute the P-value out of the D-score produced by <tt>kolmogorov_smirnov_statistic</tt>.
	 * 
	 * @param maxDiv
	 * @param lengthX
	 * @param lengthY
	 * @return p-value
	 */
	public static final double kolmogorov_smirnov_pvalue(double maxDiv, int lengthX, int lengthY) {
		/*
		Set<Double> set = new HashSet<Double>();
		for (double x: X)
			set.add(x);
		m = set.size();
		set.clear();
		for (double y: Y)
			set.add(y);
		n = set.size();
		set.clear();
		set = null;
		//*/
	
		if (lengthX > lengthY) {
			int temp = lengthY;
			lengthY = lengthX;
			lengthX = temp;
		}
		double
			q = floor(maxDiv * lengthX * lengthY - 1e-7) / (lengthX * lengthY),
			u[] = new double[lengthY + 1],
			md = lengthX,
			nd = lengthY;
	
		for (int j = 0; j <= lengthY; j++)
			u[j] = (j / nd) > q ? 0: 1;
		for(int i = 1; i <= lengthX; i++) {
			double w = (double)(i) / ((double)(i + lengthY));
			u[0] = (i / md) > q ? 0 : w * u[0];
			for(int j = 1; j <= lengthY; j++)
				u[j] = abs(i / md - j / nd) > q ? 0 : w * u[j] + u[j - 1];
		}
		return 1 - u[lengthY];
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
		double[] r = rank(c(vmin(x, mu), y));
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
	 * Two-sample Cramer-Von Mises test
	 * @param X
	 * @param Y
	 * @return statistic
	 */
	private static final double cramer_vonmises_statistic(double[] X, double[] Y) {
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

	public static final void main(String[] args) {
		double[] x = new double[] {
			-1.2315764307891696738295, 0.1076666048919862200828, -0.2507677102611699515577,	0.1865730243313593050836,
			0.7674721840239807635342, -0.1874640529241502207025, 0.1376975996921310230192, 0.3722658431557314684390,
			1.8257862598243677076937, -1.4691239378183402752853
		};
		double[] y = new double[] {
			2.633833206002905935605, -1.041337574910569774289, -1.081121838223072728624, 2.702460192243479220053,
			1.626548966201278201282, 1.336642538096019183769, 1.075145021293279601338, 1.543056949670002397923,
			-0.085039987328253241472, 1.357930215887039437916
		};
		// Correct answer: T = 0.405, P-value: 0.07656584901166944845397
		System.out.println(cramer_vonmises_statistic(x, y));
	}
}
