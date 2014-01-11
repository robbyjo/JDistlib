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
import jdistlib.Ansari;
import jdistlib.Normal;


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
			double limit = (nx + 1) * (nx + 1) / 4 + (nx * ny / 2) / 2.0;
			double p = h > limit ? 1 - Ansari.cumulative((int) h - 1, nx, ny)
				: Ansari.cumulative((int) h, nx, ny);
			p = min(2 * p, 1);
			return new double[] {h, p};
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
