/*
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
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
package jdistlib;

import jdistlib.rng.QMersenneTwister;
import org.junit.Test;

import static java.lang.Math.*;
import static jdistlib.Constants.*;

/**
 * Ported tests/d-p-q-r-tests.R plus some more.
 * @author Roby Joehanes
 *
 */
public class TestDPQR {
	static final QMersenneTwister random = new QMersenneTwister(123L);
	static final double defaultNumericalError = DBL_EPSILON * 64;

	static final double rErr(double approx, double truval) {
		return rErr(approx, truval, 1e-30);
	}

	static final double rErr(double approx, double truval, double eps) {
		return abs(truval) >= eps ? 1 - approx / truval : approx - truval;
	}

	static final boolean isEqual(double a, double b) {
		return isEqual(a, b, defaultNumericalError);
	}

	static final boolean isEqual(double a, double b, double tol) {
		return !Double.isNaN(a+b) && (abs(a - b) < tol);
	}

	@Test
	public static final boolean test_binom() {
		System.out.println("##__ 1. Binomial __");
		int n0 = 50, n1 = 16, n2 = 20 + 2, n3 = 8;
		boolean success = true;
		for (int i = 0; i < n1; i++) {
			int n = (int) Binomial.random(2*n0, 0.4, random);
			System.out.print("n=" + n + ": ");
			for (int j = 0; j < n2; j++) {
				double p = j == 0 ? 0 : j == 1 ? 1 : Beta.random(2, 4, random);
				System.out.print(".");
				for (int j2 = 0; j2 < n3; j2++) {
					int k = (int) Binomial.random(n, random.nextDouble(), random);
					double prev_dbinom = 0;
					for (int l = 0; l <= k; l++) {
						double pbinom = Binomial.cumulative(l, n, p, true, false);
						double dbinom = prev_dbinom + Binomial.density(l, n, p, false);
						prev_dbinom = dbinom;
						if (!isEqual(pbinom, dbinom)) {
							System.err.println(String.format("Error: pbinom = %g, cumsum(dbinom) = %g", pbinom, dbinom));
							success = false;
						}
					}
					double f_eq = k == n || p == 0 ? 1 : F.cumulative((k+1.0)/(n-k)*(1.0-p)/p, 2*(n-k), 2*(k+1), true, false);
					if (!isEqual(f_eq, prev_dbinom)) {
						System.err.println(String.format("Error: pf = %g, sum(dbinom) = %g", f_eq, prev_dbinom));
						success = false;
					}
				}
			}
			System.out.println();
		}
		return success;
	}

	@Test
	public static final boolean test_geom() {
		System.out.println("##__ 2. Geometric __");
		boolean success = true;
		int n = 15;
		double from = 1e-10, to = 1;
		for (int i = 0; i < n; i++) {
			double pr = from + (to - from) * i / (n - 1.0), q = 1 - pr, prev_dg = 0;
			boolean success1 = true, success2 = true;
			for (int j = 0; j <= 10; j++) {
				double dg = Geometric.density(j, pr, false);
				double direct = pr * pow(q, j);
				if (!isEqual(dg, direct)) {
					System.err.println(String.format("Error: dgeom = %g, direct = %g", dg, direct));
					success = success1 = false;
				}
				dg += prev_dg;
				double pgeom = Geometric.cumulative(j, pr, true, false);
				if (!isEqual(dg, pgeom)) {
					System.err.println(String.format("Error: cumsum(dgeom) = %g, pgeom = %g", dg, pgeom));
					success = success2 = false;
				}
				prev_dg = dg;
			}
			System.out.println(success1 ? "[1] TRUE" : "[1] FALSE");
			System.out.println(success2 ? "[1] TRUE" : "[1] FALSE");
		}
		return success;
	}

	@Test
	public static final boolean test_hyper() {
		System.out.println("##__ 3. Hypergeometric __");
		boolean success = true;
		int m = 10, n = 7;
		for (int k = 2; k <= m; k++) {
			double prev_dhyper = 0;
			boolean cur_success = true;
			for (int j = 0; j <= k+1; j++) {
				double phyper = HyperGeometric.cumulative(j, m, n, k, true, false);
				double dhyper = prev_dhyper + HyperGeometric.density(j, m, n, k, false);
				prev_dhyper = dhyper;
				if (!isEqual(phyper, dhyper)) {
					System.err.println(String.format("Error: phyper = %g, cumsum(dhyper) = %g", phyper, dhyper));
					success = cur_success = false;
				}
			}
			System.out.println(cur_success ? "[1] TRUE" : "[1] FALSE");
		}
		return success;
	}

	@Test
	public static final boolean test_negbin() {
		System.out.println("##__ 4. Negative Binomial __");
		boolean success = true;
		for (int i = 8; i <= 20; i++) {
			double size = i / 10.0;
			double prev_dnbinom = 0;
			boolean cur_success = true;
			for (int j = 0; j <= 7; j++) {
				double dnbinom = prev_dnbinom + NegBinomial.density(j, size, 0.5, false);
				double pnbinom = NegBinomial.cumulative(j, size, 0.5, true, false);
				prev_dnbinom = dnbinom;
				if (!isEqual(dnbinom, pnbinom)) {
					System.err.println(String.format("Error: cumsum(dnbinom) = %g, pnbinom = %g", dnbinom, dnbinom));
					success = cur_success = false;
				}
			}
			System.out.println(cur_success ? "[1] TRUE" : "[1] FALSE");
		}
		System.out.println();
		if (!isEqual(NegBinomial.cumulative(1, 0.9, 0.5, true, false), 0.777035760338812)
			|| !isEqual(NegBinomial.cumulative(3, 0.9, 0.5, true, false), 0.946945347071519)) {
			success = false;
			System.out.println("[1] FALSE");
		} else {
			System.out.println("[1] TRUE");
		}
		return success;
	}

	public static final void main(String[] args) {
		test_binom();
		test_geom();
		test_hyper();
		test_negbin();
	}
}
