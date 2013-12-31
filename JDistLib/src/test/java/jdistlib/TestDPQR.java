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

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;
import static jdistlib.MathFunctions.round;
import static jdistlib.MathFunctions.gammafn;
import static jdistlib.Constants.DBL_EPSILON;

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
		return !Double.isNaN(a+b) && (a == b || abs(a - b) < tol);
	}

	static final void printBool(boolean b) {
		System.out.println(b ? "[1] TRUE" : "[1] FALSE");
	}

	static final void printBool(boolean... b) {
		if (b == null || b.length == 0) return;
		System.out.print("[1]");
		for (int i = 0; i < b.length; i++)
			System.out.print(b[i] ? " TRUE" : " FALSE");
		System.out.println();
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
			printBool(success1);
			printBool(success2);
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
			printBool(cur_success);
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
			printBool(cur_success);
		}
		System.out.println();
		boolean b = !isEqual(NegBinomial.cumulative(1, 0.9, 0.5, true, false), 0.777035760338812)
				|| !isEqual(NegBinomial.cumulative(3, 0.9, 0.5, true, false), 0.946945347071519);
		if (b) success = false;
		printBool(!b);
		return success;
	}

	@Test
	public static final boolean test_poisson() {
		System.out.println("##__ 5. Poisson __");
		boolean success = true;

		boolean cur_success = true;
		for (int i = 0; i <= 5; i++) {
			double dpois = Poisson.density(i, 0, false);
			if (!isEqual(dpois, i == 0 ? 1 : 0)) {
				System.err.println(String.format("Error: dpois = %g", dpois));
				success = cur_success = false;
			}
		}
		printBool(cur_success);

		cur_success = true;
		for (int i = 0; i <= 5; i++) {
			double dpois = Poisson.density(i, 0, true);
			if (!isEqual(dpois, i == 0 ? 0 : Double.NEGATIVE_INFINITY)) {
				System.err.println(String.format("Error: log(dpois) = %g", dpois));
				success = cur_success = false;
			}
		}
		printBool(cur_success);

		int n1 = 20, n2 = 16;
		for (int i = 0; i < n1; i++) {
			double lambda = Exponential.random(1, random);
			for (int j = 0; j < n2; j++) {
				int k = (int) Poisson.random(lambda, random);
				double prev_dpois = 0;
				cur_success = true;
				for (int j2 = 0; j2 <= k; j2++) {
					double comp_pchisq = 1 - ChiSquare.cumulative(2 * lambda, 2 * j2+ 2, true, false);
					double dpois = prev_dpois + Poisson.density(j2, lambda, false);
					prev_dpois = dpois;
					if (!isEqual(comp_pchisq, dpois)) {
						System.err.println(String.format("Error: comp. pchisq = %g, dpois = %g", comp_pchisq, dpois));
						success = cur_success = false;
					}
					double ppois = Poisson.cumulative(j2, lambda, true, false);
					if (!isEqual(ppois, dpois)) {
						System.err.println(String.format("Error: ppois = %g, dpois = %g", ppois, dpois));
						success = cur_success = false;
					}
					ppois = Poisson.cumulative(j2, lambda, false, false);
					if (!isEqual(ppois, 1 - dpois)) {
						System.err.println(String.format("Error: upper ppois = %g, dpois = %g", ppois, dpois));
						success = cur_success = false;
					}
				}
			}
		}
		return success;
	}

	@Test
	public static final boolean test_signrank() {
		System.out.println("##__ 6. SignRank __");
		boolean success = true;
		for (int i = 0; i < 32; i++) {
			int n = (int) Poisson.random(8, random);
			SignRank d = new SignRank(n);
			double prev_dsignrank = 0;
			for (int x = -1; x <= n + 4; x++) {
				double psignrank = d.cumulative(x, true, false);
				double dsignrank = prev_dsignrank + d.density(x, false);
				prev_dsignrank = dsignrank;
				if (!isEqual(psignrank, dsignrank)) {
					System.err.println(String.format("Error: psignrank = %g, dsignrank = %g", psignrank, dsignrank));
					success = false;
				}
			}
		}
		return success;
	}

	@Test
	public static final boolean test_wilcox() {
		System.out.println("##__ 7. Wilcoxon (symmetry & cumulative) __");
		boolean success = true, is_sym = true;
		for (int i = 0; i < 5; i++) {
			int n = (int) Poisson.random(6, random);
			for (int j = 0; j < 15; j++) {
				int m = (int) Poisson.random(8, random);
				Wilcoxon d = new Wilcoxon(n, m);
				Wilcoxon d_sym = new Wilcoxon(m, n);
				int limit = n*m + 1;
				double cum_dwilcox = 0;
				for (int x = -1; x <= limit; x++) {
					double pwilcox = d.cumulative(x, true, false);
					double dwilcox = d.density(x, false);
					double dwilcox_sym = d_sym.density(x, false);
					cum_dwilcox += dwilcox;
					if (!isEqual(pwilcox, cum_dwilcox)) {
						System.err.println(String.format("Error: pwilcox = %g, dwilcox = %g", pwilcox, cum_dwilcox));
						success = false;
					}
					is_sym = is_sym & isEqual(dwilcox, dwilcox_sym);
				}
			}
		}
		printBool(is_sym);
		return success;
	}

	@Test
	public static final boolean test_gamma() {
		System.out.println("##__ Gamma Density (incl. central chi^2) __");
		boolean success = true;
		for (int i = 0; i < 100; i++) {
			double x = round(Gamma.random(2, 1, random), 2);
			for (int j = 0; j < 30; j++) {
				double sh = round(LogNormal.random(0, 1, random), 2);
				double Ga = gammafn(sh);
				for (int k = 0; k < 30; k++) {
					double sig = round(LogNormal.random(0, 1, random), 2);
					double d1 = Gamma.density(x, sh, sig, false);
					double d2 = Gamma.density(x/sig, sh, 1, false) / sig;
					if (!isEqual(d1, d2)) {
						System.err.println(String.format("Error: scaled dgamma = %g, manually scaled dgamma = %g", d1, d2));
						success = false;
					}
					double d3 = 1.0 / (Ga * pow(sig, sh)) * pow(x, sh - 1.0) * exp(-x / sig);
					if (!isEqual(d1, d3)) {
						System.err.println(String.format("Error: scaled dgamma = %g, manually comp dgamma = %g", d1, d3));
						success = false;
					}
				}
			}
		}

		double Inf = Double.POSITIVE_INFINITY, xMax = Double.MAX_VALUE;
		printBool(Gamma.cumulative(1, Inf, Inf, true, false) == 0);
		printBool(Double.isNaN(Gamma.cumulative(Inf, 1, Inf, true, false))
			&& Double.isNaN(Gamma.cumulative(Inf, Inf, Inf, true, false)));
		double p = Gamma.cumulative(Inf, 1, xMax, true, false);
		if (!isEqual(p, 1)) {
			System.err.println(String.format("Error: pgamma(Inf, 1, xMax) = %g", p));
			success = false;
		}
		p = Gamma.cumulative(xMax, 1, Inf, true, false);
		if (!isEqual(p, 0)) {
			System.err.println(String.format("Error: pgamma(xMax, 1, Inf) = %g", p));
			success = false;
		}

		double[] scLrg = new double[] { 2, 100, 1e299, 1e300, 1e301, 1e302, 1e307, xMax, Inf };
		// Supplied values by R authors seem to be lacking precision
		//double[] ans = new double[] {0, 0, -0.000499523968713701, -1.33089326820406,
		//		-5.36470502873211, -9.91015144019122, -32.9293385491433, -38.707517174609, Double.NEGATIVE_INFINITY
		//};
		double[] ans = new double[] {0, 0, -0.0004995239687137007075432, -1.330893268204054846748, -5.364705028732111635748,
			-9.910151440191221183795, -32.9293385491432459844, -38.70751717460898788659, Double.NEGATIVE_INFINITY
		};
		for (int i = 0; i < scLrg.length; i++) {
			p = Gamma.cumulative(1e300, 2, scLrg[i], true, true);
			if (!isEqual(p, ans[i], 2e-15)) {
				System.err.println(String.format("Error: pgamma(1e300, 2, %g) = %3.18g. Correct answer = %3.18g", scLrg[i], p, ans[i]));
				success = false;
			}
		}

		p = 7e-4; double df = 0.9, lim = 1e-15;
		double[] d = new double[] {
		abs(1 - ChiSquare.cumulative(ChiSquare.quantile(p, df, true, false), df, true, false) / p),
		abs(1 - ChiSquare.cumulative(ChiSquare.quantile(1-p, df, false, false), df, false, false) / (1-p)),
		abs(1 - ChiSquare.cumulative(ChiSquare.quantile(log(p), df, true, true), df, true, true) / log(p)),
		abs(1 - ChiSquare.cumulative(ChiSquare.quantile(log1p(-p), df, false, true), df, false, true) / log1p(-p))
		};
		printBool(d[0] < lim, d[1] < lim, d[2] < lim, d[3] < lim);
		return success;
	}

	public static final void main(String[] args) {
		test_binom();
		test_geom();
		test_hyper();
		test_negbin();
		test_poisson();
		test_signrank();
		test_wilcox();
		test_gamma();
	}
}
