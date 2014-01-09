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

import jdistlib.math.VectorMath;
import jdistlib.rng.QMersenneTwister;
import jdistlib.rng.QRandomEngine;

import org.junit.Test;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static jdistlib.math.Constants.DBL_EPSILON;
import static jdistlib.math.Constants.DBL_MAX;
import static jdistlib.math.MathFunctions.gammafn;
import static jdistlib.math.MathFunctions.isInfinite;
import static jdistlib.math.MathFunctions.round;
import static jdistlib.math.VectorMath.*;
import static jdistlib.util.Utilities.*;

/**
 * Ported tests/d-p-q-r-tests.R plus some more.
 * @author Roby Joehanes
 *
 */
public class TestDPQR {
	static QRandomEngine random = new QMersenneTwister(123L);
	static final double defaultNumericalError = DBL_EPSILON * 64;

	public static final void setRandomEngine(QRandomEngine rng) {
		random = rng;
	}

	static final double rErr(double approx, double truval) {
		return rErr(approx, truval, 1e-30);
	}

	static final double rErr(double approx, double truval, double eps) {
		return abs(truval) >= eps ? 1 - approx / truval : approx - truval;
	}

	public static final boolean isEqual(double a, double b) {
		return isEqual(a, b, defaultNumericalError);
	}

	public static final boolean isEqual(double a, double b, double tol) {
		return (Double.isNaN(a) && Double.isNaN(b)) || (a == b || abs(a - b) <= tol);
	}

	public static final boolean isEqualScaled(double a, double b) {
		return isEqualScaled(a, b, defaultNumericalError);
	}

	public static final boolean isEqualScaled(double a, double b, double tol) {
		return (Double.isNaN(a) && Double.isNaN(b)) || (a == b || abs(a - b)/(Double.isNaN(a) ? 0 : a) <= tol);
	}

	public static final boolean allEqual(double[] a, double[] b, double tol) {
		int n = a.length;
		if (n != b.length) throw new RuntimeException();
		for (int i = 0; i < n; i++)
			if (!isEqual(a[i], b[i], tol)) return false;
		return true;
	}

	public static final boolean allEqual(double[] a, double[] b) {
		return allEqual(a, b, defaultNumericalError);
	}

	public static final boolean allEqualScaled(double[] a, double[] b, double tol) {
		int n = a.length;
		if (n != b.length) throw new RuntimeException();
		for (int i = 0; i < n; i++)
			if (!isEqualScaled(a[i], b[i], tol)) return false;
		return true;
	}

	public static final boolean allEqualScaled(double[] a, double[] b) {
		return allEqualScaled(a, b, defaultNumericalError);
	}

	public static final boolean printBool(boolean b) {
		System.out.println(b ? "[1] TRUE" : "[1] FALSE");
		return b;
	}

	public static final boolean printBool(boolean... b) {
		if (b == null || b.length == 0) return false;
		System.out.print("[1]");
		boolean bb = true;
		for (int i = 0; i < b.length; i++) {
			System.out.print(b[i] ? " TRUE" : " FALSE");
			bb = bb & b[i];
		}
		System.out.println();
		return bb;
	}

	public static final boolean printAllEqual(double[] a, double[] b, double tol) {
		boolean v = allEqual(a, b, tol);
		printBool(v);
		if (v) return true;
		int n = a.length;
		boolean[] vv = new boolean[n];
		for (int i = 0; i < a.length; i++)
			vv[i] = isEqual(a[i], b[i], tol);
		System.out.print("True values: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(a[i]+ " ");
		System.out.println();

		System.out.print("Results: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(b[i]+ " ");
		System.out.println();

		System.out.print("|Diff|: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(abs(a[i]-b[i])+ " ");
		System.out.println();
		return false;
	}

	public static final boolean printAllEqual(double[] a, double[] b) {
		return printAllEqual(a, b, defaultNumericalError);
	}

	public static final boolean printAllEqualScaled(double[] a, double[] b, double tol) {
		boolean v = allEqualScaled(a, b, tol);
		printBool(v);
		if (v) return true;
		int n = a.length;
		boolean[] vv = new boolean[n];
		for (int i = 0; i < a.length; i++)
			vv[i] = isEqualScaled(a[i], b[i], tol);
		System.out.print("True values: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(a[i]+ " ");
		System.out.println();

		System.out.print("Results: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(b[i]+ " ");
		System.out.println();

		System.out.print("Relative Diff: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(abs(a[i]-b[i])/a[i]+ " ");
		System.out.println();
		return false;
	}

	public static final boolean printAllEqualScaled(double[] a, double[] b) {
		return printAllEqualScaled(a, b, defaultNumericalError);
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
						System.err.println(String.format("x = %g, sh = %g, sig = %g, Ga(sh) = %g", x, sh, sig, Ga));
						success = false;
					}
					double d3 = 1.0 / (Ga * pow(sig, sh)) * pow(x, sh - 1.0) * exp(-x / sig);
					if (!isEqual(d1, d3, 2 * defaultNumericalError)) { // Still within error limit
						System.err.println(String.format("Error: scaled dgamma = %3.18g, manually comp dgamma = %3.18g", d1, d3));
						System.err.println(String.format("x = %g, sh = %g, sig = %g, Ga(sh) = %3.30g", x, sh, sig, Ga));
						success = false;
					}
				}
			}
		}

		double Inf = Double.POSITIVE_INFINITY, xMax = DBL_MAX;
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

	@Test
	public static final boolean test_noncentralchisq() {
		System.out.println("##-- non central Chi^2 :");
		boolean success = true, cur_success;
		for (double df : new double[] { 0.1, 1, 10 }) {
			for (double ncp : new double[] { 0, 1, 10, 100 }) {
				for (double xB : new double [] { 2000, 1e6, 1e50, Double.POSITIVE_INFINITY}) {
					if (!isEqual(NonCentralChiSquare.cumulative(xB, df, ncp, true, false), 1)) {
						System.err.println(String.format("Error: pchisq(x=%g, df=%g, ncp=%g) = %3.18g. Correct answer = 1", xB, df, ncp));
						success = false;
					}
				}
			}
		}
		double cor_val = 49.77662465605547481573; // This is the value I took from R
		//double cor_val = 49.7766246561514; // This is the value given in d-p-q-r-test.R
		double val = NonCentralChiSquare.quantile(0.025, 31, 1, false, false); // Inf. loop PR#875
		if (!isEqual(val, cor_val, 1e-11)) {
			System.err.println(String.format("Error: qchisq(x=0.025, df=31, ncp=1) = %3.18g. Correct answer = %3.18", val, cor_val));
			success = false;
		}

		for (double df : new double[] {0.1, 0.5, 1.5, 4.7, 10, 20, 50, 100}) {
			System.out.print("df =" + df);
			cur_success = true;
			double dtol = 1e-12 * (2 < df && df <= 50 ? 64 : (df > 50 ? 20000 : 501));
			for (double xx : new double[] {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9, 1.2, df+3, df+7, df+20, df+30, df+35, df+38}) {
				double pval = NonCentralChiSquare.cumulative(xx, df, 1, true, false);
				double qval = NonCentralChiSquare.quantile(pval, df, 1, true, false);
				if (!isEqual(qval, xx, dtol)) {
					System.err.println(String.format("Error: xx=%g, df=%g, ncp=1, pchisq = %3.18g, qchisq = %3.18g != xx", xx, df, pval, qval));
					success = cur_success = false;
				}
			}
			printBool(cur_success);
		}

		// ## p ~= 1 (<==> 1-p ~= 0) -- gave infinite loop in R <= 1.8.1 -- PR#6421
		cur_success = true;
		boolean cur_success2 = true;
		for (int i = 10; i <= 54; i++) {
			double psml = pow(2, -i);
			double q0 = NonCentralChiSquare.quantile(psml, 1.2, 10, false, false);
			double q1 = NonCentralChiSquare.quantile(1-psml, 1.2, 10, true, false);
			double p0 = NonCentralChiSquare.cumulative(q0, 1.2, 10, false, false);
			double p1 = NonCentralChiSquare.cumulative(q1, 1.2, 10, false, false);
			// R code: up to 54, but only the first 30 is tested for accuracy
			if (i < 29 & !isEqual(q0, q1, 1e-5)) {
				System.err.println(String.format("Error: psml=%g, q0=%3.18g, q1 = %3.18g", psml, q0, q1));
				success = cur_success = false;
			}
			if (i < 29 & !isEqual(p0, psml)) {
				System.err.println(String.format("Error: psml=%g, p0=%3.18g", psml, q0));
				success = cur_success2 = false;
			}
			if (p1 > 0) {}; // To mute the compilation warning
		}
		printBool(cur_success);
		printBool(cur_success2);
		return success;
	}

	@Test
	public static final boolean test_beta() {
		System.out.println("##--- Beta (need more):");
		boolean success = true;
		for (int i = 0; i < 20; i++) {
			double a = LogNormal.random(5.5, 1, random);
			for (int j = 0; j < 20; j++) {
				double b = LogNormal.random(6.6, 1, random);
				for (int k = 0; k <= 10; k++) {
					double p = k/10.0;
					double v1 = Beta.density(p, a, b, false);
					double v2log = Beta.density(p, a, b, true);
					double v2 = exp(v2log);
					//System.out.println(String.format("Debug: p=%g, a=%3.18g, b=%3.18g, dbeta(p,a,b) = %3.18g, dbeta(p,a,b,TRUE) = %3.18g", p, a, b, v1, v2log));
					if (!isEqual(v1, v2, 1e-11)) {
						System.err.println(String.format("Error: p=%g, a=%3.18g, b=%3.18g, dbeta(p,a,b) = %3.18g, exp(dbeta(p,a,b,TRUE)) = %3.18g", p, a, b, v1, v2));
						success = false;
					}
				}
			}
		}
		return success;
	}

	@Test
	public static final boolean test_normal() {
		// Includes T distribution apparently
		System.out.println("##--- Normal (& Lognormal) :");
		boolean success = Normal.quantile(0, 1, 0, true, false) == Double.NEGATIVE_INFINITY
			&& Normal.quantile(Double.NEGATIVE_INFINITY, 1, 0, true, true) == Double.NEGATIVE_INFINITY;
		printBool(success);
		success &= Normal.quantile(1, 1, 0, true, false) == Double.POSITIVE_INFINITY
			&& Normal.quantile(0, 1, 0, true, true) == Double.POSITIVE_INFINITY;
		printBool(success);
		success &= Double.isNaN(Normal.quantile(1.1, 1, 0, true, false))
			&& Double.isNaN(Normal.quantile(-0.1, 1, 0, true, false));
		printBool(success);

		double[] xx = new double[] {Double.NEGATIVE_INFINITY, -1e100, 1,2,3,4,5,6, 1e200, Double.POSITIVE_INFINITY};
		double val;
		System.out.print("d.s0");
		for (int i = 0; i < xx.length; i++) {
			val = Normal.density(xx[i], 3, 0, false);
			System.out.print(" " + val);
			success &= (val == (i == 4 ? Double.POSITIVE_INFINITY : 0));
		}
		System.out.println();
		System.out.print("p.s0");
		for (int i = 0; i < xx.length; i++) {
			val = Normal.cumulative(xx[i], 3, 0, true, false);
			System.out.print(" " + val);
			success &= (val == (i >= 4 ? 1 : 0));
		}
		System.out.println();
		System.out.print("d.sI");
		for (int i = 0; i < xx.length; i++) {
			val = Normal.density(xx[i], 3, Double.POSITIVE_INFINITY, false);
			System.out.print(" " + val);
			success &= (val == 0);
		}
		System.out.println();
		System.out.print("p.sI");
		for (int i = 0; i < xx.length; i++) {
			val = Normal.cumulative(xx[i], 3, Double.POSITIVE_INFINITY, true, false);
			System.out.print(" " + val);
			success &= (val == (i == 0 ? 0 : i == 9 ? 1 : 0.5));
		}
		System.out.println();
		// ## 3 Test data from Wichura (1988) :
		double
			q1 = Normal.quantile(0.25, 0, 1, true, false),
			q2 = Normal.quantile(0.001, 0, 1, true, false),
			q3 = Normal.quantile(1e-20, 0, 1, true, false);
		// Supplied values from d-p-q-r-tests.R seem to be lacking precision
//		boolean cur_success = isEqual(q1, -0.6744897501960817, 1e-15) &&
//			isEqual(q2, -3.090232306167814, 1e-15) &&
//			isEqual(q3, -9.262340089798408, 1e-15);
		// These figures are taken from R console
		boolean cur_success = isEqual(q1, -0.6744897501960817054467, 1e-15) &&
			isEqual(q2, -3.0902323061678131921326, 1e-15) &&
			isEqual(q3, -9.2623400897984051738376, 1e-15);
		success &= cur_success;
		printBool(cur_success);
		q1 = Normal.quantile(-1e5, 0, 1, true, true);
		// Supplied value from d-p-q-r-tests.R seems to be lacking precision
		//cur_success = isEqual(q1, -447.1974945);
		// This figure is taken from R console
		cur_success = isEqual(q1, -447.1974944650480097152);
		success &= cur_success;
		printBool(cur_success);

		cur_success = true;
		for (int i = 0; i < 1000; i++) {
			double z = Normal.random_standard(random);
			double pz = Normal.cumulative(z, 0, 1, true, false);
			double pz_comp = 1-Normal.cumulative(-z, 0, 1, true, false);
			if (!isEqual(pz, pz_comp, 1e-15)) {
				System.err.println(String.format("Error: z=%3.18g, pnorm(z) = %3.18g, 1-pnorm(-z) = %3.18g", z, pz, pz_comp));
				success = cur_success = false;
			}
		}
		printBool(cur_success);
		boolean cur_success2 = true, cur_success3 = true, cur_success4 = true, cur_success5 = true, cur_success6 = true;
		StringBuilder buf = new StringBuilder();
		StringBuilder buf2 = new StringBuilder();
		// Java does not have NA
		for (int i = 0; i < 1003; i++) {
			double z = (i == 0 ? Double.NEGATIVE_INFINITY : i == 1 ? Double.POSITIVE_INFINITY : i == 2 ? Double.NaN :
				T.random(2, random));
			for (int df = 1; df <= 10; df++) {
				double pt = T.cumulative(z, df, true, false);
				double pt_comp = 1 - T.cumulative(-z, df, true, false);
				if (!isEqual(pt, pt_comp, 1e-15)) {
					System.err.println(String.format("Error: z=%3.18g, df=%d, pt(z,df) = %3.18g, 1-pt(-z,df) = %3.18g", z, df, pt, pt_comp));
					success = cur_success = false;
				}
			}
			double pz = Normal.cumulative(z, 0, 1, true, false);
			double pz_comp = 1-Normal.cumulative(z, 0, 1, false, false);
			if (!isEqual(pz, pz_comp)) {
				System.err.println(String.format("Error: z=%3.18g, pnorm(z) = %3.18g, 1-pnorm(z, lower=FALSE) = %3.18g", z, pz, pz_comp));
				success = cur_success2 = false;
			}
			double pz_comp2 = Normal.cumulative(-z, 0, 1, false, false);
			if (!isEqual(pz, pz_comp2)) {
				System.err.println(String.format("Error: z=%3.18g, pnorm(z) = %3.18g, pnorm(-z, lower=FALSE) = %3.18g", z, pz, pz_comp2));
				success = cur_success3 = false;
			}
			if (isInfinite(z) || z > -37.5) {
				double log_pz = log(pz);
				pz_comp = Normal.cumulative(z, 0, 1, true, true);
				if (!isEqual(log_pz, pz_comp, 2 * defaultNumericalError)) {
					// Special allowance. See bug #10
					System.err.println(String.format("Error: z=%3.18g, log(pnorm(z)) = %3.18g, pnorm(z, log=TRUE) = %3.18g", z, log_pz, pz_comp));
					success = cur_success4 = false;
				}
			}
			double plnorm_exp_z = LogNormal.cumulative(exp(z), 0, 1, true, false);
			if (!isEqual(pz, plnorm_exp_z)) {
				buf.append(String.format("Error: z=%3.18g, pnorm(z) = %3.18g, plnorm(exp(z)) = %3.18g", z, pz, plnorm_exp_z) + "\n");
				success = cur_success5 = false;
			}
			if (1e-5 < pz && pz < 1 - 1e-5) {
				double qnorm_pz = Normal.quantile(pz, 0, 1, true, false);
				if (!isEqual(z, qnorm_pz, 1e-12)) {
					buf2.append(String.format("Error: z=%3.18g, qnorm(pnorm(z)) = %3.18g", z, qnorm_pz) + "\n");
					success = cur_success6 = false;
				}
			}
		}
		printBool(cur_success);
		printBool(cur_success2);
		printBool(cur_success3);
		printBool(cur_success4);

		for (int y = -70; y <= 0; y += 10) {
			double log_pnorm_y = log(Normal.cumulative(y, 0, 1, true, false));
			double pnorm_y_log = Normal.cumulative(y, 0, 1, true, true);
			System.out.println(String.format("y=%d, log(pnorm(y)) = %3.18g, pnorm(y, log=TRUE) = %3.18g", y, log_pnorm_y, pnorm_y_log));
		}

		System.out.println();
		for (int y: c(colon(1,15), seq(20,40,5))) {
			double log_pnorm_y = log(Normal.cumulative(y, 0, 1, true, false));
			double pnorm_y_log = Normal.cumulative(y, 0, 1, true, true);
			double log_pnorm_min_y = log(Normal.cumulative(-y, 0, 1, true, false));
			double pnorm_min_y_log = Normal.cumulative(-y, 0, 1, true, true);
			System.out.println(String.format("y=%d, log(pnorm(y)) = %3.18g, pnorm(y, log=TRUE) = %3.18g, log(pnorm(-y)) = %3.18g, pnorm(-y, log=TRUE) = %3.18g", y, log_pnorm_y, pnorm_y_log, log_pnorm_min_y, pnorm_min_y_log));
		}
		double[] yy = c(colon(1., 50), vpow(10, c(colon(3,10), c(20,50,150,250))));
		yy = c(vmin(yy), new double[] {0}, yy);
		for (double y: yy) {
			double py_minus = Normal.cumulative(-y, 0, 1, true, false);
			double py_plus = Normal.cumulative(+y, 0, 1, false, false);
			if (py_plus != py_minus) {
				System.err.println(String.format("y=%d, pnorm(-y) = %3.18g, pnorm(y, lower=FALSE)", y, py_minus, py_plus));
				success = false;
			}
			py_minus = Normal.cumulative(-y, 0, 1, true, true);
			py_plus = Normal.cumulative(+y, 0, 1, false, true);
			if (py_plus != py_minus) {
				System.err.println(String.format("y=%d, pnorm(-y, log=TRUE) = %3.18g, pnorm(y, lower=FALSE, log=TRUE)", y, py_minus, py_plus));
				success = false;
			}
		}
		printBool(cur_success5);
		if (!cur_success5)
			System.err.println(buf.toString());
		printBool(cur_success6);
		if (!cur_success6)
			System.err.println(buf2.toString());
		return success;
	}

	@Test
	public static final void test_random() {
		// Set up the instances to ensure proper parameterization
		Beta beta = new Beta(0.8, 2);
		beta.setRandomEngine(random);
		Binomial binom = new Binomial(25, Math.PI/16.0);
		binom.setRandomEngine(random);
		Cauchy cauchy = new Cauchy(12, 2);
		cauchy.setRandomEngine(random);
		ChiSquare chisq = new ChiSquare(3);
		chisq.setRandomEngine(random);
		Exponential exp = new Exponential(1/2.0);
		exp.setRandomEngine(random);
		F f = new F(12, 6);
		f.setRandomEngine(random);
		Gamma gamma = new Gamma(2, 5);
		gamma.setRandomEngine(random);
		Geometric geom = new Geometric(Math.PI/16);
		geom.setRandomEngine(random);
		HyperGeometric hyper = new HyperGeometric(40, 30, 20);
		hyper.setRandomEngine(random);
		LogNormal lnorm = new LogNormal(-1, 3);
		lnorm.setRandomEngine(random);
		Logistic logis = new Logistic(12, 2);
		logis.setRandomEngine(random);
		NegBinomial nbinom = new NegBinomial(7, 0.01);
		nbinom.setRandomEngine(random);
		Normal norm = new Normal(-1, 3);
		norm.setRandomEngine(random);
		Poisson pois = new Poisson(12);
		pois.setRandomEngine(random);
		SignRank signrank = new SignRank(47);
		signrank.setRandomEngine(random);
		T t = new T(11);
		t.setRandomEngine(random);
		Uniform unif = new Uniform(0.2, 2);
		unif.setRandomEngine(random);
		Weibull weibull = new Weibull(3, 2);
		weibull.setRandomEngine(random);
		Wilcoxon wilcox = new Wilcoxon(13, 17);
		wilcox.setRandomEngine(random);
		T t2 = new T(1.01);
		t2.setRandomEngine(random);

		int n = 20;
		double[]
			Rbeta = beta.random(n),
			Rbinom = binom.random(n),
			Rcauchy = cauchy.random(n),
			Rchisq = chisq.random(n),
			Rexp = exp.random(n),
			Rf = f.random(n),
			Rgamma = gamma.random(n),
			Rgeom = geom.random(n),
			Rhyper = hyper.random(n),
			Rlnorm = lnorm.random(n),
			Rlogis = logis.random(n),
			Rnbinom = nbinom.random(n),
			Rnorm = norm.random(n),
			Rpois = pois.random(n),
			Rsignrank = signrank.random(n),
			Rt = t.random(n),
			Runif = unif.random(n),
			Rweibull = weibull.random(n),
			Rwilcox = wilcox.random(n),
			Rt2 = t2.random(n);

		System.out.println();
		System.out.println("Random beta(0.8, 2)");
		print(Rbeta);
		System.out.println("Random binomial(25, pi/16.0)");
		print(Rbinom);
		System.out.println("Random cauchy(12, 2)");
		print(Rcauchy);
		System.out.println("Random chisq(3)");
		print(Rchisq);
		System.out.println("Random exp(rate = 2.0)");
		print(Rexp);
		System.out.println("Random f(12, 6)");
		print(Rf);
		System.out.println("Random gamma(2, 5)");
		print(Rgamma);
		System.out.println("Random geom(pi/16.0)");
		print(Rgeom);
		System.out.println("Random hyper(40, 30, 20)");
		print(Rhyper);
		System.out.println("Random lnorm(-1, 3)");
		print(Rlnorm);
		System.out.println("Random logis(12, 2)");
		print(Rlogis);
		System.out.println("Random nbinom(7, 0.01)");
		print(Rnbinom);
		System.out.println("Random norm(-1, 3)");
		print(Rnorm);
		System.out.println("Random pois(12)");
		print(Rpois);
		System.out.println("Random signrank(47)");
		print(Rsignrank);
		System.out.println("Random t(11)");
		print(Rt);
		System.out.println("Random t(1.01)");
		print(Rt2);
		System.out.println("Random unif(0.2, 2)");
		print(Runif);
		System.out.println("Random weibull(3, 2)");
		print(Rweibull);
		System.out.println("Random wilcox(13, 17)");
		print(Rwilcox);

		boolean lower_tail = true, log_p = false;
		double[]
			Pbeta = beta.cumulative(Rbeta, lower_tail, log_p),
			Pbinom = binom.cumulative(Rbinom, lower_tail, log_p),
			Pcauchy = cauchy.cumulative(Rcauchy, lower_tail, log_p),
			Pchisq = chisq.cumulative(Rchisq, lower_tail, log_p),
			Pexp = exp.cumulative(Rexp, lower_tail, log_p),
			Pf = f.cumulative(Rf, lower_tail, log_p),
			Pgamma = gamma.cumulative(Rgamma, lower_tail, log_p),
			Pgeom = geom.cumulative(Rgeom, lower_tail, log_p),
			Phyper = hyper.cumulative(Rhyper, lower_tail, log_p),
			Plnorm = lnorm.cumulative(Rlnorm, lower_tail, log_p),
			Plogis = logis.cumulative(Rlogis, lower_tail, log_p),
			Pnbinom = nbinom.cumulative(Rnbinom, lower_tail, log_p),
			Pnorm = norm.cumulative(Rnorm, lower_tail, log_p),
			Ppois = pois.cumulative(Rpois, lower_tail, log_p),
			Psignrank = signrank.cumulative(Rsignrank, lower_tail, log_p),
			Pt = t.cumulative(Rt, lower_tail, log_p),
			Pt2 = t2.cumulative(Rt2, lower_tail, log_p),
			Punif = unif.cumulative(Runif, lower_tail, log_p),
			Pweibull = weibull.cumulative(Rweibull, lower_tail, log_p),
			Pwilcox = wilcox.cumulative(Rwilcox, lower_tail, log_p);

		System.out.println();
		System.out.println("Cumulative beta(0.8, 2)");
		print(Pbeta);
		System.out.println("Cumulative binomial(25, pi/16.0)");
		print(Pbinom);
		System.out.println("Cumulative cauchy(12, 2)");
		print(Pcauchy);
		System.out.println("Cumulative chisq(3)");
		print(Pchisq);
		System.out.println("Cumulative exp(rate = 2.0)");
		print(Pexp);
		System.out.println("Cumulative f(12, 6)");
		print(Pf);
		System.out.println("Cumulative gamma(2, 5)");
		print(Pgamma);
		System.out.println("Cumulative geom(pi/16.0)");
		print(Pgeom);
		System.out.println("Cumulative hyper(40, 30, 20)");
		print(Phyper);
		System.out.println("Cumulative lnorm(-1, 3)");
		print(Plnorm);
		System.out.println("Cumulative logis(12, 2)");
		print(Plogis);
		System.out.println("Cumulative nbinom(7, 0.01)");
		print(Pnbinom);
		System.out.println("Cumulative norm(-1, 3)");
		print(Pnorm);
		System.out.println("Cumulative pois(12)");
		print(Ppois);
		System.out.println("Cumulative signrank(47)");
		print(Psignrank);
		System.out.println("Cumulative t(11)");
		print(Pt);
		System.out.println("Cumulative t(1.01)");
		print(Pt2);
		System.out.println("Cumulative unif(0.2, 2)");
		print(Punif);
		System.out.println("Cumulative weibull(3, 2)");
		print(Pweibull);
		System.out.println("Cumulative wilcox(13, 17)");
		print(Pwilcox);

		double[]
			Dbeta = beta.density(Rbeta, false),
			Dbinom = binom.density(Rbinom, false),
			Dcauchy = cauchy.density(Rcauchy, false),
			Dchisq = chisq.density(Rchisq, false),
			Dexp = exp.density(Rexp, false),
			Df = f.density(Rf, false),
			Dgamma = gamma.density(Rgamma, false),
			Dgeom = geom.density(Rgeom, false),
			Dhyper = hyper.density(Rhyper, false),
			Dlnorm = lnorm.density(Rlnorm, false),
			Dlogis = logis.density(Rlogis, false),
			Dnbinom = nbinom.density(Rnbinom, false),
			Dnorm = norm.density(Rnorm, false),
			Dpois = pois.density(Rpois, false),
			Dsignrank = signrank.density(Rsignrank, false),
			Dt = t.density(Rt, false),
			Dt2 = t2.density(Rt2, false),
			Dunif = unif.density(Runif, false),
			Dweibull = weibull.density(Rweibull, false),
			Dwilcox = wilcox.density(Rwilcox, false);

		System.out.println();
		System.out.println("Density beta(0.8, 2)");
		print(Dbeta);
		System.out.println("Density binomial(25, pi/16.0)");
		print(Dbinom);
		System.out.println("Density cauchy(12, 2)");
		print(Dcauchy);
		System.out.println("Density chisq(3)");
		print(Dchisq);
		System.out.println("Density exp(rate = 2.0)");
		print(Dexp);
		System.out.println("Density f(12, 6)");
		print(Df);
		System.out.println("Density gamma(2, 5)");
		print(Dgamma);
		System.out.println("Density geom(pi/16.0)");
		print(Dgeom);
		System.out.println("Density hyper(40, 30, 20)");
		print(Dhyper);
		System.out.println("Density lnorm(-1, 3)");
		print(Dlnorm);
		System.out.println("Density logis(12, 2)");
		print(Dlogis);
		System.out.println("Density nbinom(7, 0.01)");
		print(Dnbinom);
		System.out.println("Density norm(-1, 3)");
		print(Dnorm);
		System.out.println("Density pois(12)");
		print(Dpois);
		System.out.println("Density signrank(47)");
		print(Dsignrank);
		System.out.println("Density t(11)");
		print(Dt);
		System.out.println("Density t(1.01)");
		print(Dt2);
		System.out.println("Density unif(0.2, 2)");
		print(Dunif);
		System.out.println("Density weibull(3, 2)");
		print(Dweibull);
		System.out.println("Density wilcox(13, 17)");
		print(Dwilcox);

		double[]
			Qbeta = beta.quantile(Pbeta, lower_tail, log_p),
			Qbinom = binom.quantile(Pbinom, lower_tail, log_p),
			Qcauchy = cauchy.quantile(Pcauchy, lower_tail, log_p),
			Qchisq = chisq.quantile(Pchisq, lower_tail, log_p),
			Qexp = exp.quantile(Pexp, lower_tail, log_p),
			Qf = f.quantile(Pf, lower_tail, log_p),
			Qgamma = gamma.quantile(Pgamma, lower_tail, log_p),
			Qgeom = geom.quantile(Pgeom, lower_tail, log_p),
			Qhyper = hyper.quantile(Phyper, lower_tail, log_p),
			Qlnorm = lnorm.quantile(Plnorm, lower_tail, log_p),
			Qlogis = logis.quantile(Plogis, lower_tail, log_p),
			Qnbinom = nbinom.quantile(Pnbinom, lower_tail, log_p),
			Qnorm = norm.quantile(Pnorm, lower_tail, log_p),
			Qpois = pois.quantile(Ppois, lower_tail, log_p),
			Qsignrank = signrank.quantile(Psignrank, lower_tail, log_p),
			Qt = t.quantile(Pt, lower_tail, log_p),
			Qt2 = t2.quantile(Pt2, lower_tail, log_p),
			Qunif = unif.quantile(Punif, lower_tail, log_p),
			Qweibull = weibull.quantile(Pweibull, lower_tail, log_p),
			Qwilcox = wilcox.quantile(Pwilcox, lower_tail, log_p);

		System.out.println();
		System.out.println("Lower tail equality beta(0.8, 2)");
		printAllEqual(Rbeta, Qbeta);
		System.out.println("Lower tail equality binomial(25, pi/16.0)");
		printAllEqual(Rbinom, Qbinom);
		System.out.println("Lower tail equality cauchy(12, 2)");
		printAllEqual(Rcauchy, Qcauchy, 2 * defaultNumericalError); // See bug #8
		System.out.println("Lower tail equality chisq(3)");
		printAllEqual(Rchisq, Qchisq, 2 * defaultNumericalError);
		System.out.println("Lower tail equality exp(rate = 2.0)");
		printAllEqual(Rexp, Qexp);
		System.out.println("Lower tail equality f(12, 6)");
		printAllEqual(Rf, Qf);
		System.out.println("Lower tail equality gamma(2, 5)");
		printAllEqual(Rgamma, Qgamma, 2 * defaultNumericalError);
		System.out.println("Lower tail equality geom(pi/16.0)");
		printAllEqual(Rgeom, Qgeom);
		System.out.println("Lower tail equality hyper(40, 30, 20)");
		printAllEqual(Rhyper, Qhyper);
		System.out.println("Lower tail equality lnorm(-1, 3)");
		printAllEqual(Rlnorm, Qlnorm);
		System.out.println("Lower tail equality logis(12, 2)");
		printAllEqual(Rlogis, Qlogis);
		System.out.println("Lower tail equality nbinom(7, 0.01)");
		printAllEqual(Rnbinom, Qnbinom);
		System.out.println("Lower tail equality norm(-1, 3)");
		printAllEqual(Rnorm, Qnorm);
		System.out.println("Lower tail equality pois(12)");
		printAllEqual(Rpois, Qpois);
		System.out.println("Lower tail equality signrank(47)");
		printAllEqual(Rsignrank, Qsignrank);
		System.out.println("Lower tail equality t(11)");
		printAllEqual(Rt, Qt);
		System.out.println("Lower tail equality t(1.01)");
		printAllEqual(Rt2, Qt2);
		System.out.println("Lower tail equality unif(0.2, 2)");
		printAllEqual(Runif, Qunif);
		System.out.println("Lower tail equality weibull(3, 2)");
		printAllEqual(Rweibull, Qweibull);
		System.out.println("Lower tail equality wilcox(13, 17)");
		printAllEqual(Rwilcox, Qwilcox);

		lower_tail = false; log_p = false;
		Qbeta = beta.quantile(vcomp(Pbeta), lower_tail, log_p);
		Qbinom = binom.quantile(vcomp(Pbinom), lower_tail, log_p);
		Qcauchy = cauchy.quantile(vcomp(Pcauchy), lower_tail, log_p);
		Qchisq = chisq.quantile(vcomp(Pchisq), lower_tail, log_p);
		Qexp = exp.quantile(vcomp(Pexp), lower_tail, log_p);
		Qf = f.quantile(vcomp(Pf), lower_tail, log_p);
		Qgamma = gamma.quantile(vcomp(Pgamma), lower_tail, log_p);
		Qgeom = geom.quantile(vcomp(Pgeom), lower_tail, log_p);
		Qhyper = hyper.quantile(vcomp(Phyper), lower_tail, log_p);
		Qlnorm = lnorm.quantile(vcomp(Plnorm), lower_tail, log_p);
		Qlogis = logis.quantile(vcomp(Plogis), lower_tail, log_p);
		Qnbinom = nbinom.quantile(vcomp(Pnbinom), lower_tail, log_p);
		Qnorm = norm.quantile(vcomp(Pnorm), lower_tail, log_p);
		Qpois = pois.quantile(vcomp(Ppois), lower_tail, log_p);
		Qsignrank = signrank.quantile(vcomp(Psignrank), lower_tail, log_p);
		Qt = t.quantile(vcomp(Pt), lower_tail, log_p);
		Qt2 = t2.quantile(vcomp(Pt2), lower_tail, log_p);
		Qunif = unif.quantile(vcomp(Punif), lower_tail, log_p);
		Qweibull = weibull.quantile(vcomp(Pweibull), lower_tail, log_p);
		Qwilcox = wilcox.quantile(vcomp(Pwilcox), lower_tail, log_p);

		System.out.println();
		System.out.println("Upper tail equality beta(0.8, 2)");
		printAllEqual(Rbeta, Qbeta);
		System.out.println("Upper tail equality binomial(25, pi/16.0)");
		printAllEqual(Rbinom, Qbinom);
		System.out.println("Upper tail equality cauchy(12, 2)");
		printAllEqual(Rcauchy, Qcauchy, 100 * defaultNumericalError); // See bug #8
		System.out.println("Upper tail equality chisq(3)");
		printAllEqual(Rchisq, Qchisq, 2 * defaultNumericalError);
		System.out.println("Upper tail equality exp(rate = 2.0)");
		printAllEqual(Rexp, Qexp);
		System.out.println("Upper tail equality f(12, 6)");
		printAllEqual(Rf, Qf);
		System.out.println("Upper tail equality gamma(2, 5)");
		printAllEqual(Rgamma, Qgamma, 4 * defaultNumericalError);
		System.out.println("Upper tail equality geom(pi/16.0)");
		printAllEqual(Rgeom, Qgeom);
		System.out.println("Upper tail equality hyper(40, 30, 20)");
		printAllEqual(Rhyper, Qhyper);
		System.out.println("Upper tail equality lnorm(-1, 3)");
		printAllEqual(Rlnorm, Qlnorm);
		System.out.println("Upper tail equality logis(12, 2)");
		printAllEqual(Rlogis, Qlogis);
		System.out.println("Upper tail equality nbinom(7, 0.01)");
		printAllEqual(Rnbinom, Qnbinom);
		System.out.println("Upper tail equality norm(-1, 3)");
		printAllEqual(Rnorm, Qnorm);
		System.out.println("Upper tail equality pois(12)");
		printAllEqual(Rpois, Qpois);
		System.out.println("Upper tail equality signrank(47)");
		printAllEqual(Rsignrank, Qsignrank);
		System.out.println("Upper tail equality t(11)");
		printAllEqual(Rt, Qt);
		System.out.println("Upper tail equality t(1.01)");
		printAllEqual(Rt2, Qt2);
		System.out.println("Upper tail equality unif(0.2, 2)");
		printAllEqual(Runif, Qunif);
		System.out.println("Upper tail equality weibull(3, 2)");
		printAllEqual(Rweibull, Qweibull);
		System.out.println("Upper tail equality wilcox(13, 17)");
		printAllEqual(Rwilcox, Qwilcox);

		lower_tail = true; log_p = true;
		Qbeta = beta.quantile(vlog(Pbeta), lower_tail, log_p);
		Qbinom = binom.quantile(vlog(Pbinom), lower_tail, log_p);
		Qcauchy = cauchy.quantile(vlog(Pcauchy), lower_tail, log_p);
		Qchisq = chisq.quantile(vlog(Pchisq), lower_tail, log_p);
		Qexp = exp.quantile(vlog(Pexp), lower_tail, log_p);
		Qf = f.quantile(vlog(Pf), lower_tail, log_p);
		Qgamma = gamma.quantile(vlog(Pgamma), lower_tail, log_p);
		Qgeom = geom.quantile(vlog(Pgeom), lower_tail, log_p);
		Qhyper = hyper.quantile(vlog(Phyper), lower_tail, log_p);
		Qlnorm = lnorm.quantile(vlog(Plnorm), lower_tail, log_p);
		Qlogis = logis.quantile(vlog(Plogis), lower_tail, log_p);
		Qnbinom = nbinom.quantile(vlog(Pnbinom), lower_tail, log_p);
		Qnorm = norm.quantile(vlog(Pnorm), lower_tail, log_p);
		Qpois = pois.quantile(vlog(Ppois), lower_tail, log_p);
		Qsignrank = signrank.quantile(vlog(Psignrank), lower_tail, log_p);
		Qt = t.quantile(vlog(Pt), lower_tail, log_p);
		Qt2 = t2.quantile(vlog(Pt2), lower_tail, log_p);
		Qunif = unif.quantile(vlog(Punif), lower_tail, log_p);
		Qweibull = weibull.quantile(vlog(Pweibull), lower_tail, log_p);
		Qwilcox = wilcox.quantile(vlog(Pwilcox), lower_tail, log_p);

		System.out.println();
		System.out.println("Lower tail, log equality beta(0.8, 2)");
		printAllEqual(Rbeta, Qbeta);
		System.out.println("Lower tail, log equality binomial(25, pi/16.0)");
		printAllEqual(Rbinom, Qbinom);
		System.out.println("Lower tail, log equality cauchy(12, 2)");
		printAllEqual(Rcauchy, Qcauchy, 5 * defaultNumericalError); // See bug #8
		System.out.println("Lower tail, log equality chisq(3)");
		printAllEqual(Rchisq, Qchisq, 2 * defaultNumericalError);
		System.out.println("Lower tail, log equality exp(rate = 2.0)");
		printAllEqual(Rexp, Qexp);
		System.out.println("Lower tail, log equality f(12, 6)");
		printAllEqual(Rf, Qf);
		System.out.println("Lower tail, log equality gamma(2, 5)");
		printAllEqual(Rgamma, Qgamma, 2 * defaultNumericalError);
		System.out.println("Lower tail, log equality geom(pi/16.0)");
		printAllEqual(Rgeom, Qgeom);
		System.out.println("Lower tail, log equality hyper(40, 30, 20)");
		printAllEqual(Rhyper, Qhyper);
		System.out.println("Lower tail, log equality lnorm(-1, 3)");
		printAllEqual(Rlnorm, Qlnorm);
		System.out.println("Lower tail, log equality logis(12, 2)");
		printAllEqual(Rlogis, Qlogis);
		System.out.println("Lower tail, log equality nbinom(7, 0.01)");
		printAllEqual(Rnbinom, Qnbinom);
		System.out.println("Lower tail, log equality norm(-1, 3)");
		printAllEqual(Rnorm, Qnorm);
		System.out.println("Lower tail, log equality pois(12)");
		printAllEqual(Rpois, Qpois);
		System.out.println("Lower tail, log equality signrank(47)");
		printAllEqual(Rsignrank, Qsignrank);
		System.out.println("Lower tail, log equality t(11)");
		printAllEqual(Rt, Qt);
		System.out.println("Lower tail, log equality t(1.01)");
		printAllEqual(Rt2, Qt2);
		System.out.println("Lower tail, log equality unif(0.2, 2)");
		printAllEqual(Runif, Qunif);
		System.out.println("Lower tail, log equality weibull(3, 2)");
		printAllEqual(Rweibull, Qweibull);
		System.out.println("Lower tail, log equality wilcox(13, 17)");
		printAllEqual(Rwilcox, Qwilcox);

		lower_tail = false; log_p = true;
		Qbeta = beta.quantile(vlog1pComps(Pbeta), lower_tail, log_p);
		Qbinom = binom.quantile(vlog1pComps(Pbinom), lower_tail, log_p);
		Qcauchy = cauchy.quantile(vlog1pComps(Pcauchy), lower_tail, log_p);
		Qchisq = chisq.quantile(vlog1pComps(Pchisq), lower_tail, log_p);
		Qexp = exp.quantile(vlog1pComps(Pexp), lower_tail, log_p);
		Qf = f.quantile(vlog1pComps(Pf), lower_tail, log_p);
		Qgamma = gamma.quantile(vlog1pComps(Pgamma), lower_tail, log_p);
		Qgeom = geom.quantile(vlog1pComps(Pgeom), lower_tail, log_p);
		Qhyper = hyper.quantile(vlog1pComps(Phyper), lower_tail, log_p);
		Qlnorm = lnorm.quantile(vlog1pComps(Plnorm), lower_tail, log_p);
		Qlogis = logis.quantile(vlog1pComps(Plogis), lower_tail, log_p);
		Qnbinom = nbinom.quantile(vlog1pComps(Pnbinom), lower_tail, log_p);
		Qnorm = norm.quantile(vlog1pComps(Pnorm), lower_tail, log_p);
		Qpois = pois.quantile(vlog1pComps(Ppois), lower_tail, log_p);
		Qsignrank = signrank.quantile(vlog1pComps(Psignrank), lower_tail, log_p);
		Qt = t.quantile(vlog1pComps(Pt), lower_tail, log_p);
		Qt2 = t2.quantile(vlog1pComps(Pt2), lower_tail, log_p);
		Qunif = unif.quantile(vlog1pComps(Punif), lower_tail, log_p);
		Qweibull = weibull.quantile(vlog1pComps(Pweibull), lower_tail, log_p);
		Qwilcox = wilcox.quantile(vlog1pComps(Pwilcox), lower_tail, log_p);

		System.out.println();
		System.out.println("Upper tail, log equality beta(0.8, 2)");
		printAllEqual(Rbeta, Qbeta);
		System.out.println("Upper tail, log equality binomial(25, pi/16.0)");
		printAllEqual(Rbinom, Qbinom);
		System.out.println("Upper tail, log equality cauchy(12, 2)");
		printAllEqual(Rcauchy, Qcauchy);
		System.out.println("Upper tail, log equality chisq(3)");
		printAllEqual(Rchisq, Qchisq, 2 * defaultNumericalError);
		System.out.println("Upper tail, log equality exp(rate = 2.0)");
		printAllEqual(Rexp, Qexp);
		System.out.println("Upper tail, log equality f(12, 6)");
		printAllEqual(Rf, Qf);
		System.out.println("Upper tail, log equality gamma(2, 5)");
		printAllEqual(Rgamma, Qgamma, 4 * defaultNumericalError);
		System.out.println("Upper tail, log equality geom(pi/16.0)");
		printAllEqual(Rgeom, Qgeom);
		System.out.println("Upper tail, log equality hyper(40, 30, 20)");
		printAllEqual(Rhyper, Qhyper);
		System.out.println("Upper tail, log equality lnorm(-1, 3)");
		printAllEqual(Rlnorm, Qlnorm);
		System.out.println("Upper tail, log equality logis(12, 2)");
		printAllEqual(Rlogis, Qlogis);
		System.out.println("Upper tail, log equality nbinom(7, 0.01)");
		printAllEqual(Rnbinom, Qnbinom);
		System.out.println("Upper tail, log equality norm(-1, 3)");
		printAllEqual(Rnorm, Qnorm);
		System.out.println("Upper tail, log equality pois(12)");
		printAllEqual(Rpois, Qpois);
		System.out.println("Upper tail, log equality signrank(47)");
		printAllEqual(Rsignrank, Qsignrank);
		System.out.println("Upper tail, log equality t(11)");
		printAllEqual(Rt, Qt);
		System.out.println("Upper tail, log equality t(1.01)");
		printAllEqual(Rt2, Qt2);
		System.out.println("Upper tail, log equality unif(0.2, 2)");
		printAllEqual(Runif, Qunif);
		System.out.println("Upper tail, log equality weibull(3, 2)");
		printAllEqual(Rweibull, Qweibull);
		System.out.println("Upper tail, log equality wilcox(13, 17)");
		printAllEqual(Rwilcox, Qwilcox);

		lower_tail = false; log_p = true;
		double[]
			_Pbeta = beta.cumulative(Rbeta, lower_tail, log_p),
			_Pbinom = binom.cumulative(Rbinom, lower_tail, log_p),
			_Pcauchy = cauchy.cumulative(Rcauchy, lower_tail, log_p),
			_Pchisq = chisq.cumulative(Rchisq, lower_tail, log_p),
			_Pexp = exp.cumulative(Rexp, lower_tail, log_p),
			_Pf = f.cumulative(Rf, lower_tail, log_p),
			_Pgamma = gamma.cumulative(Rgamma, lower_tail, log_p),
			_Pgeom = geom.cumulative(Rgeom, lower_tail, log_p),
			_Phyper = hyper.cumulative(Rhyper, lower_tail, log_p),
			_Plnorm = lnorm.cumulative(Rlnorm, lower_tail, log_p),
			_Plogis = logis.cumulative(Rlogis, lower_tail, log_p),
			_Pnbinom = nbinom.cumulative(Rnbinom, lower_tail, log_p),
			_Pnorm = norm.cumulative(Rnorm, lower_tail, log_p),
			_Ppois = pois.cumulative(Rpois, lower_tail, log_p),
			_Psignrank = signrank.cumulative(Rsignrank, lower_tail, log_p),
			_Pt = t.cumulative(Rt, lower_tail, log_p),
			_Pt2 = t2.cumulative(Rt2, lower_tail, log_p),
			_Punif = unif.cumulative(Runif, lower_tail, log_p),
			_Pweibull = weibull.cumulative(Rweibull, lower_tail, log_p),
			_Pwilcox = wilcox.cumulative(Rwilcox, lower_tail, log_p);

		System.out.println();
		System.out.println("Upper tail cumulative equality beta(0.8, 2)");
		printAllEqual(vlog1pComps(Pbeta), _Pbeta);
		System.out.println("Upper tail cumulative equality binomial(25, pi/16.0)");
		printAllEqual(vlog1pComps(Pbinom), _Pbinom);
		System.out.println("Upper tail cumulative equality cauchy(12, 2)");
		printAllEqual(vlog1pComps(Pcauchy), _Pcauchy);
		System.out.println("Upper tail cumulative equality chisq(3)");
		printAllEqual(vlog1pComps(Pchisq), _Pchisq);
		System.out.println("Upper tail cumulative equality exp(rate = 2.0)");
		printAllEqual(vlog1pComps(Pexp), _Pexp);
		System.out.println("Upper tail cumulative equality f(12, 6)");
		printAllEqual(vlog1pComps(Pf), _Pf);
		System.out.println("Upper tail cumulative equality gamma(2, 5)");
		printAllEqual(vlog1pComps(Pgamma), _Pgamma);
		System.out.println("Upper tail cumulative equality geom(pi/16.0)");
		printAllEqual(vlog1pComps(Pgeom), _Pgeom);
		System.out.println("Upper tail cumulative equality hyper(40, 30, 20)");
		printAllEqual(vlog1pComps(Phyper), _Phyper);
		System.out.println("Upper tail cumulative equality lnorm(-1, 3)");
		printAllEqual(vlog1pComps(Plnorm), _Plnorm);
		System.out.println("Upper tail cumulative equality logis(12, 2)");
		printAllEqual(vlog1pComps(Plogis), _Plogis);
		System.out.println("Upper tail cumulative equality nbinom(7, 0.01)");
		printAllEqual(vlog1pComps(Pnbinom), _Pnbinom);
		System.out.println("Upper tail cumulative equality norm(-1, 3)");
		printAllEqual(vlog1pComps(Pnorm), _Pnorm);
		System.out.println("Upper tail cumulative equality pois(12)");
		printAllEqual(vlog1pComps(Ppois), _Ppois);
		System.out.println("Upper tail cumulative equality signrank(47)");
		printAllEqual(vlog1pComps(Psignrank), _Psignrank);
		System.out.println("Upper tail cumulative equality t(11)");
		printAllEqual(vlog1pComps(Pt), _Pt);
		System.out.println("Upper tail cumulative equality t(1.01)");
		printAllEqual(vlog1pComps(Pt2), _Pt2);
		System.out.println("Upper tail cumulative equality unif(0.2, 2)");
		printAllEqual(vlog1pComps(Punif), _Punif);
		System.out.println("Upper tail cumulative equality weibull(3, 2)");
		printAllEqual(vlog1pComps(Pweibull), _Pweibull);
		System.out.println("Upper tail cumulative equality wilcox(13, 17)");
		printAllEqual(vlog1pComps(Pwilcox), _Pwilcox);
	}

	@Test
	public static final boolean test_extreme() {
		System.out.println("### (Extreme) tail tests added more recently:");
		boolean success = true;
		double neginf = Double.NEGATIVE_INFINITY, inf = Double.POSITIVE_INFINITY;
		double x[], val;

		//*
		success = printBool(isEqual(1, -1e-17/Exponential.cumulative(Exponential.quantile(-1e-17, 1, true, true), 1, true, true)));
		success &= printBool(isEqual(abs(Gamma.cumulative(30, 100, 1, false, true)), 7.3384686328784e-24, 1e-36));
		success &= printBool(isEqual(1, Cauchy.cumulative(-1e20, 0, 1, true, false) / 3.18309886183791e-21));
		success &= printBool(isEqual(1, Cauchy.cumulative(+1e15, 0, 1, true, true) / -3.18309886183791e-16)); // PR#6756

		Cauchy cauchy = new Cauchy(0, 1);
		double[] ex = new double[] {1,2,5,10,15,20,25,50,100,200,300, Double.POSITIVE_INFINITY};
		x = vpow(10, ex);
		for (double _x : x)
			if (_x > 1e10)
				printBool(isEqual(T.cumulative(-_x, 1, true, false), cauchy.cumulative(-_x), 1e-15));
		System.out.println("## for PR#7902:");
		double[] rec_x = rec(x), mins_x = vmin(x);
		success &= printAllEqualScaled(mins_x, cauchy.quantile(cauchy.cumulative(mins_x)));
		success &= printAllEqualScaled(x, cauchy.quantile(cauchy.cumulative(x, true, true), true, true));
		success &= printAllEqual(rec_x, cauchy.quantile(cauchy.cumulative(rec_x)));
		ex = vmin(c(rev(rec_x), ex));
		success &= printAllEqualScaled(ex, cauchy.quantile(cauchy.cumulative(ex, true, true), true, true));

		x = new double[] { 0, 1};
		ex = new double[] { neginf, inf };
		if (!allEqual(cauchy.cumulative(ex), x) ||
			!allEqual(cauchy.quantile(x), ex) ||
			!allEqual(cauchy.quantile(new double[] {neginf, 0}, true, true), ex)) {
			System.err.println("Boundary exception error in Cauchy distribution");
			success = false;
		}

		System.out.println("## PR#6757:");
		if (!isEqualScaled(pow(1e-23, 12), Binomial.cumulative(11, 12, 1e-23, false, false), 1e-12)) {
			System.err.println("Extreme tail error in Binomial.cumulative");
			success = false;
		}

		System.out.println("## PR#6792:");
		val = Geometric.cumulative(1, 1e-17, true, false);
		if (!isEqualScaled(2*1e-17, val)) {
			System.err.println("Extreme tail error in Geometric.cumulative");
			success = false;
		}

		x = vpow(10, colon(100, 295));
		for (double v : new double[] {1e-250, 1e-25, 0.9, 1.1, 101, 1e10, 1e100}) {
			Gamma pgamma = new Gamma(v, 1);
			success &= printAllEqualScaled(vmin(x), pgamma.cumulative(x, false, true));
		}
		x = vpow(2, colon(-1022, -900));
		Gamma g = new Gamma(10, 1);
		success &= printAllEqual(vmin(g.cumulative(x, true, true), vtimes(10, vlog(x))), rep(-15.104412573076, x.length), 1e-12);
		g = new Gamma(0.1, 1);
		success &= printAllEqual(vmin(g.cumulative(x, true, true), vtimes(0.1, vlog(x))), rep(0.0498724412598364, x.length), 1e-13);

		Poisson pois = new Poisson(3e-308);
		success &= printAllEqualScaled(c(-7096.080376108055133955, -14204.287543530712355278), pois.density(c(10.0,20.0), true));
		val = Poisson.density(1e20, 1e-290, true);
		success &= printBool(isEqualScaled(-71280137882815411781632.0, val));

		String fmt = " %3.18g";
		{
			x = c(1.0/Math.PI, 1.0, Math.PI);
			F f1 = new F(3, 1e6), f2 = new F(3, inf);
			System.out.println("## Inf df in pf etc.");
			print(fmt, f1.density(x));
			print(fmt, f2.density(x));
			print(fmt, f1.cumulative(x));
			print(fmt, f2.cumulative(x));

			f1 = new F(1e6, 5); f2 = new F(inf, 5);
			print(fmt, f1.density(x));
			print(fmt, f2.density(x));
			print(fmt, f1.cumulative(x));
			print(fmt, f2.cumulative(x));

			f1 = new F(inf, inf);
			print(f1.density(x));
			print(f1.cumulative(x));

			f1 = new F(5, inf);
			print(fmt, f1.cumulative(x));

			NonCentralF ncf = new NonCentralF(5, 1e6, 1);
			success &= printAllEqualScaled(c(0.06593319432457067641451, 0.47087998660583602061891, 0.97887586737053189356317),
				ncf.cumulative(x));
			ncf = new NonCentralF(5, 1e7, 1);
			success &= printAllEqualScaled(c(0.06593308950344137220334, 0.47088028378103324866899, 0.97887640681761456384891),
				ncf.cumulative(x));
			ncf = new NonCentralF(5, 1e8, 1);
			success &= printAllEqualScaled(c(0.06593307522941961595908, 0.47088029999414682258418, 0.97887645916474952390018),
				ncf.cumulative(x));
			ncf = new NonCentralF(5, inf, 1);
			print(fmt, ncf.cumulative(x));
			print(fmt, T.density(1, inf, false));
			print(fmt, NonCentralT.density(1, inf, 0, false));
			print(fmt, NonCentralT.density(1, inf, 1, false));
			print(fmt, NonCentralT.density(1, 1e6, 1, false));
			print(fmt, NonCentralT.density(1, 1e7, 1, false));
			print(fmt, NonCentralT.density(1, 1e8, 1, false));
			print(fmt, NonCentralT.density(1, 1e10, 1, false));
		}


		for (double _x : new double[] {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-100, 0}) {
			System.out.println(String.format("%3.18g %3.18g", _x, NonCentralT.density(_x, 2, 1, false)));
		}

		x = rep(vpow(10, c(colon(-3.,2.), colon(6.,9.), vtimes(10, colon(2.,30.)))), 12);
		boolean cur_success = true;
		for (double nu : new double[] {0.75, 1.2, 4.5, 999, 1e50}) {
			T t = new T(nu);
			double[] lfx = t.density(x, true);
			cur_success &= allFinite(lfx);
			cur_success &= allEqual(vexp(lfx), t.density(x));
		}
		success &= cur_success;
		if (!cur_success)
			System.err.println("Error at extreme values of T density");

		val = ChiSquare.cumulative(1, 1, true, false);
		double[] nus = vpow(2, seq(25, 34, 0.5));
		for (int i = 0; i < nus.length; i++) {
			double nu = nus[i];
			double _f = F.cumulative(1, 1, inf, true, false);
			if (!isEqual(_f, val)) {
				System.err.println(String.format("Error: target=%3.18g, pf(1,1,Inf) = %3.18g", val, _f));
				success = false;
			}
			double y = F.cumulative(1, 1, nu, true, false);
			double y_next = i == nus.length - 1 ? val : F.cumulative(1, 1, nus[i+1], true, false);
			if (y_next < y) {
				System.err.println(String.format("Not monotonic increasing: %3.18g %3.18g", y, y_next));
				success = false;
			}
			if (i == 0 && abs(y - (val - 7.21129e-9)) > 1e-11) {
				System.err.println(String.format("Precision error: %3.18g", y));
				success = false;
			}
		}

		if (Gamma.cumulative(inf, 1.1, 1, true, false) != 1) {
			System.err.println("Error at Gamma.cumulative(Inf, 1.1, 1, true, false)!");
			success = false;
		}

		System.out.println("## qgamma(q, *) should give {0,Inf} for q={0,1}");
		for (double sh : new double[] { 1.1, 0.5, 0.2, 0.15, 1e-2, 1e-10}) {
			if (Gamma.quantile(1, sh, 1, true, false) != inf) {
				System.err.println(String.format("Error at Gamma.cumulative(1, %f, 1, true, false)!", sh));
				success = false;
			}
			if (Gamma.quantile(0, sh, 1, true, false) != 0) {
				System.err.println(String.format("Error at Gamma.cumulative(0, %f, 1, true, false)!", sh));
				success = false;
			}
		}

		System.out.println("## In extreme left tail {PR#11030}");
		x = vtimes(1e-12, colon(10.,123.));
		double[] qg = new Gamma(19, 1).quantile(x),
			qg2 = new Gamma(11, 1).quantile(vtimes(1e-9, colon(1.,100.))),
			dqg = diff(qg, 1, 2),
			dqg2 = diff(qg2, 1, 2);
		if (!allLt(dqg, -6e-6)) {
			System.err.println("Error at Gamma.cumulative(x, 19, 1, true, false)!");
			success = false;
		}
		if (!allLt(dqg2, -6e-6)) {
			System.err.println("Error at Gamma.cumulative(x, 11, 1, true, false)!");
			success = false;
		}
		if (!allLt(vabs(vcomp(vdiv(new Gamma(19, 1).cumulative(qg), x))), 1e-13)) {
			System.err.println("Error at 1-Gamma.cumulative(x, 19, 1, true, false)!");
			success = false;
		}
		if (!isEqual(qg[0], 2.35047385139143)) {
			System.err.println("Error at Gamma.cumulative(1e-11, 19, 1, true, false)!=2.35047385139143");
			success = false;
		}
		if (!isEqual(qg2[29], 1.11512318734547)) {
			System.err.println("Error at Gamma.cumulative(3e-8, 11, 1, true, false)!=1.11512318734547");
			success = false;
		}
		// was non-continuous in R 2.6.2 and earlier

		for (double f2 : new double[] {0.5, 1, 2, 3, 4}) {
			if (F.density(0, 1, f2, false) != inf) {
				System.err.println(String.format("Error: F.density(0, 1, %f, false) != Inf", f2));
				success = false;
			}
			if (F.density(0, 2, f2, false) != 1) {
				System.err.println(String.format("Error: F.density(0, 2, %f, false) != 1", f2));
				success = false;
			}
			if (F.density(0, 3, f2, false) != 0) {
				System.err.println(String.format("Error: F.density(0, 3, %f, false) != 0", f2));
				success = false;
			}
		}
		// only the last one was ok in R 2.2.1 and earlier
	
		for (double x0 : new double[] {-2e-22, -2e-10, -2e-7, -2e-5}) {
			if (Binomial.cumulative(x0, 3, 1, true, false) != 0) {
				System.err.println(String.format("Error: Binomial.cumulative(%f, 3, 0.1, true, false) != 0", x0));
				success = false;
			}
			if (Binomial.density(x0, 3, 1, false) != 0) {
				System.err.println(String.format("Error: Binomial.density(%f, 3, 0.1, false) != 0", x0));
				success = false;
			}
		}
		// very small negatives were rounded to 0 in R 2.2.1 and earlier

		{
			System.out.println("## dbeta(*, ncp):");
			double[] a = new LogNormal(0, 1).random(100);
			for (double a_ : a) {
				if (Beta.density(0, 1, a_, false) != a_) {
					System.err.println(String.format("Error: Beta.density(0, 1, %3.18g, false) != %3.18g", a_, a_));
					success = false;
				}
				if (NonCentralBeta.density(0, 0.9, 2.2, a_, false) != inf) {
					System.err.println(String.format("Error: NonCentralBeta.density(0, 1, 2.2, %3.18g, false) != Inf", a_));
					success = false;
				}
			}
			if (NonCentralBeta.density(0, 0.9, 2.2, 0, false) != inf) {
				System.err.println("Error: NonCentralBeta.density(0, 1, 2.2, 0, false) != Inf");
				success = false;
			}
			double[] dbx = new double[] {0, 5, 80, 405, 1280, 3125, 6480, 12005, 20480, 32805,
				50000, 73205, 103680, 142805, 192080, 253125, 327680};
			double[] cc = vdiv(colon(0., 16.), 16.);
			success &= allEqual(vtimes(65536, new Beta(5,1).density(cc)), dbx);
			success &= allEqual(vexp(vplus(16*log(2), new Beta(5,1).density(cc, true))), dbx);

			System.out.println("## the first gave 0, the 2nd NaN in R <= 2.3.0; others use 'TRUE' values");
			val = NonCentralBeta.density(0.8, 0.5, 5, 1000, false);
			if (val != 3.001852308908624616864e-35) {
				System.err.println(String.format("Precision loss: NonCentralBeta.density(0.8, 0.5, 5, 1000, false) %3.18g != 3.001852308908624616864e-35", val));
				success = false;
			}
			// Integration tests --- We cannot do this until the integration engine is up and running
			// all.equal(1, integrate(dbeta, 0,1, 0.8, 0.5, ncp=1000)$value, tol=1e-4) // FIXME
			// all.equal(1, integrate(dbeta, 0,1, 0.5, 200, ncp=720)$value)
			// all.equal(1, integrate(dbeta, 0,1, 125, 200, ncp=2000)$value)
		}

		{
			System.out.println("## df(*, ncp):");
			x = seq(0, 10, 0.1);
			NonCentralF ncf = new NonCentralF(7, 5, 2.5);
			success &= printAllEqual(ncf.density(x),
				vdiv(vmin(ncf.cumulative(vplus(x, 1e-7)), ncf.cumulative(vmin(x, 1e-7))), 2e-7), 1e-6);
			for (double _x : x) {
				val = NonCentralF.density(0, 2, 4, _x, false);
				double val2 = NonCentralF.density(1e-300, 2, 4, _x, false);
				if (!isEqual(val, val2)) {
					System.err.println(String.format("Error: x = %f, NonCentralF.density(0, 2, 4, x, false) = %3.18g != NonCentralF.density(1e-300, 2, 4, x, false)  = %3.18g", x, val, val2));
					success = false;
				}
			}
		}

		{
			System.out.println("## qt(p ~ 0, df=1) - PR#9804");
			T t = new T(1);
			x = vpow(10, colon(-10., -20.));
			success &= printAllEqual(x, t.cumulative(t.quantile(x)), 1e-14);
			System.out.println("## Similarly for df = 2 --- both for p ~ 0  *and*  p ~ 1/2");
			System.out.println("## P ~ 0");
			val = T.quantile(-740, 2, true, true);
			if (!isEqualScaled(val, -exp(370)/sqrt(2))) {
				System.err.println(String.format("Precision loss: T.quantile(-740, 2, true, true) %3.18g != %3.18g", val, -exp(370)/sqrt(2)));
				success = false;
			}
			System.out.println("## P ~ 1 (=> p ~ 0.5):");
			double[] p5 = vplus(0.5, vpow(2, seq(-25, -40, -5)));
			t = new T(2);
			success &= printAllEqual(t.quantile(p5), c(8.429369702178821491988e-08, 2.634178031930877166753e-09, 8.231806349783991146103e-11, 2.572439484307497233157e-12));
			System.out.println("## qt(<large>, log = TRUE)  is now more finite and monotone (again!):");
			val = T.quantile(-1000, 4, true, true);
			if (!isEqualScaled(val, -4.930611e108)) {
				System.err.println(String.format("Precision loss: T.quantile(-1000, 4, true, true) %3.18g != -4.930611e108", val));
				success = false;
			}
			System.out.println("##almost: stopifnot(all(abs(5/6 - diff(log(qtp))) < 1e-11)):");
			x = new T(1.2).quantile(colon(-20., -850), false, true);
			p5 = diff(vlog(x));
			sort(p5);
			p5 = quantile(p5, c(0., 0.995));
			success &= printAllEqual(p5, c(5./6., 5./6.), 1e-11);

			System.out.println("## close to df=1 (where Taylor steps are important!):");
			t = new T(1.02);
			val = t.cumulative(t.quantile(-20, true, true), true, true);
			if (!isEqual(val, -20)) {
				System.err.println(String.format("Precision loss: T.cumulative(T.quantile(-20, 1.02, true, true), 1.02, true, true) == %3.18g != -20", val));
				success = false;
			}
			val = t.quantile(t.cumulative(-20, true, true), true, true);
			if (!isEqual(val, -20)) {
				System.err.println(String.format("Precision loss: T.quantile(T.cumulative(-20, 1.02, true, true), 1.02, true, true) == %3.18g != -20", val));
				success = false;
			}
			x = vpow(-2, colon(-10., -600));
			t = new T(1.1);
			if (!allGt(diff(vlog(t.quantile(x, true, true))), 0.6)) {
				System.err.println("Precision loss: diff(T.quantile(x, 1.1, true, true)) <= 0.6");
				success = false;
			}
			x = vpow(-2, colon(-20., -600));
			t = new T(1);
			if (mean(vabs(vmin(diff(t.quantile(x, true, true)), log(2)))) >= 1e-8) {
				System.err.println("Precision loss: diff(T.quantile(x, 1, true, true)) != log(2)");
				success = false;
			}
			t = new T(2);
			if (mean(vabs(vmin(diff(t.quantile(x, true, true)), log(sqrt(2))))) >= 1e-8) {
				System.err.println("Precision loss: diff(T.quantile(x, 2, true, true)) != log(sqrt(2))");
				success = false;
			}
			System.out.println("## Case, where log.p=TRUE was fine, but log.p=FALSE (default) gave NaN:");
			x = colon(40., 406.);
			t = new T(1.2);
			success &= printAllEqualScaled(x, vmin(t.cumulative(t.quantile(vexp(vmin(x))), true, true)));
		}

		{
			System.out.println("## pbeta(*, log=TRUE) {toms708} -- now improved tail behavior:");
			x = c(.01, .10, .25, .40, .55, .71, .98);
			double[] pbval = c(-0.04605755624088, -0.3182809860569, -0.7503593555585,
				-1.241555830932, -1.851527837938, -2.76044482378, -8.149862739881);
			success &= printAllEqualScaled(new Beta(0.8, 2).cumulative(x, false, true), pbval);
			success &= printAllEqualScaled(new Beta(2, 0.8).cumulative(vcomp(x), true, true), pbval);
			x = vmin(vpow(2, colon(0, 1022)));
			for (double nu : c(0.1, 0.2, 0.5, 1, 1.2, 2.2, 5, 10, 20, 50, 100, 200)) {
				if (!allFinite(new T(nu).cumulative(x, true, true))) {
					System.err.println(String.format("Bad numeric behavior: T.cumulative(x, %f, true, true))", nu));
					success = false;
				}
			}
			val = T.cumulative(pow(2, -30), 10, true, false);
			if (!isEqual(val, 0.50000000036238542)) {
				System.err.println(String.format("Precision loss: T.cumulative(2^-30, 10, true, true)) = %3.18g != 0.50000000036238542", val));
				success = false;
			}
		}

		{
			System.out.println("## rbinom(*, size) gave NaN for large size up to R <= 2.6.1");
			x = Binomial.random(100, Integer.MAX_VALUE, 1e-9, random);
			if (!allFinite(x) || sum(table(x)) != 100) {
				System.err.println("Produces NaN: Binomial.random(100, Integer.MAX_VALUE, 1e-9, random)");
				success = false;
			}
			x = Binomial.random(100, 10.*Integer.MAX_VALUE, 1e-10, random);
			if (!allFinite(x) || sum(table(x)) != 100) {
				System.err.println("Produces NaN: Binomial.random(100, 10*Integer.MAX_VALUE, 1e-10, random)");
				success = false;
			}
		}

		{
			System.out.println("## qf() with large df1, df2  and/or  small p:");
			val = F.quantile(1.0/4.0, inf, inf, true, false);
			if (val != 1) {
				System.err.println(String.format("F.quantile(1/4, inf, inf, true, false) != 1, but produces %3.18g", val));
				success = false;
			}
			F f = new F(12, 50);
			val = f.cumulative(f.quantile(1e-18));
			if (!isEqual(1, 1e-18 / val, 1e-10)) {
				System.err.println(String.format("F.cumulative(F.quantile(1e-18, 12, 50, true, false), true, false) != 1e-18, but produces %3.18g", val));
				success = false;
			}
			f = new F(1e60, 1e90);
			val = f.quantile(f.cumulative(0.01, true, true), true, true);
			if (!isEqual(0.01, val, 1e-4)) {
				System.err.println(String.format("F.quantile(F.cumulative(0.01, 1e60, 1e90, true, true), true, true) != 0.01, but produces %3.18g", val));
				success = false;
			}
		}

		{
			System.out.println("## qbeta(*, log.p) for \"border\" case:");
			val = Beta.quantile(-1e10, 50, 40, true, true);
			if (isInfinite(val)) {
				System.err.println("Beta.quantile(-1e10, 50, 40, true, true) is infinite");
				success = false;
			}
			val = Beta.quantile(-1e10, 2, 3, false, true);
			if (isInfinite(val)) {
				System.err.println("Beta.quantile(-1e10, 2, 3, false, true) is infinite");
				success = false;
			}
			// infinite loop or NaN in R <= 2.7.0
		}

		{
			System.out.println("## phyper(x, 0,0,0), notably for huge x");
			HyperGeometric h = new HyperGeometric(0, 0, 0);
			x = h.cumulative(c(0., 1, 2, 3, 1e67));
			success &= printAllEqual(rep(1, x.length), x);
		}

		{
			System.out.println("## plnorm(<= 0, . , log.p=TRUE)");
			if (LogNormal.cumulative(-1, 0, 1, false, true) != 0) {
				System.err.println("LogNormal.cumulative(-1, 0, 1, false, true) != 0");
				success = false;
			}
			if (LogNormal.cumulative(0, 0, 1, false, true) != 0) {
				System.err.println("LogNormal.cumulative(0, 0, 1, false, true) != 0");
				success = false;
			}
			if (LogNormal.cumulative(-1, 0, 1, true, true) != neginf) {
				System.err.println("LogNormal.cumulative(-1, 0, 1, true, true) != -Inf");
				success = false;
			}
			if (LogNormal.cumulative(0, 0, 1, true, true) != neginf) {
				System.err.println("LogNormal.cumulative(0, 0, 1, true, true) != -Inf");
				success = false;
			}
			// was wrongly == 'log.p=FALSE' up to R <= 2.7.1 (PR#11867)
		}

		{
			System.out.println("## pchisq(df=0) was wrong in 2.7.1; then, upto 2.10.1, P*(0,0) gave 1");
			ChiSquare chisq = new ChiSquare(0);
			x = c(-1., 0, 1);
			if (!VectorMath.allEqual(chisq.cumulative(x), c(0.,0,1))) {
				System.err.println("ChiSquare.cumulative(c(-1,0,1), 0, true, false) != c(0,0,1)");
				success = false;
			}
			if (!VectorMath.allEqual(chisq.cumulative(x, false, false), c(1.,1,0))) {
				System.err.println("ChiSquare.cumulative(c(-1,0,1), 0, false, false) != c(1,1,0)");
				success = false;
			}
		}

		{
			System.out.println("## dnbinom for extreme  size and/or mu :");
			x = vtimes(1e11, vpow(2, colon(1., 10)));
			double[] d = new double[x.length];
			for (int i = 0; i < x.length; i++) {
				double size = x[i];
				val = NegBinomial.density_mu(17, size, 20, false);
				double val2 = Poisson.density(17, 20, false);
				d[i] = val - val2;
				if (d[i] >= 0) {
					System.err.println(String.format("NegBinomial.density_mu(17, %3.18g, 20, false) = %3.18g > Poisson.density(17, 20, false) = %3.18g", size, val, val2));
					success = false;
				}
			}
			if (!allGt(diff(d), 0)) {
				System.err.println("diff(NegBinomial.density_mu(17, size, 20, false) - Poisson.density(17, 20, false)) <= 0");
				success = false;
			}
			// was wrong up to 2.7.1
			// The fix to the above, for x = 0, had a new cancellation problem
			for (double _x : vtimes(1e12, vpow(2, colon(0., 20)))) {
				val = NegBinomial.density_mu(0, 1, _x, false);
				if (!isEqual(1.0/(1.0+_x), val, 1e-13)) {
					System.err.println(String.format("NegBinomial.density_mu(0, 1, %3.18, false) = %3.18g != %3.18", _x, val, 1.0/(1.0+_x)));
					success = false;
				}
			}
			// was wrong in 2.7.2 (only)
		}

		{
			System.out.println("## Non-central F for large x");
			x = vtimes(1e16, vpow(1.1, colon(0., 20)));
			NonCentralF f = new NonCentralF(1, 1, 20);
			x = f.cumulative(x, false, true);
			if (!allGt(x, -0.047) && !allLt(x, -0.0455)) {
				System.err.println("NonCentralF.cumulative(large X, 1, 1, 20, false, true) jumped prematurely to -Inf");
				success = false;
			}
			// pf(*, log) jumped to -Inf prematurely in 2.8.0 and earlier
		}

		{
			System.out.println("## Non-central Chi^2 density for large x");
			NonCentralChiSquare nc = new NonCentralChiSquare(10, 1);
			x = nc.density(c(inf, 1e80, 1e50, 1e40));
			if (!VectorMath.allEqual(rep(0, x.length), x)) {
				System.err.println("NonCentralChiSquare.density(x, 10, 1, false) != 0 for huge x");
				success = false;
			}
			// did hang in 2.8.0 and earlier (PR#13309).
		}

		{
			System.out.println("## qbinom() .. particularly for large sizes, small prob:");
			x = c(.01, .001, .1, .25);
			double[] pr = vtimes(1e-7, colon(2., 20));
			double[] sizes = c(5000279., 5006279., 5016279);
			double[] ks = colon(0., 15);
			for (double sz: sizes) {
				for (double p : x) {
					for (double _pr : pr) {
						val = Binomial.quantile(p, sz, _pr, true, false);
						double val2 = Poisson.quantile(p, sz * _pr, true, false);
						if (val != val2) {
							System.err.println(String.format("p=%3.18g, sz=%3.18g, pr=%3.18g, Binomial.quantile(p, sz, pr, true, false) = %3.18g != Poisson.quantile(p, sz*pr, true, false) = %3.18", p, sz, _pr, val, val2));
							success = false;
						}
					}
				}
				for (double _pr : pr) {
					for (double _ks : ks) {
						val = Binomial.cumulative(_ks, sz, _pr, true, false);
						double val2 = Binomial.quantile(val, sz, _pr, true, false);
						if (val2 != _ks) {
							System.err.println(String.format("Binomial.quantile(Binomial.cumulative(%d, %f, %3.18g, true, false) = %3.18g != %f", _ks, sz, _pr, val, val2));
							success = false;
						}
					}
				}
			}
			//  do_search() in qbinom() contained a thinko up to 2.9.0 (PR#13711)
		}

		{
			System.out.println("## pbeta(x, a,b, log=TRUE)  for small x and a  is ~ log-linear");
			x = vpow(2, colon(-200., -10));
			for (double a : c(1e-8, 1e-12, 16e-16, 4e-16)) {
				for (double b : c(0.6, 1, 2, 10)) {
					double[] dp = diff(new Beta(a, b).cumulative(x, true, true));
					val = sd(dp) / mean(dp);
					if (val >= 0.0007) {
						System.err.println(String.format("a=%f, b=%f, dp=diff(Beta(x, a, b, true, true)), sd(dp)/mean(db) = %3.18> 0.0007", a, b, val));
						success = false;
					}
				}
			}
			// had  accidental cancellation '1 - w'
		}

		{
			System.out.println("## qgamma(p, a) for small a and (hence) small p");
			System.out.println("## pgamma(x, a) for very very small a");

			val = Gamma.quantile(0.99, 0.0001, 1, true, false);
			if (val != 0) {
				System.err.println(String.format("Gamma.quantile(0.99,  0.0001, 1, true, false) = %3.18g != 0", val));
				success = false;
			}
			x = vplus(1.0, vtimes(1e-7, c(-1., 1)));
			double[] pg = new Gamma(pow(2, -64), 1).cumulative(x, false, false);
			if (abs(pg[1] - 1.18928249197237758088243e-20) >= 1e-33) {
				System.err.println(String.format("Precision loss: Gamma.cumulative(%g, 2^-64, 1, true, false) %3.18g != 1.18928249197237758088243e-20", x[1], pg[1]));
				success = false;
			}

			val = abs(diff(pg)[0] + diff(x)[0]*Gamma.density(1, pow(2,-64), 1, false));
			if (val >= 1e-13*mean(pg)) {
				System.err.println(String.format("abs(diff(pg)[0] + diff(x)[0]*Gamma.density(1, pow(2,-64), 1, false)) = %3.18g", val));
				success = false;
			}

			for (double a: vpow(2, vmin(seq(10., 1000, .25)))) {
				double
					q1c = Gamma.quantile(1e-100, a, 1, false, false),
					q3c = Gamma.quantile(1e-300, a, 1, false, false);
				if (q1c > 0) {
					double p1c = Gamma.cumulative(q1c, a, 1, false, false);
					if (abs(1 - p1c/1e-100) >= 10e-13) {
						System.err.println(String.format("Precision loss: a=%f. Gamma.cumulative(Gamma.quantile(1e-300, a, 1, false, false)) != 1e-100", a));
						success = false;
					}
				}
				if (q3c > 0) {
					double p3c = Gamma.cumulative(q3c, a, 1, false, false);
					if (abs(1 - p3c/1e-300) >= 28e-13) {
						System.err.println(String.format("Precision loss: a=%f. Gamma.cumulative(Gamma.quantile(1e-300, a, 1, false, false)) != 1e-100", a));
						success = false;
					}
				}
			}
		}
		//*/

		{
			System.out.println("## gave Inf as p==1 was checked *before* lambda==0");
			x = vdiv(colon(0., 8), 8);
			if (!VectorMath.allEqual(new Poisson(0).quantile(x), rep(0, x.length))) {
				System.err.println("Poisson.quantile(x, lambda=0, true, false) != 0");
				success = false;
			}
		}
		return success;
	}

	public static final void main(String[] args) {
		//System.out.println(String.format("%3.18g", MathFunctions.gammafn(13.51)));
		test_binom();
		test_geom();
		test_hyper();
		test_negbin();
		test_poisson();
		test_signrank();
		test_wilcox();
		test_gamma();
		test_noncentralchisq();
		test_beta();
		test_normal();
		test_random();
		test_extreme();
	}
}
