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
import static jdistlib.MathFunctions.isInfinite;
import static jdistlib.MathFunctions.round;
import static jdistlib.MathFunctions.gammafn;
import static jdistlib.Constants.DBL_EPSILON;
import static jdistlib.util.Utilities.*;

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
		return (Double.isNaN(a) && Double.isNaN(b)) || (a == b || abs(a - b) < tol);
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

	static final double[] pows(double x, double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	static final double[] pows(double x, int[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	static final double[] mins(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = -e[i];
		return v;
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
					if (!isEqual(d1, d3)) {
						System.err.println(String.format("Error: scaled dgamma = %3.18g, manually comp dgamma = %3.18g", d1, d3));
						System.err.println(String.format("x = %g, sh = %g, sig = %g, Ga(sh) = %3.30g", x, sh, sig, Ga));
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
		// Includes T disribution apparently
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
				if (!isEqual(log_pz, pz_comp)) {
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
		double[] yy = c(colon(1., 50), pows(10, c(colon(3,10), c(20,50,150,250))));
		yy = c(mins(yy), new double[] {0}, yy);
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

	public static final void main(String[] args) {
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
	}
}
