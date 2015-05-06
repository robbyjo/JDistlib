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

import java.util.Map;

import jdistlib.disttest.DistributionTest;
import jdistlib.disttest.NormalityTest;
import jdistlib.disttest.TestKind;
import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.math.VectorMath;
import jdistlib.rng.MersenneTwister;
import jdistlib.rng.RandomEngine;

import org.junit.Test;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static jdistlib.DebugFun.*;
import static jdistlib.disttest.DistributionTest.*;
import static jdistlib.math.Constants.DBL_EPSILON;
import static jdistlib.math.Constants.DBL_MAX;
import static jdistlib.math.MathFunctions.gammafn;
import static jdistlib.math.MathFunctions.isInfinite;
import static jdistlib.math.MathFunctions.round;
import static jdistlib.math.VectorMath.*;
import static jdistlib.util.Utilities.c;
import static jdistlib.util.Utilities.colon;
import static jdistlib.util.Utilities.print;
import static jdistlib.util.Utilities.rec;
import static jdistlib.util.Utilities.rep;
import static jdistlib.util.Utilities.rep_each;
import static jdistlib.util.Utilities.rev;
import static jdistlib.util.Utilities.sample_int;
import static jdistlib.util.Utilities.seq;
import static jdistlib.util.Utilities.sort;

/**
 * Ported tests/d-p-q-r-tests.R plus some more.
 * @author Roby Joehanes
 *
 */
public class TestDPQR {
	static RandomEngine random = new MersenneTwister(123L);
	static final double defaultNumericalError = DBL_EPSILON * 64;

	public static final void setRandomEngine(RandomEngine rng) {
		random = rng;
	}

	public static final boolean dkwtest(GenericDistribution d) {
		return dkwtest(d, 10000, 0.001);
	}

	/**
	 * <P>RNG tests using DKW inequality for rate of convergence
	 * 
	 * <P>P(sup | F_n - F | > t) < 2 exp(-2nt^2)
	 * 
	 * <P>The 2 in front of exp() was derived by Massart. It is the best possible
	 * constant valid uniformly in t,n,F. For large n*t^2 this agrees with the
	 * large-sample approximation to the Kolmogorov-Smirnov statistic.
	 * 
	 * <P>Taken from p-r-random-tests.R
	 * @param d
	 * @param n
	 * @param p0
	 * @return
	 */
	public static final boolean dkwtest(GenericDistribution d, int n, double p0) {
		d.setRandomEngine(random);
		double[] x = d.random(n);
		Map<String, Integer> tbl_x = table(vsignif(x, 12));
		double[] xi = as_numeric(tbl_x.keySet());
		sort(xi);
		double[] f = d.cumulative(xi);
		double[] tx = new double[xi.length];
		for (int i = 0; i < xi.length; i++)
			tx[i] = tbl_x.get(String.valueOf(xi[i]));
		double[] fhat = vdiv(cumsum(tx), n);
		double s =  max(vabs(vmin(fhat, f)));
		//double pdkwbound = Math.min(1, 2*exp(-2*n*s*s)); // P-value of s
		double qdkwbound = sqrt(log(p0/2)/(-2*n));
		return s < qdkwbound;
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
					if (!VectorMath.isEqual(d1, d3, 2 * defaultNumericalError)) { // Still within error limit
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
			if (!VectorMath.isEqual(p, ans[i], 2e-15)) {
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
					double val = NonCentralChiSquare.cumulative(xB, df, ncp, true, false);
					if (!isEqual(val, 1)) {
						System.err.println(String.format("Error: pchisq(x=%g, df=%g, ncp=%g) = %3.18g. Correct answer = 1", xB, df, ncp, val));
						success = false;
					}
				}
			}
		}
		double cor_val = 49.77662465605547481573; // This is the value I took from R
		//double cor_val = 49.7766246561514; // This is the value given in d-p-q-r-test.R
		double val = NonCentralChiSquare.quantile(0.025, 31, 1, false, false); // Inf. loop PR#875
		if (!VectorMath.isEqual(val, cor_val, 1e-11)) {
			System.err.println(String.format("Error: qchisq(x=0.025, df=31, ncp=1) = %3.18g. Correct answer = %3.18g", val, cor_val));
			success = false;
		}

		for (double df : new double[] {0.1, 0.5, 1.5, 4.7, 10, 20, 50, 100}) {
			System.out.print("df =" + df);
			cur_success = true;
			double dtol = 1e-12 * (2 < df && df <= 50 ? 64 : (df > 50 ? 20000 : 501));
			for (double xx : new double[] {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9, 1.2, df+3, df+7, df+20, df+30, df+35, df+38}) {
				double pval = NonCentralChiSquare.cumulative(xx, df, 1, true, false);
				double qval = NonCentralChiSquare.quantile(pval, df, 1, true, false);
				if (!VectorMath.isEqual(qval, xx, dtol)) {
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
			if (i < 29 & !VectorMath.isEqual(q0, q1, 1e-5)) {
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
					if (!VectorMath.isEqual(v1, v2, 1e-11)) {
						System.err.println(String.format("Error: p=%g, a=%3.18g, b=%3.18g, dbeta(p,a,b) = %3.18g, exp(dbeta(p,a,b,TRUE)) = %3.18g", p, a, b, v1, v2));
						success = false;
					}
				}
			}
		}
		/* This part of the test is highly dependent on the random number generation and is therefore not applicable
		 * sp <- sample(pab, 50)
		 *	if(!interactive()) stopifnot(which(isI <- sp == -Inf) == c(3, 11, 15, 20, 22, 23, 30, 39, 42, 43, 46, 47, 49),
		 *   all.equal(range(sp[!isI]), c(-2906.123981, 2.197270387)))
		 */
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
		// R 3.2.x seems to change this part into an imperative form (i.e., using stopifnot). But the test is still fundamentally the same.
		// JDistlib got it right too.
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
		boolean cur_success = VectorMath.isEqual(q1, -0.6744897501960817054467, 1e-15) &&
			VectorMath.isEqual(q2, -3.0902323061678131921326, 1e-15) &&
			VectorMath.isEqual(q3, -9.2623400897984051738376, 1e-15);
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
			if (!VectorMath.isEqual(pz, pz_comp, 1e-15)) {
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
				if (!VectorMath.isEqual(pt, pt_comp, 1e-15)) {
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
				if (!VectorMath.isEqual(log_pz, pz_comp, 2 * defaultNumericalError)) {
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
				if (!VectorMath.isEqual(z, qnorm_pz, 1e-12)) {
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
		success &= printBool(VectorMath.isEqual(abs(Gamma.cumulative(30, 100, 1, false, true)), 7.3384686328784e-24, 1e-36));
		success &= printBool(isEqual(1, Cauchy.cumulative(-1e20, 0, 1, true, false) / 3.18309886183791e-21));
		success &= printBool(isEqual(1, Cauchy.cumulative(+1e15, 0, 1, true, true) / -3.18309886183791e-16)); // PR#6756

		Cauchy cauchy = new Cauchy(0, 1);
		double[] ex = new double[] {1,2,5,10,15,20,25,50,100,200,300, Double.POSITIVE_INFINITY};
		x = vpow(10, ex);
		for (double _x : x)
			if (_x > 1e10)
				printBool(VectorMath.isEqual(T.cumulative(-_x, 1, true, false), cauchy.cumulative(-_x), 1e-15));
		System.out.println("## for PR#7902:");
		double[] rec_x = rec(x), mins_x = vmin(x);
		success &= printAllEqualScaled(mins_x, cauchy.quantile(cauchy.cumulative(mins_x)));
		success &= printAllEqualScaled(x, cauchy.quantile(cauchy.cumulative(x, true, true), true, true));
		success &= printAllEqual(rec_x, cauchy.quantile(cauchy.cumulative(rec_x)));
		ex = vmin(c(rev(rec_x), ex));
		success &= printAllEqualScaled(ex, cauchy.quantile(cauchy.cumulative(ex, true, true), true, true));

		x = new double[] { 0, 1};
		ex = new double[] { neginf, inf };
		if (!DebugFun.allEqual(cauchy.cumulative(ex), x) ||
			!DebugFun.allEqual(cauchy.quantile(x), ex) ||
			!DebugFun.allEqual(cauchy.quantile(new double[] {neginf, 0}, true, true), ex)) {
			System.err.println("Boundary exception error in Cauchy distribution");
			success = false;
		}

		//## PR#15521 :
		success &= VectorMath.isEqualScaled(cauchy.quantile(1 - 1.0/4096), 1303.7970381453319163, 1e-14);

		System.out.println("## PR#6757:");
		if (!VectorMath.isEqualScaled(pow(1e-23, 12), Binomial.cumulative(11, 12, 1e-23, false, false), 1e-12)) {
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
		// ## where all completely off in R 2.0.1
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
			cur_success &= DebugFun.allEqual(vexp(lfx), t.density(x));
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
				if ((val = NonCentralBeta.density(0, 1, a_, 0, false)) != a_) {
					System.err.println(String.format("Error: Beta.density(0, 1, %3.18g, false) %3.18g != %3.18g", a_, val, a_));
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
			success &= DebugFun.allEqual(vtimes(65536, new Beta(5,1).density(cc)), dbx);
			success &= DebugFun.allEqual(vexp(vplus(16*log(2), new Beta(5,1).density(cc, true))), dbx);

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
			if (!VectorMath.isEqual(1, 1e-18 / val, 1e-10)) {
				System.err.println(String.format("F.cumulative(F.quantile(1e-18, 12, 50, true, false), true, false) != 1e-18, but produces %3.18g", val));
				success = false;
			}
			f = new F(1e60, 1e90);
			val = f.quantile(f.cumulative(0.01, true, true), true, true);
			if (!VectorMath.isEqual(0.01, val, 1e-4)) {
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
				if (!VectorMath.isEqual(1.0/(1.0+_x), val, 1e-13)) {
					System.err.println(String.format("NegBinomial.density_mu(0, 1, %3.18, false) = %3.18g != %3.18", _x, val, 1.0/(1.0+_x)));
					success = false;
				}
			}
			// was wrong in 2.7.2 (only)
		}

		{
			final double maxerr = 9*DBL_EPSILON;
			for (int expo: new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 150, 200, 250, 300}) {
				for (int i = 1; i <= 7; i++) {
					double
						mu = i * pow(10, expo),
						NB = NegBinomial.density_mu(5, 1e305, mu, true),
						P = Poisson.density(5, mu, true),
						abserr = abs(rErr(NB, P));
					if (abserr >= maxerr) {
						System.err.println(String.format("abs(NegBinomial.density_mu(5, 1e305, mu=%g, log=true) - Poisson.density(5, %g, log=true)) = %3.18g > %3.18g (%3.18g vs. %3.18g)", mu, mu, abserr, maxerr, NB, P));
						success = false;
					}
				}
			}
			//## wrong in 3.1.0 and earlier
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
			for (double sz: c(5000279., 5006279., 5016279)) {
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
					for (int ks = 0; ks <= 15; ks++) {
						val = Binomial.cumulative(ks, sz, _pr, true, false);
						double val2 = Binomial.quantile(val, sz, _pr, true, false);
						if (val2 != ks) {
							System.err.println(String.format("Binomial.quantile(Binomial.cumulative(%d, %f, %3.18g, true, false) = %3.18g != %f", ks, sz, _pr, val, val2));
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

			val = Gamma.quantile(0.99, 0.00001, 1, true, false);
			if (val != 0) {
				System.err.println(String.format("Gamma.quantile(0.99,  0.00001, 1, true, false) = %3.18g != 0", val));
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

		{
			System.out.println("## gave Inf as p==1 was checked *before* lambda==0");
			x = vdiv(colon(0., 8), 8);
			if (!VectorMath.allEqual(new Poisson(0).quantile(x), rep(0, x.length))) {
				System.err.println("Poisson.quantile(x, lambda=0, true, false) != 0");
				success = false;
			}
		}

		{
			System.out.println("## extreme tail of non-central chisquare");
			val = NonCentralChiSquare.cumulative(200, 4, 0.001, true, true);
			if (!isEqualScaled(-3.851e-42, val)) {
				System.err.println(String.format("Precision loss: NonCentralChiSquare.cumulative(200, 4, 0.001, true, true) = %3.18g != -3.851e-42", val));
				success = false;
			}
			// ## jumped to zero too early up to R 2.10.1 (PR#14216)
			System.out.println("## left \"extreme tail\"");
			// The following test is a response to bug PR#15635
			double[] lp = new double[201];
			for (int i = 0; i <= 200; i++) {
				val = pow(2, -i);
				lp[i] = NonCentralChiSquare.cumulative(val, 100, 1, true, true);
				if (isInfinite(lp[i])) {
					System.err.println(String.format("Precision loss: NonCentralChiSquare.cumulative(%3.18g, 100, 1, true, true) is not finite", val));
					success = false;
				}
				if (lp[i] >= -184) {
					System.err.println(String.format("Precision loss: NonCentralChiSquare.cumulative(%3.18g, 100, 1, true, true) = %3.18g >= -184", val, lp[i]));
					success = false;
				}
			}
			if (!isEqualScaled(lp[200], -7115.10693158)) {
				System.err.println(String.format("Precision loss: NonCentralChiSquare.cumulative(2^-200, 100, 1, true, true) = %3.18g != -7.115.10693158", lp[201]));
				success = false;
			}
			double[] dlp = diff(lp);
			double[] dd = new double[dlp.length - 30];
			for (int i = 0; i < dd.length; i++)
				dd[i] = abs(dlp[i + 30] - -34.65735902799);
			print(range(dd));
			for (int i = 0; i < dlp.length; i++)
				if (-34.66 >= dlp[i] || dlp[i] >= -34.41) {
					System.err.println(String.format("Precision loss: %3.18g not within -34.41 and -34.66", dlp[i]));
					success = false;
				}
			for (int i = 0; i < dd.length; i++)
				if (dd[i] >= 1e-8) {
					System.err.println(String.format("Precision loss: Differential error of %3.18g >= 1e-8 detected!", dd[i]));
					success = false;
				}
			for (double e: c(0., 2e-16)) {
				val = NonCentralChiSquare.cumulative(1, 1.01, 80*(1-e), true, true);
				if (!isEqualScaled(val, -34.57369629)) {
					System.err.println(String.format("Continuity error detected: NonCentralChiSquare.cumulative(1, 1.01, 80*(1-%g), true, true) = %3.18g != -34.57369629", e, val));
					success = false;
				}
				val = NonCentralChiSquare.cumulative(2, 1.01, 80*(1-e), true, true);
				if (!isEqualScaled(val, -31.31514671)) {
					System.err.println(String.format("Continuity error detected: NonCentralChiSquare.cumulative(2, 1.01, 80*(1-%g), true, true) = %3.18g != -31.31514671", e, val));
					success = false;
				}
			}
			// Test for PR#15635 ends
		}

		{
			System.out.println("## logit() == qlogit() on the right extreme:");
			x = c(colon(10., 80.), vplus(80, vtimes(5, colon(1., 24))), vplus(200, vtimes(20, colon(1., 25))));
			Logistic l = new Logistic(0, 1);
			success &= printAllEqual(x, l.quantile(l.cumulative(x, true, true), true, true));
			// qlogis() gave Inf much too early for R <= 2.12.1

			System.out.println("## Part 2:");
			x = c(x, seq(700., 800, 10));
			success &= printAllEqual(x, l.quantile(l.cumulative(x, false, true), false, true));
			// plogis() underflowed to -Inf too early for R <= 2.15.0
		}

		{
			System.out.println("## log upper tail pbeta():");
			x = vdiv(colon(25., 50.), 128);
			double[] pbx = new Beta(1./2, 2200).cumulative(x, false, true);
			double[] dp = diff(pbx);
			double[] d2p = diff(dp);
			if (!allGt(pbx, -1094) || !allLt(pbx, -481.66)) {
				System.err.println("Anomalous Beta.cumulative(x, 0.5, 2200, false, true)");
				success = false;
			}
			if (!allGt(dp, -29) || !allLt(dp, -20)) {
				System.err.println("Anomalous Beta.cumulative(x, 0.5, 2200, false, true) at the first derivative");
				success = false;
			}
			if (!allGt(d2p, -.36) || !allLt(d2p, -.2)) {
				System.err.println("Anomalous Beta.cumulative(x, 0.5, 2200, false, true) at the second derivative");
				success = false;
			}
			double[] b = new double[51];
			double[] y = new double[b.length];
			for (int i = 0; i < b.length; i++) {
				b[i] = 2200*pow(2, i);
				y[i] = log(-Beta.cumulative(.28, 1./2, b[i], false, true)) + 1.113;
				//if (!isEqualScaled(log(b[i]), y[i], 0.00002)) {
				//	System.err.println(String.format("Precision loss at log(-Beta.cumulative(x, 0.5, %3.18g, false, true))+1.113 = %3.18g != %3.18g", b[i], y[i], log(b[i])));
				//	success = false;
				//}
			}
			b = vlog(b);
			val = mean(vabs(vmin(b, y))) / mean(b);
			printBool(abs(val) <= 0.00002);
			if (abs(val) > 0.00002) {
				System.err.println("Precision loss at log(-Beta.cumulative(x, 0.5, 2200+, false, true))");
				success = false;
			}
			// pbx had two -Inf; y was all Inf  for R <= 2.15.3;  PR#15162
		}
		{
			System.out.println("## dnorm(x) for \"large\" |x|");
			val = Normal.density(35+pow(3, -9), 0, 1, false);
			if (abs(1 - val/ 3.933395747534971e-267) >= 1e-15) {
				System.err.println(String.format("Precision loss at Normal.density(35+3^-9, 0, 1, false) %3.18g != 3.933395747534971e-267", val));
				success = false;
			}
		}
		{
			System.out.println("## pbeta(x, <small a>,<small b>, .., log):");
			final double loghalf = log(1.0/2), pow2_60 = pow(2, -60);
			double[] ldp = new double[25];
			for (int i = 0; i < ldp.length; i++)
				ldp[i] = Beta.cumulative(0.5, pow(2, -91-i), pow2_60, true, true);
			ldp = diff(vlog(diff(ldp)));
			for (int i = 0; i < ldp.length; i++)
				if (abs(ldp[i] - loghalf) >= 1e-9) {
					System.err.println(String.format("Precision loss %3.18g vs %3.18g", ldp[i], loghalf));
					success = false;
				}
			// ## pbeta(*, log) lost all precision here, for R <= 3.0.x (PR#15641)
		}
		{
			System.out.println("## \"stair function\" effect (from denormalized numbers)");
			double[] dpx = new double[101];
			for (int i = 0; i < 101; i++) {
				dpx[i] = Beta.cumulative(.9833 + i*1e-6, 43779, 0.06728, true, true);
				if (i == 0) {
					if (!isEqualScaled(dpx[0], -746.0986886924, 1e-12)) {
						System.err.println(String.format("Precision loss pbeta(.9833, 43779, 0.06728, true, true) = %3.18g vs -746.0986886924", dpx[0]));
						success = false;
					}
				}
			}
			dpx = diff(dpx);
			for (int i = 0; i < dpx.length; i++) {
				if (dpx[i] <= 0.0445741 || dpx[i] >= 0.0445783) {
					System.err.println(String.format("Stair case detected %3.18g outside (0.0445741, 0.0445783) range", dpx[i]));
					success = false;
				}
			}
			dpx = diff(dpx);
			for (int i = 0; i < dpx.length; i++) {
				if (dpx[i] <= -4.2e-8 || dpx[i] >= -4.18e-8) {
					System.err.println(String.format("Stair case detected %3.18g outside (-4.2e-8, -4.18e-8) range", dpx[i]));
					success = false;
				}
			}
			// ## were way off in R <= 3.1.0
		}
		{
			System.out.println("## Infinite loop check");
			long time1, cB, c1, c2;
			time1 = System.currentTimeMillis();
			double p0 = Beta.cumulative(.9999, 1e30, 1.001, true, true);
			cB = (long) Math.max(.001, System.currentTimeMillis() - time1);
			time1 = System.currentTimeMillis();
			double p1 = Beta.cumulative(1 - 1e-9, 1e30, 1.001, true, true);
			c1 = System.currentTimeMillis() - time1;
			time1 = System.currentTimeMillis();
			double p2 = Beta.cumulative(1 - 1e-12, 1e30, 1.001, true, true);
			c2 = System.currentTimeMillis() - time1;
			if (!VectorMath.isEqualScaled(p0, -1.000050003333e26, 1e-10)) {
				System.err.println(String.format("Beta.cumulative(.9999, 1e30, 1.001, true, true) = %3.18g != -1.000050003333e26", p0));
				success = false;
			}
			if (!VectorMath.isEqualScaled(p1, -1e-21, 1e-6)) {
				System.err.println(String.format("Beta.cumulative(1 - 1e-9, 1e30, 1.001, true, true) = %3.18g != -1e21", p1));
				success = false;
			}
			if (!VectorMath.isEqualScaled(p2, -9.9997788e17, 1e-14)) {
				System.err.println(String.format("Beta.cumulative(1 - 1e-12, 1e30, 1.001, true, true) = %3.18g != -9.9997788e17", p2));
				success = false;
			}
			if (c1 > 1000*cB || c2 > 1000*cB) {
				System.err.println("Near infinite loop when computing Beta cumulative with x close to 1, huge alpha, and beta near to 1");
				success = false;
			}
			// ## (almost?) infinite loop in R <= 3.1.0
		}
		// For bug PR#15734
		{
			System.out.println("## pbinom(), dbinom(), dhyper(),.. : R allows \"almost integer\" n");
			double[] arr = sample_int(10000, 1000);
			for (int i = 0; i < arr.length; i++) {
				double[] n = vtimes((arr[i] / 100), vpow(10., colon(2, 20)));
				for (int j = 0; j < n.length; j++) {
					val = Binomial.density(1, n[j], 0.5, false);
					if (Double.isNaN(val)) {
						System.err.println(String.format("NaN detected in Binomial.density(1, %g, 0.5, false)", n[j]));
						success = false;
					}
					val = Binomial.cumulative(1, n[j], 0.5, true, false);
					if (Double.isNaN(val)) {
						System.err.println(String.format("NaN detected in Binomial.cumulative(1, %g, 0.5, true, false)", n[j]));
						success = false;
					}
					val = Poisson.density(n[j], n[j], false);
					if (Double.isNaN(val)) {
						System.err.println(String.format("NaN detected in Poisson.density(%g, %g, false)", n[j], n[j]));
						success = false;
					}
					val = HyperGeometric.density(n[j]+1, n[j]+5, n[j]+5, n[j], false);
					if (Double.isNaN(val)) {
						System.err.println(String.format("NaN detected in HyperGeometric.density(%g, %g, %g, %g, false)", n[j]+1, n[j]+5, n[j]+5, n[j]));
						success = false;
					}
				}
			}
			System.out.println("## check was too tight for large n in R <= 3.1.0 (PR#15734)");
		}
		{
			System.out.println("## [dpqr]beta(*, a,b) where a and/or b are Inf");
			if (Beta.cumulative(.1, Double.POSITIVE_INFINITY, 40, true, false) != 0) {
				System.err.println("Beta.cumulative(.1, Inf, 40, true, false) != 0");
				success = false;
			}
			if (Beta.cumulative(.5, 40, Double.POSITIVE_INFINITY, true, false) != 1) {
				System.err.println("Beta.cumulative(.5, 40, Inf, true, false) != 1");
				success = false;
			}
			if (Beta.cumulative(.4, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, true, false) != 0) {
				System.err.println("Beta.cumulative(.5, Inf, Inf, true, false) != 0");
				success = false;
			}
			if (Beta.cumulative(.5, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, true, false) != 1) {
				System.err.println("Beta.cumulative(.5, Inf, Inf, true, false) != 1");
				success = false;
			}
			// ## gave infinite loop (or NaN) in R <= 3.1.0
			if (Beta.quantile(.9, Double.POSITIVE_INFINITY, 100, true, false) != 1) {
				System.err.println("Beta.quantile(.9, Inf, 100, true, false) != 1");
				success = false;
			}
			if (Beta.quantile(.1, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, true, false) != 0.5) {
				System.err.println("Beta.quantile(.1, Inf, Inf, true, false) != 0.5");
				success = false;
			}
			System.out.println("## range check (in \"close\" cases):");
			for (double v: vpow(-2, c(-10, -100, -1000))) {
				if (!Double.isNaN(Beta.quantile(v, 2, 3, true, true))) {
					System.err.println(String.format("Beta.quantile(%3.18g, 2, 3, true, true) != NaN", v));
					success = false;
				}
			}
			for (double v: c(-.1, -1e-300, 1.25)) {
				if (!Double.isNaN(Beta.quantile(v, 2, 3, true, false))) {
					System.err.println(String.format("Beta.quantile(%3.18g, 2, 3, true, false) != NaN", v));
					success = false;
				}
			}
		}
		{
			System.out.println("## lognormal boundary case sdlog = 0:");
			for (int i = 0; i <= 8; i++) {
				double p = i / 8.0;
				int mean = i % 2 == 0 ? 1 : 2;
				double v1 = LogNormal.quantile(p, mean, 0, true, false),
					v2 = LogNormal.quantile(p, mean, 1e-200, true, false);
				if (!isEqual(v1, v2)) {
					System.err.println(String.format("LogNormal.quantile(%3.4g, %d, 0, true, false) %3.18g != %3.18g LogNormal.quantile(%3.4g, %d, 0, true, false)", p, mean, v1, v2, p, mean));
					success = false;
				}
			}
			for (int i = -10; i <= 10; i++) {
				val = LogNormal.density(pow(2, i), 0, 0, false);
				if (i == 0) {
					if (val != Double.POSITIVE_INFINITY) {
						System.err.println(String.format("LogNormal.density(1, 0, 0, false) %3.18g != Inf", val));
						success = false;
					}
				} else {
					if (val != 0) {
						System.err.println(String.format("LogNormal.density(%3.18g, 0, 0, false) %3.18g != 0", pow(2, i), val));
						success = false;
					}
				}
			}
		}
		{
			System.out.println("## qbeta(*, a,b) when  a,b << 1 : can easily fail");
			System.out.println(Beta.quantile(pow(2, -28), 0.125, pow(2, -26), true, false));
			double a = 1.0/8, oldp = 0;
			for (int i = 4; i <= 200; i++) {
				double
					b = pow(2, -i),
					alpha = b / 4,
					qq = Beta.quantile(alpha, a, b, true, false),
					pp = Beta.cumulative(qq, a, b, true, false);
				if (pp <= 0 || (i > 4 && pp - oldp >= 0) || abs(1 - pp/alpha) >= 4e-15) {
					System.err.println(String.format("b = %3.18g, alpha = %3.18g, Beta.quantile(alpha, 0.125, b, true, false) %3.18g != Beta.cumulative(%3.18g, 0.125, b, true, false) %3.18g",
						b, alpha, qq, qq, pp));
					System.err.println(String.format("diff(pp) = %3.18g", pp - oldp));
					success = false;
				}
				oldp = pp;
			}
		}
		{
			System.out.println("## orig. qbeta() using *many* Newton steps; case where we \"know the truth\"");
			x = vpow(2, c(-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-100,-200,-250,-300,-400,-500,-600,-700,-800,-900,-1000));
			double[] pb = new double [] { //## via Rmpfr's roundMpfr(pbetaI(x, a,b, log.p=TRUE, precBits = 2048), 64) :
			    -40.7588797271766572448, -57.7574063441183625303, -74.9287878018119846216,
			    -92.1806244636893542185, -109.471318248524419364, -126.781111923947395655,
			    -144.100375042814531426, -161.424352961544612370, -178.750683324909148353,
			    -196.078188674895169383, -213.406281209657976525, -230.734667259724367416,
			    -248.063200048177428608, -1721.00081201679567511, -3453.86876341665894863,
			    -4320.30273911659058550, -5186.73671481652222237, -6919.60466621638549567,
			    -8652.47261761624876897, -10385.3405690161120427, -12118.2085204159753165,
			    -13851.0764718158385902, -15583.9444232157018631, -17316.8123746155651368
			};
			for (int i = 0; i < x.length; i++) {
				double qp = Beta.quantile(pb[i], 25, 6, true, true);
				if (qp <= 0 || !isEqual(qp, x[i], 1e-15)) {
					System.err.println(String.format("Beta.quantile(%3.18g, 25, 6, true, true) = %3.18g <= 0 != %3.18g", pb[i], qp, x[i]));
					success = false;
				}
			}
		}
		{
			System.out.println("## qbeta(), PR#15755");
			double qp = Beta.quantile(0.6948886, 0.0672788, 226390, true, false);
			if (qp >= 2e-8 || !isEqual(0.6948886, val = Beta.cumulative(qp, 0.0672788, 226390, true, false))) {
				System.err.println(String.format("Beta.quantile(0.6948886, 0.0672788, 226390, true, false) = %3.18g", qp));
				System.err.println(String.format("Beta.cumulative(%3.18g, 0.0672788, 226390, true, false) = %3.18g != 0.6948886", qp, val));
				success = false;
				
			}
			System.out.println("## less extreme example, same phenomenon:");
			double a = 43779, b = 0.06728;
			qp = Beta.quantile(0.695, b, a, true, false);
			val = Beta.cumulative(qp, b, a, true, false);
			if (!isEqual(0.695, val)) {
				System.err.println(String.format("Beta.quantile(0.695, 0.06728, 43779, true, false) = %3.18g", qp));
				System.err.println(String.format("Beta.cumulative(%3.18g, 0.06728, 43779, true, false) = %3.18g != 0.695", qp, val));
				success = false;
			}
			x = vmin(vexp(seq(0, 14, pow(2, -9))));
			double[] qx = new double[x.length], pqx = new double[x.length];
			long time1 = System.currentTimeMillis();
			for (int i = 0; i < x.length; i++)
				qx[i] = Beta.quantile(x[i], a, b, true, true);
			long c1 = System.currentTimeMillis() - time1;
			for (int i = 0; i < x.length; i++)
				pqx[i] = Beta.cumulative(qx[i], a, b, true, true);
			success &= printAllEqualScaled(x, pqx, 2e-15);
			System.out.println("## note that qx[x > -exp(2)] is too close to 1 to get full accuracy:");
			System.out.println("## i2 <- x > -exp(2); all.equal(x[i2], pqx[i2], tol= 0)#-> 5.849e-12");
			System.out.println("System time = " + c1/1000.0);

			System.out.println("## was Inf, and much slower, for R <= 3.1.0");
			double[] x3 = vdiv(colon(-15450.0, -15700), pow(2, 11));
			double[] pq3 = new double[x3.length];
			for (int i = 0; i < x3.length; i++)
				pq3[i] = Beta.cumulative(Beta.quantile(x3[i], a, b, true, true), a, b, true, true);
			double[] vv = vabs(vmin(pq3, x3));
			double vmean = mean(vv), vmax = max(vv);
			if (vmean >= 4e-12 || vmax >= 8e-12) {
				System.err.println(String.format("Mean(pq3 - x3) = %3.18g, Max(pq3 - x3) = %3.18g", vmean, vmax));
				success = false;
			}
		}
		{
			for (int i = 1; i <= 323; i++) {
				double lp = -pow(10, -i),
					qq = Beta.quantile(lp, .2, .03, true, true);
				if (1 - qq >= 1e-15) {
					System.err.println(String.format("1 - Beta.quantile(%3.18g, 0.2, 0.03, true, true) = %3.18g > 1e-15", lp, 1 - qq));
					success = false;
				}
			}
			System.out.println("# warnings in R <= 3.1.0");
			if (!Double.isNaN(val = Beta.quantile(0.5, 2, 3, true, true))) {
				System.err.println(String.format("Beta.quantile(0.5, 2, 3, true, true) = %3.18g != NaN", val));
				success = false;
			}
			if (!Double.isNaN(val = Beta.quantile(-0.1, 2, 3, true, false))) {
				System.err.println(String.format("Beta.quantile(-0.1, 2, 3, true, false) = %3.18g != NaN", val));
				success = false;
			}
			if (!Double.isNaN(val = Beta.quantile(1.25, 2, 3, true, false))) {
				System.err.println(String.format("Beta.quantile(1.25, 2, 3, true, false) = %3.18g != NaN", val));
				success = false;
			}
			System.out.println("# typically qq == 1  exactly");
			System.out.println("## failed in intermediate versions");
			double a = pow(2, -8);
			for (int i = 200; i <= 500; i++) {
				double b = pow(2, i),
					pq = Beta.cumulative(Beta.quantile(1.0/8, a, b, true, false), a, b, true, false);
				if ((val = abs(pq - 1.0/8)) > 1.0/8) {
					System.err.println(String.format("|Beta.cumulative(Beta.quantile(1.0/8, 2^-8, %3.18g, true, false), 2^-8, %3.18g, true, false) -1/8| = %3.18g > 1/8", b, b, val));
					success = false;
				}
			}
			System.out.println("## whereas  qbeta() would underflow to 0 \"too early\", for R <= 3.1.0");
			System.out.println("#");
			System.out.println("## very extreme tails on both sides");
			for (double xx : c(1e-300, 1e-12, 1e-5, 0.1, 0.21, 0.3)) {
				val = Beta.quantile(xx, pow(2, -12), pow(2, -10), true, false);
				if (val != 0) {
					System.err.println(String.format("Beta.quantile(%3.18g, 2^-12, 2^-19, true, false) = %3.18g != 0", xx, val));
					success = false;
				}
			}
			double[] ax = vpow(10, colon(-8, -323));
			for (int i = 0; i < ax.length; i++) {
				val = Beta.quantile(0.95, ax[i], 20, true, false);
				if (MathFunctions.isInfinite(val) || val >= 1e-300) {
					System.err.println(String.format("Beta.quantile(0.95, %3.18g, 20, true, false) = %3.18g >= 1e-300", ax[i], val));
					success = false;
				}
			}

			long time1 = System.currentTimeMillis();
			for (int i = 0; i < ax.length; i++) {
				val = Beta.quantile(0.95, ax[i], ax[i], true, false);
				if (val != 1) {
					System.err.println(String.format("Beta.quantile(0.95, %3.18g, %3.18g, true, false) = %3.18g != 1", ax[i], ax[i], val));
					success = false;
				}
			}
			long ct2 = System.currentTimeMillis() - time1;
			System.out.println("System time = " + (ct2 / 1000.0));
		}
		// TODO Add more test cases here
		//*/
		return success;
	}

	/**
	 * DKW test, taken from p-r-random-tests.R
	 * @return
	 */
	@Test
	public static final boolean test_dkwtest() {
		boolean success = true;
		System.out.println("DKW test");
		System.out.println("Binomial");
		success &= printBool(dkwtest(new Binomial(1, 0.2)));
		success &= printBool(dkwtest(new Binomial(2, 0.2)));
		success &= printBool(dkwtest(new Binomial(100, 0.2)));
		success &= printBool(dkwtest(new Binomial(1e4, 0.2)));
		success &= printBool(dkwtest(new Binomial(1, 0.8)));
		success &= printBool(dkwtest(new Binomial(100, 0.8)));
		success &= printBool(dkwtest(new Binomial(100, 0.999)));

		System.out.println("Poisson");
		success &= printBool(dkwtest(new Poisson(0.095)));
		success &= printBool(dkwtest(new Poisson(0.95)));
		success &= printBool(dkwtest(new Poisson(9.5)));
		success &= printBool(dkwtest(new Poisson(95)));

		System.out.println("Negative Binomial");
		success &= printBool(dkwtest(new NegBinomial(1, 0.2)));
		success &= printBool(dkwtest(new NegBinomial(2, 0.2)));
		success &= printBool(dkwtest(new NegBinomial(100, 0.2)));
		success &= printBool(dkwtest(new NegBinomial(1e4, 0.2)));
		success &= printBool(dkwtest(new NegBinomial(1, 0.8)));
		success &= printBool(dkwtest(new NegBinomial(100, 0.8)));
		success &= printBool(dkwtest(new NegBinomial(100, 0.999)));

		System.out.println("Normal");
		success &= printBool(dkwtest(new Normal()));
		success &= printBool(dkwtest(new Normal(5, 3)));

		System.out.println("Gamma");
		success &= printBool(dkwtest(new Gamma(0.1, 1)));
		success &= printBool(dkwtest(new Gamma(0.2, 1)));
		success &= printBool(dkwtest(new Gamma(10, 1)));
		success &= printBool(dkwtest(new Gamma(20, 1)));

		System.out.println("Hypergeometric");
		success &= printBool(dkwtest(new HyperGeometric(40, 30, 20)));
		success &= printBool(dkwtest(new HyperGeometric(40,  3, 20)));
		success &= printBool(dkwtest(new HyperGeometric( 6,  3,  2)));
		success &= printBool(dkwtest(new HyperGeometric( 5,  3,  2)));
		success &= printBool(dkwtest(new HyperGeometric( 4,  3,  2)));

		System.out.println("SignRank");
		success &= printBool(dkwtest(new SignRank(1)));
		success &= printBool(dkwtest(new SignRank(2)));
		success &= printBool(dkwtest(new SignRank(10)));
		success &= printBool(dkwtest(new SignRank(30)));

		System.out.println("Wilcoxon");
		success &= printBool(dkwtest(new Wilcoxon(40, 30)));
		success &= printBool(dkwtest(new Wilcoxon(40, 10)));
		success &= printBool(dkwtest(new Wilcoxon( 6,  3)));
		success &= printBool(dkwtest(new Wilcoxon( 5,  3)));
		success &= printBool(dkwtest(new Wilcoxon( 4,  3)));

		System.out.println("ChiSquare");
		success &= printBool(dkwtest(new ChiSquare(1)));
		success &= printBool(dkwtest(new ChiSquare(10)));

		System.out.println("Logistic");
		success &= printBool(dkwtest(new Logistic()));
		success &= printBool(dkwtest(new Logistic(4, 2)));

		System.out.println("T");
		success &= printBool(dkwtest(new T(1)));
		success &= printBool(dkwtest(new T(10)));
		success &= printBool(dkwtest(new T(40)));

		System.out.println("Beta");
		success &= printBool(dkwtest(new Beta(1, 1)));
		success &= printBool(dkwtest(new Beta(2, 1)));
		success &= printBool(dkwtest(new Beta(1, 2)));
		success &= printBool(dkwtest(new Beta(2, 2)));
		success &= printBool(dkwtest(new Beta(0.2, 0.2)));

		System.out.println("Cauchy");
		success &= printBool(dkwtest(new Cauchy()));
		success &= printBool(dkwtest(new Cauchy(4, 2)));

		System.out.println("F");
		success &= printBool(dkwtest(new F(1, 1)));
		success &= printBool(dkwtest(new F(1, 10)));
		success &= printBool(dkwtest(new F(10, 10)));
		success &= printBool(dkwtest(new F(30, 3)));

		System.out.println("Weibull");
		success &= printBool(dkwtest(new Weibull(1, 1)));
		success &= printBool(dkwtest(new Weibull(4, 4)));

		System.out.println("## regression test for PR#7314");
		success &= printBool(dkwtest(new HyperGeometric(60, 100, 50)));
		success &= printBool(dkwtest(new HyperGeometric(6, 10, 5)));
		success &= printBool(dkwtest(new HyperGeometric(600, 1000, 500)));

		System.out.println("## regression test for non-central t bug");
		success &= printBool(dkwtest(new NonCentralT(20, 3)));

		System.out.println("## regression test for non-central F bug");
		success &= printBool(dkwtest(new NonCentralF(10, 2, 3)));
		return success;
	}

	@Test
	public static final boolean test_disttest() {
		boolean success = true;
		// Taken from ansari.test.R
		//Hollander & Wolfe (1973, p. 86f):
		//Serum iron determination using Hyland control sera
		double[] ramsay = new double [] {111, 107, 100, 99, 102, 106, 109, 108, 104, 99,
            101, 96, 97, 102, 107, 113, 116, 113, 110, 98};
		double[] jung_parekh = new double[] {107, 108, 106, 98, 105, 103, 110, 105, 104,
	         100, 96, 108, 103, 104, 114, 114, 113, 108, 106, 99};
		System.out.println("Ansari-Bradley Test");
		double[] result = DistributionTest.ansari_bradley_test(ramsay, jung_parekh, true);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(185.5, result[0]) && isEqual(0.18668552840821545, result[1]));
		System.out.println("Mood Test");
		result = DistributionTest.mood_test(ramsay, jung_parekh);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(1.0371275614960966, result[0]) && isEqual(0.2996764118570592, result[1]));
		double[] x = new double [] {1.83,  0.50,  1.62,  2.48, 1.68, 1.88, 1.55, 3.06, 1.30};
		double[] y = new double [] {0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29};
		System.out.println("One-sample Wilcoxon Test");
		result = DistributionTest.wilcoxon_test(x, 0, true, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(45, result[0]) && isEqual(0.00390625, result[1]));
		System.out.println("Paired Mann-Whitney-U Test");
		result = DistributionTest.mann_whitney_u_test(x, y, 0, true, true, TestKind.GREATER);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(45, result[0]) && isEqual(0.001953125, result[1]));
		System.out.println("Two-sample Mann-Whitney-U Test");
		x = new double [] {0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46};
		y = new double [] {1.15, 0.88, 0.90, 0.74, 1.21};
		result = DistributionTest.mann_whitney_u_test(x, y, 0, true, false, TestKind.GREATER);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(35, result[0]) && isEqual(0.1272061272061272008127, result[1]));
		System.out.println("Var Test");
		x = new double[] {-0.005915285439157303565283,-1.995049358415717177806,-1.771051769481722493182,0.5288625402887254800532,2.066390112695046799018,0.7320344587152662896301,1.873492976768343254435,0.2179080945240439992627,2.422614245474107086409,-0.2019968311562227447631,-0.1667974334512828393784,1.352799101715276286484,1.735829567885912139147,4.976315242127600235733,0.817962538806016437043,-0.288765563552636161937,0.1983413319843171929158,-0.9592436326403221968917,1.873291957531057372321,3.439905549378247773262,-1.754855117887118343134,1.648422551214854925306,1.705558268988422110368,1.862012741987735697791,-3.682123201108596699527,-3.023799046634866538596,-2.119212102031017508352,-1.877945465249680312425,0.02142247140925247414489,3.121802213185540963991,-2.644542898146826193084,2.673253669524354947384,2.497415928782492411386,2.825257305101088967092,1.550290797587723812256,-1.421035101213272255904,1.109415261021953869047,1.990423724568782137823,-2.144307394197637606226,2.514795676176211358666,1.738512235679824380341,-0.05401121886275747630002,1.164752171157828586345,-0.551498232065821936132,1.004429216738475405535,-0.4915129833379534574078,4.977208786370868054405,-1.820448359659552650669,-1.556725998088647644479,0.8420680350214259091146};
		y = new double[] {-0.02319414047216561414189,1.428122031169983152665,2.780355586626735764355,0.842387368513194512154,0.6153979643142353239682,1.715529410166719781472,1.770721477752376316062,0.1262469481967164464109,-0.278483750482336533949,1.547831228024979743907,-0.5680164593480081514087,0.5357179604790032190209,0.8482273496325224426684,2.284973376732966965363,0.6152147019912435066402,0.03001237172672799324857,0.8150918187609034193386,2.535170571801844907611,2.232098069877867185795,0.1569715957206057055373,2.510567787654540694575,0.1861565750666677176994,0.6123666074302951400909,2.251588754996833863231,2.482495890568089613737,1.234855993304855514836,1.485982046515017085397,2.592649904509894298599,1.114295729589295680384,0.3105760796473991947053};
		result = DistributionTest.var_test(x, y, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(4.092220946951346860487, result[0]) && isEqual(0.0001238966302667954266781, result[1]));
		System.out.println("Two-sample T Test");
		x = colon(1., 10);
		y = colon(7., 20);
		result = DistributionTest.t_test(x, y, 0, false, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(-5.434929763894059462359, result[0]) && isEqual(1.855281832511811190324e-05, result[1]));
		result = DistributionTest.t_test(x, c(y, new double[]{200}), 0, false, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(-1.632902633201205322422, result[0]) && isEqual(0.124513498089745308639, result[1]));
		x = new double[] {0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0};
		y = new double[] {1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4};
		result = DistributionTest.t_test(x, y, 0, false, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(-1.860813467486853056698, result[0]) && isEqual(0.07939414018735817257788, result[1]));
		System.out.println("Paired T Test");
		result = DistributionTest.t_test_paired(x, y, 0, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(-4.062127683382036558157, result[0]) && isEqual(0.002832890197384270187381, result[1]));
		System.out.println("Two-sample T Test");
		result = DistributionTest.t_test(x, y, 0, true, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(-1.860813467486853056698, result[0]) && isEqual(0.07918671421593817538742, result[1]));
		System.out.println("Bartlett Test");
		x = new double[] {10,7,20,14,14,12,10,23,17,20,14,13,11,17,21,11,16,14,17,17,19,21,7,13,0,1,7,2,3,1,2,1,3,0,1,4,3,5,12,6,4,3,5,5,5,5,2,4,3,5,3,5,3,6,1,1,3,2,6,4,11,9,15,22,15,16,13,10,26,26,24,13};
		int[] g = rep_each(colon(1, 6), 12);
		result = DistributionTest.bartlett_test(x, g);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqualScaled(25.95982532036868661862, result[0]) && isEqualScaled(9.085122332945314439778e-05, result[1]));
		System.out.println("Fligner Test");
		result = DistributionTest.fligner_test(x, g);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(VectorMath.isEqual(14.4827810384586079806, result[0], 1e-12) && VectorMath.isEqual(0.01281677918970919316521, result[1], 1e-12));
		System.out.println("Kruskal-Wallis Test");
		x = new double[] {2.9, 3.0, 2.5, 2.6, 3.2, 3.8, 2.7, 4.0, 2.4, 2.8, 3.4, 3.7, 2.2, 2.0};
		g = new int[] {1,1,1,1,1,2,2,2,2,3,3,3,3,3};
		result = DistributionTest.kruskal_wallis_test(x, g);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(0.7714285714285722406203, result[0]) && isEqual(0.6799647735788936220303, result[1]));
		System.out.println("Poisson Test");
		result = DistributionTest.poisson_test(137, 24.19893, 1, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(137, result[0]) && isEqual(2.845227264114483019947e-56, result[1]));
		result = DistributionTest.poisson_test(11, 6+8+7, 800, 1083+1050+878, 1, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(11, result[0]) && isEqual(0.07966863303332952228608, result[1]));
		System.out.println("Binomial Test");
		result = DistributionTest.binomial_test(682, 682+243, 0.75, TestKind.TWO_SIDED);
		System.out.println(result[0]);
		System.out.println(result[1]);
		success &= printBool(isEqual(682, result[0]) && isEqual(0.3824915595748519248076, result[1]));
		return success;
	}

	static final void test_diptest() {
		double[] x, statfaculty =
		  new double[] {30,33,35,36,37,37,39,39,39,39,39,40,40,40,40,41,42,43,43,43,44,44,45,45,46,
		    46,47,47,48,48,48,49,50,50,51,52,52,53,53,53,53,53,54,54,57,57,59,60,60,60,
		    61,61,61,61,62,62,62,62,63,66,70,72,72};
		x = DistributionTest.diptest(statfaculty);
		System.out.println(x[0]); // == 0.05952381;
		System.out.println(x[1]); // == 0.08672336;
		x = DistributionTest.diptest(c(1.0,1,2,2));
		System.out.println(x[0]); // == 0.25;
		x = DistributionTest.diptest(c(0.0,2,3,5,6));
		System.out.println(x[0]); // == 0.133333;
		x = DistributionTest.diptest(c(6.0,4,3,1,0));
		System.out.println(x[0]); // == 0.133333;
		x = new double[] {3.6, 1.8, 3.333, 2.283, 4.533, 2.883, 4.7, 3.6, 1.95, 4.35, 1.833, 3.917, 4.2, 1.75, 4.7, 2.167, 1.75, 4.8, 1.6, 4.25, 1.8, 1.75, 3.45, 3.067, 4.533, 3.6, 1.967, 4.083, 3.85, 4.433, 4.3, 4.467, 3.367, 4.033, 3.833, 2.017, 1.867, 4.833, 1.833, 4.783, 4.35, 1.883, 4.567, 1.75, 4.533, 3.317, 3.833, 2.1, 4.633, 2, 4.8, 4.716, 1.833, 4.833, 1.733, 4.883, 3.717, 1.667, 4.567, 4.317, 2.233, 4.5, 1.75, 4.8, 1.817, 4.4, 4.167, 4.7, 2.067, 4.7, 4.033, 1.967, 4.5, 4, 1.983, 5.067, 2.017, 4.567, 3.883, 3.6, 4.133, 4.333, 4.1, 2.633, 4.067, 4.933, 3.95, 4.517, 2.167, 4, 2.2, 4.333, 1.867, 4.817, 1.833, 4.3, 4.667, 3.75, 1.867, 4.9, 2.483, 4.367, 2.1, 4.5, 4.05, 1.867, 4.7, 1.783, 4.85, 3.683, 4.733, 2.3, 4.9, 4.417, 1.7, 4.633, 2.317, 4.6, 1.817, 4.417, 2.617, 4.067, 4.25, 1.967, 4.6, 3.767, 1.917, 4.5, 2.267, 4.65, 1.867, 4.167, 2.8, 4.333, 1.833, 4.383, 1.883, 4.933, 2.033, 3.733, 4.233, 2.233, 4.533, 4.817, 4.333, 1.983, 4.633, 2.017, 5.1, 1.8, 5.033, 4, 2.4, 4.6, 3.567, 4, 4.5, 4.083, 1.8, 3.967, 2.2, 4.15, 2, 3.833, 3.5, 4.583, 2.367, 5, 1.933, 4.617, 1.917, 2.083, 4.583, 3.333, 4.167, 4.333, 4.5, 2.417, 4, 4.167, 1.883, 4.583, 4.25, 3.767, 2.033, 4.433, 4.083, 1.833, 4.417, 2.183, 4.8, 1.833, 4.8, 4.1, 3.966, 4.233, 3.5, 4.366, 2.25, 4.667, 2.1, 4.35, 4.133, 1.867, 4.6, 1.783, 4.367, 3.85, 1.933, 4.5, 2.383, 4.7, 1.867, 3.833, 3.417, 4.233, 2.4, 4.8, 2, 4.15, 1.867, 4.267, 1.75, 4.483, 4, 4.117, 4.083, 4.267, 3.917, 4.55, 4.083, 2.417, 4.183, 2.217, 4.45, 1.883, 1.85, 4.283, 3.95, 2.333, 4.15, 2.35, 4.933, 2.9, 4.583, 3.833, 2.083, 4.367, 2.133, 4.35, 2.2, 4.45, 3.567, 4.5, 4.15, 3.817, 3.917, 4.45, 2, 4.283, 4.767, 4.533, 1.85, 4.25, 1.983, 2.25, 4.75, 4.117, 2.15, 4.417, 1.817, 4.467};
		x = DistributionTest.diptest(x);
		System.out.println(x[0]); // == 0.09238103;
		System.out.println(x[1]); // == 0;
		x = new double[] {67, 54.7, 7, 48.5, 14, 17.2, 20.7, 13, 43.4, 40.2, 38.9, 54.5, 59.8, 48.3, 22.9, 11.5, 34.4, 35.1, 38.7, 30.8, 30.6, 43.1, 56.8, 40.8, 41.8, 42.5, 31, 31.7, 30.2, 25.9, 49.2, 37, 35.9, 15, 30.2, 7.2, 36.2, 45.5, 7.8, 33.4, 36.1, 40.2, 42.7, 42.5, 16.2, 39, 35, 37, 31.4, 37.6, 39.9, 36.2, 42.8, 46.4, 24.7, 49.1, 46, 35.9, 7.8, 48.2, 15.2, 32.5, 44.7, 42.6, 38.8, 17.4, 40.8, 29.1, 14.6, 59.2};
		x = DistributionTest.diptest(x);
		System.out.println(x[0]); // == 0.03571429;
		System.out.println(x[1]); // == 0.7724761;
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

	static final void kstest_example2() {
		int n = 50000;
		double[] x = new double[n], y = new double[n];
		long time1, time2;
		for (int i = 0; i < n; i++) {
			x[i] = random.nextDouble();
			y[i] = random.nextDouble();
		}
		time1 = System.currentTimeMillis();
		System.out.println(kolmogorov_smirnov_test(x, y)[1]);
		time2 = System.currentTimeMillis();
		System.out.println("Time (Exact) = " + (time2 - time1));
		time1 = System.currentTimeMillis();
		System.out.println(kolmogorov_smirnov_test(x, y, false)[1]);
		time2 = System.currentTimeMillis();
		System.out.println("Time (Inexact) = " + (time2 - time1));
	}

	static final void kstest_example3() {
		double[] x = {-0.4327256096589832,0.8556666526274761,-0.5546375278179974,1.866000905204401,-1.033452361074729,-1.5128029893029131,-0.8133759541734086,0.687640682247052,1.5419273795363184,-2.2295925458536794,1.42558868046075,-2.3797684265098655,-0.46027673367988814,0.4297141073628944,1.0848742490404304,1.9887805457720291,0.5271399436789418,2.1457932121313377,0.17732741027681803,0.5901870929879354,-0.7486019696536792,-1.4427382726841995,0.9137934132833048,0.5241007177131126,0.5613852341528923,-1.3266402599057574,-1.323280288237919,0.5814876355837048,0.6308838159891056,1.3385488233125724,-0.06834424563535381,-1.1671401628369862,0.09379937639386121,0.17521277521223777,-1.3168087117893892,-0.5244543795040156,0.24032301793814181,0.5426753659697515,0.22182011780487396,0.6737536223042431,1.3402529813509523,-1.6817165954775795,-0.20353732094995078,-0.7690838029162941,0.7984057628094654,0.3943074921556351,0.8177375877054202,-0.9211829231536616,0.7630904005539952,0.11572488935311727,0.39220521737386826,-1.8541504624965093},
		y = {-1.4429961932875182,-0.7818279436142089,-1.1966511647322804,10.171962424735371,11.327019721485414,11.99503521664728,10.952447325187496,8.929430883277462};
		double[] p = kolmogorov_smirnov_test(x, y);
		System.out.println(p[0]);
		System.out.println(p[1]);
		x = c(-1.8008263,-0.3398806,0.6062646,1.3411303);
		y = c(0.7672873,2.1937257,3.1405667,2.0138648,0.8946941);
		p = kolmogorov_smirnov_test(x, y);
		System.out.println(p[0]);
		System.out.println(p[1]);
	}

	public static final void norm_test() {
		double[] x = {4281.099776,4376.951826,4378.025799,4613.572586,4666.54245,4675.439476,
			4709.961628,4709.999973,4709.999973,4709.999973,4743.40979,4751.232853,4756.295879,
			4756.869617,4779.386288,4801.67123,4802.705246,4803.434021,4810.069633,4821.232853,
			4824.875099,4825.106827,4825.183517,4829.172607,4832.31778,4834.73423,4845.282188,
			4845.282188,4845.282188,4850.573738,4851.417549,4856.05867,4859.320534,4859.587456,
			4859.587456,4859.742439,4865.570984,4867.452042,4869.830147,4870.403885,4870.403885,
			4870.403885,4870.904083,4871.707947,4871.939728,4872.398376,4873.664146,4879.227371,
			4879.801109,4884.980827,4893.839508,4895.181915,4897.214859,4897.29155,4897.521675,
			4897.521675,4897.521675,4897.751801,4897.751801,4901.625748,4908.186325,4908.186325,
			4913.671204,4920.535393,4929.05201,4932.54245,4935.841057,4937.375298,4941.401131,
			4941.709602,4942.591759,4944.086052,4945.735382,4949.762764,4950.261414,4950.453194,
			4950.531487,4952.142418,4952.257507,4953.638313,4953.86689,4954.672356,4954.672356,
			4955.439476,4958.891571,4959.046555,4959.545204,4959.545204,4961.386314,4961.386314,
			4961.386314,4962.612083,4965.105225,4966.179214,4966.180817,4966.21756,4966.255905,
			4966.255905,4966.831245,4967.253204,4967.86689,4968.136963,4968.136963,4972.124451,
			4972.124451,4973.275078,4973.46846,4974.349068,4976.30687,4978.493141,4980.02578,
			4980.02578,4980.524429,4982.136963,4982.904137,4984.36158,4988.619141,4989.654785,
			4991.916199,4991.994492,4992.684868,4994.027382,4994.257507,4995.713455,4996.135361,
			4996.749046,4998.783539,5000.31778,5000.623047,5000.623047,5000.624649,5003.769821,
			5005.302513,5005.916199,5005.916199,5007.643768,5007.643768,5007.643768,5009.138115,
			5010.558868,5010.558868,5010.558868,5010.673958,5010.940826,5011.019119,5011.132607,
			5011.477821,5012.630104,5012.630104,5012.860283,5012.860283,5012.936974,5013.088753,
			5014.624649,5015.545151,5015.545151,5015.736931,5015.736931,5016.310722,5016.310722,
			5016.694229,5017.461403,5017.461403,5017.461403,5017.461403,5017.653183,5017.959999,
			5018.036743,5018.036743,5018.573738,5019.340858,5020.06958,5020.06958,5020.838303,
			5020.838303,5021.835602,5023.483276,5029.006607,5029.85202,5029.85202,5029.85202,
			5029.85202,5029.888763,5030.657486,5031.194534,5031.614784,5031.614784,5033.457497,
			5033.457497,5034.95179,5034.95179,5034.95179,5034.991791,5035.450439,5035.450439,
			5036.334251,5038.020271,5038.712303,5038.942429,5041.012062,5041.473969,5042.239487,
			5042.239487,5042.239487,5043.812073,5043.812073,5043.812073,5043.812073,5044.694283,
			5045.0411,5045.079445,5045.307968,5045.307968,5045.307968,5045.307968,5045.614838,
			5047.342407,5049.373695,5051.253151,5051.599968,5052.020325,5052.020325,5052.020325,
			5052.367088,5052.82579,5056.354523,5061.762817,5062.261414,5062.76001,5062.76001,
			5063.527184,5063.643822,5063.643822,5064.179214,5064.946281,5065.86689,5066.175362,
			5066.175362,5066.442177,5066.557266,5066.750648,5067.01757,5067.554565,5068.244942,
			5068.398376,5068.475121,5069.587402,5069.587402,5070.239487,5071.006607,5071.008156,
			5072.23558,5072.23558,5073.691528,5073.691528,5074.228523,5075.879402,5076.069633,
			5076.069633,5076.146324,5076.146324,5076.569832,5077.030083,5077.068481,5077.068481,
			5078.98468,5079.866837,5080.635559,5080.827339,5081.401131,5081.784637,5082.054764,
			5082.858627,5084.317833,5084.317833,5084.546303,5084.89312,5085.198387,5085.85202,
			5088.688828,5089.647675,5090.146271,5092.257507,5093.444931,5093.636765,5095.631256,
			5095.823036,5095.976471,5095.976471,5098.241089,5099.044952,5099.735329,5100.387413,
			5100.887611,5103.647728,5104.223015,5104.76001,5105.797203,5105.797203,5106.29425,
			5106.29425,5106.90799,5107.138115,5107.523224,5107.599968,5108.788994,5108.827339,
			5109.249245,5109.287643,5109.326042,5109.823036,5110.244995,5111.050407,5112.009361,
			5112.009361,5114.655884,5116.305214,5116.460251,5116.460251,5119.490387,5119.834,
			5121.675056,5122.403885,5123.824638,5123.824638,5124.129906,5126.661392,5126.853172,
			5128.157288,5129.499748,5129.769875,5130.997192,5131.26577,5132.991791,5133.066879,
			5134.104073,5135.099716,5137.784691,5137.938072,5137.938072,5137.938072,5137.938072,
			5137.938072,5137.938072,5138.515068,5139.242188,5139.855873,5140.124451,5142.15889,
			5143.0411,5145.687622,5145.687622,5145.687622,5146.683266,5146.798409,5146.915047,
			5147.834,5147.949036,5147.949036,5147.949036,5149.216408,5149.216408,5150.328743,
			5150.749046,5150.749046,5152.591705,5152.668503,5154.086052,5156.120544,5157.001152,
			5157.998344,5158.573738,5158.688828,5159.879456,5159.879456,5160.262962,5160.991791,
			5160.991791,5160.991791,5161.565475,5161.603821,5161.603821,5162.716156,5162.716156,
			5162.946335,5162.946335,5162.946335,5165.247696,5165.247696,5167.932617,5168.201088,
			5168.201088,5169.046555,5169.696983,5170.003853,5170.003853,5170.003853,5170.003853,
			5170.003853,5170.003853,5170.005455,5170.504051,5170.694283,5171.192879,5171.192879,
			5171.923256,5172.191727,5172.191727,5173.419151,5173.419151,5173.419151,5173.419151,
			5174.032837,5174.186272,5174.186272,5174.378105,5174.531487,5175.297005,5175.490387,
			5175.490387,5178.213707,5178.213707,5178.865738,5181.779236,5181.779236,5182.853172,
			5184.389015,5185.269623,5185.423058,5186.228523,5187.570984,5187.801109,5189.297005,
			5191.101372,5191.521622,5192.940826,5193.938126,5193.938126,5194.13311,5194.591759,
			5194.591759,5194.591759,5196.277832,5196.891571,5199.65313,5199.69313,5199.69313,
			5200.230125,5200.230125,5200.688774,5202.798355,5205.906784,5206.212051,5207.711151,
			5207.711151,5207.711151,5208.206596,5208.206596,5208.246544,5209.398827,5209.437172,
			5211.66024,5212.772575,5218.565979,5218.677864,5219.09977,5220.827339,5221.442627,
			5221.442627,5221.631203,5223.130302,5226.505653,5230.109581,5231.450386,5232.717812,
			5235.516167,5236.706848,5236.706848,5236.706848,5237.127151,5238.124397,5242.997246,
			5242.997246,5244.301361,5244.45314,5244.45314,5244.953392,5247.40979,5247.830147,
			5248.020271,5250.209801,5250.401581,5251.207047,5251.817581,5252.434471,5253.736931,
			5255.961655,5256,5256.039948,5256.039948,5257.379204,5261.02623,5262.097015,
			5262.44223,5265.322136,5265.437225,5266.086052,5266.394524,5267.505203,5267.505203,
			5267.505203,5268.772575,5269.041046,5269.194481,5270.421906,5270.421906,5270.421906,
			5271.149078,5274.986282,5274.986282,5274.986282,5274.986282,5280.354523,5281.736877,
			5283.349518,5285.495895,5285.495895,5287.453644,5287.798859,5288.911194,5290.138565,
			5290.982376,5291.059174,5292.706848,5293.857529,5294.31778,5295.506805,5295.506805,
			5296.005402,5296.504105,5297.61644,5297.961548,5299.535843,5302.142471,5302.485977,
			5302.485977,5302.792953,5306.094711,5306.668396,5307.242188,5308.125946,5308.70134,
			5308.893173,5308.971466,5308.971466,5311.501297,5311.808167,5312.345215,5313.073883,
			5313.380859,5313.380859,5315.490387,5316.410889,5316.449234,5316.679413,5318.597214,
			5320.975266,5320.975266,5321.35733,5322.816376,5323.391769,5325.462952,5325.462952,
			5326.690323,5327.919403,5328.301361,5329.490387,5329.605423,5330.027435,5330.027435,
			5331.214859,5332.979172,5333.861328,5334.286545,5335.360428,5335.704041,5338.007111,
			5338.007111,5338.695831,5339.347916,5339.883255,5341.187424,5341.417603,5341.417603,
			5341.417603,5344.564377,5345.064575,5347.286041,5348.476669,5350.546356,5351.314972,
			5351.928818,5352.007111,5353.769821,5353.769821,5354.345215,5356.071182,5359.908386,
			5362.821884,5363.35733,5366.080597,5366.809372,5370.339706,5373.444931,5373.444931,
			5373.444931,5374.828995,5375.17421,5375.554459,5379.008209,5379.0849,5384.913391,
			5384.913391,5384.953445,5385.14357,5385.22197,5385.29866,5385.491989,5385.79715,
			5391.012115,5391.013718,5393.813675,5393.813675,5394.427307,5395.311172,5397.495789,
			5398.683319,5398.723267,5400.219162,5400.219162,5400.3358,5402.290344,5402.290344,
			5403.822983,5406.584595,5409.731476,5410.345215,5412.14798,5415.406586,5416.098511,
			5427.450333,5428.18071,5436.464157,5436.810974,5436.964355,5439.687622,5439.687622,
			5439.687622,5439.687622,5439.687622,5439.687622,5443.636658,5450.425812,5450.425812,
			5454.109528,5454.146378,5454.14798,5454.14798,5455.52713,5455.835602,5459.401184,
			5464.887665,5464.887665,5464.887665,5468.606522,5468.913391,5469.145172,5470.794556,
			5471.369736,5483.528732,5486.060165,5499.139664,5500.597214,5502.361633,5503.318878,
			5518.469559,5526.142471,5526.142471,5528.212051,5528.212051,5528.250397,5536.038345,
			5536.038345,5536.038345,5541.024628,5541.713348,5579.724365};
		double w = NormalityTest.shapiro_wilk_statistic(x);
		double p = NormalityTest.shapiro_wilk_pvalue(w, x.length);
		System.out.println(w + " -- " + p);
		double[] pp = NormalityTest.kolmogorov_smirnov_test(x);
		System.out.println(pp[0] + " -- " + pp[1]);
		w = NormalityTest.cramer_vonmises_statistic(x);
		p = NormalityTest.cramer_vonmises_pvalue(w, x.length);
		System.out.println(w + " -- " + p);
	}

	public static final void main(String[] args) {
		//System.out.println(String.format("%3.18g", MathFunctions.gammafn(13.51)));
		//System.out.println(NonCentralChiSquare.cumulative(1e-5, 100, 1, true, false));
//		test_binom();
//		test_geom();
//		test_hyper();
//		test_negbin();
//		test_poisson();
//		test_signrank();
//		test_wilcox();
//		test_gamma();
//		test_noncentralchisq();
//		test_beta();
//		test_normal();
//		test_random();
		test_extreme();
//		test_dkwtest();
//		test_disttest();
//		norm_test();
		System.exit(0);
	}
}
