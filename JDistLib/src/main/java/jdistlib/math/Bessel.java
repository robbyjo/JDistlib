/*
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998   Ross Ihaka
 *  Copyright (C) 2000-9 The R Development Core Team
 *
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
package jdistlib.math;

import static java.lang.Math.*;
import static jdistlib.math.Constants.*;
import static jdistlib.math.MathFunctions.cospi;
import static jdistlib.math.MathFunctions.gamma_cody;
import static jdistlib.math.MathFunctions.ldexp;
import static jdistlib.math.MathFunctions.sinpi;
import static jdistlib.math.MathFunctions.trunc;

import java.util.Arrays;

/**
 * Collection of Bessel functions.
 * <ul>
 * <li>j is Bessel function of the first kind.</li>
 * <li>y is Bessel function of the second kind.</li>
 * <li>i is modified Bessel function of the first kind.</li>
 * <li>k is modified Bessel function of the third kind.</li>
 * </ul>
 *
 */
public class Bessel {
	/* *******************************************************************

	 Explanation of machine-dependent constants

	   beta	  = Radix for the floating-point system
	   minexp = Smallest representable power of beta
	   maxexp = Smallest power of beta that overflows
	   it = p = Number of bits (base-beta digits) in the mantissa
		    (significand) of a working precision (floating-point) variable
	   NSIG	  = Decimal significance desired.  Should be set to
		    INT(LOG10(2)*it+1).	 Setting NSIG lower will result
		    in decreased accuracy while setting NSIG higher will
		    increase CPU time without increasing accuracy.  The
		    truncation error is limited to a relative error of
		    T=.5*10^(-NSIG).
	   ENTEN  = 10 ^ K, where K is the largest int such that
		    ENTEN is machine-representable in working precision
	   ENSIG  = 10 ^ NSIG
	   RTNSIG = 10 ^ (-K) for the smallest int K such that K >= NSIG/4
	   ENMTEN = Smallest ABS(X) such that X/4 does not underflow
	   XINF	  = Largest positive machine number; approximately beta ^ maxexp
		    == DBL_MAX (defined in  #include <float.h>)
	   SQXMIN = Square root of beta ^ minexp = sqrt(DBL_MIN)

	   EPS	  = The smallest positive floating-point number such that 1.0+EPS > 1.0
		  = beta ^ (-p)	 == DBL_EPSILON


	  For I :

	   EXPARG = Largest working precision argument that the library
		    EXP routine can handle and upper limit on the
		    magnitude of X when IZE=1; approximately LOG(beta ^ maxexp)

	  For I and J :

	   xlrg_IJ = xlrg_BESS_IJ (was = XLARGE). Upper limit on the magnitude of X
		    (when IZE=2 for I()).  Bear in mind that if floor(abs(x)) =: N, then
		    at least N iterations of the backward recursion will be executed.
		    The value of 10 ^ 4 was used till Feb.2009, when it was increased
		    to 10 ^ 5 (= 1e5).

	  For j :
	   XMIN_J  = Smallest acceptable argument for RBESY; approximately
		    max(2*beta ^ minexp, 2/XINF), rounded up

	  For Y :

	   xlrg_Y =  (was = XLARGE). Upper bound on X;
		    approximately 1/DEL, because the sine and cosine functions
		    have lost about half of their precision at that point.

	   EPS_SINC = Machine number below which sin(x)/x = 1; approximately SQRT(EPS).
	   THRESH = Lower bound for use of the asymptotic form;
		    approximately AINT(-LOG10(EPS/2.0))+1.0


	  For K :

	   xmax_k =  (was = XMAX). Upper limit on the magnitude of X when ize = 1;
		    i.e. maximal x for UNscaled answer.

		    Solution to equation:
		       W(X) * (1 -1/8 X + 9/128 X^2) = beta ^ minexp
		    where  W(X) = EXP(-X)*SQRT(PI/2X)

	 --------------------------------------------------------------------

	     Approximate values for some important machines are:

			  beta minexp maxexp it NSIG ENTEN ENSIG RTNSIG ENMTEN	 EXPARG
	 IEEE (IBM/XT,
	   SUN, etc.) (S.P.)  2	  -126	128  24	  8  1e38   1e8	  1e-2	4.70e-38     88
	 IEEE	(...) (D.P.)  2	 -1022 1024  53	 16  1e308  1e16  1e-4	8.90e-308   709
	 CRAY-1	      (S.P.)  2	 -8193 8191  48	 15  1e2465 1e15  1e-4	1.84e-2466 5677
	 Cyber 180/855
	   under NOS  (S.P.)  2	  -975 1070  48	 15  1e322  1e15  1e-4	1.25e-293   741
	 IBM 3033     (D.P.) 16	   -65	 63  14	  5  1e75   1e5	  1e-2	2.16e-78    174
	 VAX	      (S.P.)  2	  -128	127  24	  8  1e38   1e8	  1e-2	1.17e-38     88
	 VAX D-Format (D.P.)  2	  -128	127  56	 17  1e38   1e17  1e-5	1.17e-38     88
	 VAX G-Format (D.P.)  2	 -1024 1023  53	 16  1e307  1e16  1e-4	2.22e-308   709


	And routine specific :

			    xlrg_IJ xlrg_Y xmax_k EPS_SINC XMIN_J    XINF   THRESH
	 IEEE (IBM/XT,
	   SUN, etc.) (S.P.)	1e4  1e4   85.337  1e-4	 2.36e-38   3.40e38	8.
	 IEEE	(...) (D.P.)	1e4  1e8  705.342  1e-8	 4.46e-308  1.79e308   16.
	 CRAY-1	      (S.P.)	1e4  2e7 5674.858  5e-8	 3.67e-2466 5.45e2465  15.
	 Cyber 180/855
	   under NOS  (S.P.)	1e4  2e7  672.788  5e-8	 6.28e-294  1.26e322   15.
	 IBM 3033     (D.P.)	1e4  1e8  177.852  1e-8	 2.77e-76   7.23e75    17.
	 VAX	      (S.P.)	1e4  1e4   86.715  1e-4	 1.18e-38   1.70e38	8.
	 VAX e-Format (D.P.)	1e4  1e9   86.715  1e-9	 1.18e-38   1.70e38    17.
	 VAX G-Format (D.P.)	1e4  1e8  706.728  1e-8	 2.23e-308  8.98e307   16.

	*/
	private static final double
		nsig_BESS = 16,
		ensig_BESS = 1e16,
		rtnsig_BESS = 1e-4,
		enmten_BESS = 8.9e-308,
		enten_BESS = 1e308,
		exparg_BESS = 709.,
		xlrg_BESS_IJ = 1e5,
		xlrg_BESS_Y = 1e8,
		thresh_BESS_Y = 16.,
		xmax_BESS_K = 705.342, // maximal x for UNscaled answer
		sqxmin_BESS_K = 1.49e-154, // sqrt(DBL_MIN) =	1.491668e-154
		/*
		 * x < eps_sinc	 <==>  sin(x)/x == 1 (particularly "==>");
		 * Linux (around 2001-02) gives 2.14946906753213e-08
		 * Solaris 2.5.1		 gives 2.14911933289084e-08
		 */
		M_eps_sinc = 2.149e-8;

	/**
	 * <p>Calculates Bessel functions J_{alpha} (x) for non-negative argument x, and order alpha.
	 * 
	 * <p>Acknowledgement
	 * 
	 * <p>This program is based on a program written by David J. Sookne (2) that computes values of the Bessel functions J or I of float
	 * argument and long order.  Modifications include the restriction of the computation to the J Bessel function of non-negative float
	 * argument, the extension of the computation to arbitrary positive order, and the elimination of most underflow.
	 * 
	 * <p>References:<ol>
	 * <li>Olver, F.W.J., and Sookne, D.J. (1972) "A Note on Backward Recurrence Algorithms"; Math. Comp. 26, 941-947.</li>
	 * <li>Sookne, D.J. (1973) "Bessel Functions of Real Argument and Integer Order"; NBS Jour. of Res. B. 77B, 125-132.</li>
	 * </ol>
	 * 
	 * <P>Latest modification: March 19, 1990
	 * 
	 * <P>@author W. J. Cody
	 * <P>Applied Mathematics Division<br>
	 * Argonne National Laboratory<br>
	 * Argonne, IL  60439
	 * 
	 * @param x Non-negative argument for which J's are to be calculated.
	 * @param alpha Order for which J's are to be calculated.
	 */
	public static final double j(double x, double alpha) {
		if (Double.isNaN(x) || Double.isNaN(alpha)) return x + alpha;
		if (x < 0) return Double.NaN;
		int na = (int) floor(alpha);

		if (na < 0) {
			/* Using Abramowitz & Stegun  9.1.2
			 * this may not be quite optimal (CPU and accuracy wise) */
			return (alpha - na == 0.5) ? 0 : j(x, -alpha) * cospi(alpha) + (alpha == na ? 0 :
				y(x, -alpha) * sinpi(alpha)); // PR#15554 fix
		}
		int nb = 1 + na;
		alpha -= (nb - 1);
		double[] by = new double[nb];
		int ncalc = j_internal(x, alpha, by);
		if (ncalc != nb) {
			if (ncalc < 0)
				System.err.println(String.format("bessel_j(%g): ncalc (=%ld) != nb (=%ld); alpha=%g. Arg. out of range?", x, ncalc, nb, alpha));
			else
				System.err.println(String.format("bessel_j(%g,nu=%g): precision lost in result", x, alpha+(double) nb-1));
		}
		return by[nb-1];
	}

	/**
	 * <p>This routine calculates Bessel functions Y_{alpha} (x) for non-negative argument X, and order alpha.
	 * 
	 * <P>Acknowledgement
	 * 
	 * <P>This program draws heavily on Temme's Algol program for Y(a,x) and Y(a+1,x) and on Campbell's programs for Y_nu(x).
	 * Temme's scheme is used for  x < THRESH, and Campbell's scheme is used in the asymptotic region.  Segments of code from
	 * both sources have been translated into Fortran 77, merged, and heavily modified. Modifications include parameterization
	 * of machine dependencies, use of a new approximation for ln(gamma(x)), and built-in protection against over/underflow.
	 * 
	 * <P>References:<ol>
	 * <li>"Bessel functions J_nu(x) and Y_nu(x) of float order and float argument," Campbell, J. B., Comp. Phy. Comm. 18, 1979, pp. 133-142.</li>
	 * <li>"On the numerical evaluation of the ordinary Bessel function of the second kind," Temme, N. M., J. Comput. Phys. 21, 1976, pp. 343-350.</li>
	 * </ol>
	 * 
	 * <P>Latest modification: March 19, 1990
	 * 
	 * <P>@author Modified by: W. J. Cody
	 * <P>Applied Mathematics Division<br>
	 * Argonne National Laboratory<br>
	 * Argonne, IL  60439
	 * 
	 * @param x Non-negative argument for which Y's are to be calculated.
	 * @param alpha Order for which Y's are to be calculated.
	 */
	public static final double y(double x, double alpha) {
		if (Double.isNaN(x) || Double.isNaN(alpha)) return x + alpha;
		if (x < 0) return Double.NaN;
		int na = (int) floor(alpha);

		if (na < 0) {
			/* Using Abramowitz & Stegun  9.1.2
			 * this may not be quite optimal (CPU and accuracy wise) */
			return (alpha - na == 0.5) ? 0 : y(x, -alpha) * cospi(alpha) + (alpha == na ? 0 :
				j(x, -alpha) * sinpi(alpha)); // PR#15554 fix
		}
		int nb = 1 + na;
		alpha -= (nb - 1);
		double[] by = new double[nb];
		int ncalc = y_internal(x, alpha, by);
		if (ncalc != nb) {
			if (ncalc == -1)
				return Double.POSITIVE_INFINITY;
			if (ncalc < -1)
				System.err.println(String.format("bessel_y(%g): ncalc (=%ld) != nb (=%ld); alpha=%g. Arg. out of range?", x, ncalc, nb, alpha));
			else
				System.err.println(String.format("bessel_y(%g,nu=%g): precision lost in result", x, alpha+(double) nb-1));
		}
		return by[nb - 1];
	}

	/**
	 * <p>This routine calculates Bessel functions I_{alpha} (x) for non-negative argument x,
	 * and order alpha, with or without exponential scaling.
	 * 
	 * <P>Acknowledgement
	 * <P>This program is based on a program written by David J. Sookne (2) that computes values of the Bessel functions J or
	 * I of float argument and long order.  Modifications include the restriction of the computation to the I Bessel function
	 * of non-negative float argument, the extension of the computation to arbitrary positive order, the inclusion of optional
	 * exponential scaling, and the elimination of most underflow. An earlier version was published in (3).
	 * 
	 * <P>References:<ol>
	 * <li>"A Note on Backward Recurrence Algorithms," Olver, F. W. J., and Sookne, D. J., Math. Comp. 26, 1972, pp 941-947.</li>
	 * <li>"Bessel Functions of Real Argument and Integer Order," Sookne, D. J., NBS Jour. of Res. B. 77B, 1973, pp 125-132.</li>
	 * <li>"ALGORITHM 597, Sequence of Modified Bessel Functions of the First Kind," Cody, W. J., Trans. Math. Soft., 1983, pp. 242-245.</li>
	 * </ol>
	 * 
	 * <P>Latest modification: May 30, 1989
	 * 
	 * <P>@author Modified by: W. J. Cody and L. Stoltz<br>
	 * Applied Mathematics Division<br>
	 * Argonne National Laboratory<br>
	 * Argonne, IL  60439
	 * 
	 * @param x Non-negative argument for which I's or exponentially scaled I's (I*EXP(-x)) are to be calculated.
	 * If I's are to be calculated x must be less than exparg_BESS (=709, when expo == FALSE) or xlrg_BESS_IJ (=1e5, when expo == TRUE)
	 * @param alpha - Order for which I's or exponentially scaled I's (I*EXP(-x)) are to be calculated.
	 * @param expo - set true if exponentially scaled I's are to be calculated. Else, if unscaled I's are to be calculated.
	 */
	public static final double i(double x, double alpha, boolean expo) {
		if (Double.isNaN(x) || Double.isNaN(alpha)) return x + alpha;
		if (x < 0) return Double.NaN;
		int na = (int) floor(alpha);

		if (alpha < 0) {
			/* Using Abramowitz & Stegun  9.6.2 & 9.6.6
			 * this may not be quite optimal (CPU and accuracy wise) */
			return i(x, -alpha, expo) + (alpha == na ? 0 :
				k(x, -alpha, expo) * (!expo? 2. : 2.*exp(-2.*x))/PI * sinpi(-alpha));
		}
		int nb = 1 + na;
		alpha -= (nb - 1);
		double[] bi = new double[nb];
		int ncalc = i_internal(x, alpha, expo, bi);
		if (ncalc != nb) {
			if (ncalc < 0)
				System.err.println(String.format("bessel_i(%g): ncalc (=%ld) != nb (=%ld); alpha=%g. Arg. out of range?", x, ncalc, nb, alpha));
			else
				System.err.println(String.format("bessel_i(%g,nu=%g): precision lost in result", x, alpha+(double) nb-1));
		}
		return bi[nb - 1];
	}

	/**
	 * <p>This routine calculates modified Bessel functions of the third kind, K_{alpha} (x), for non-negative argument x,
	 * and order alpha, with or without exponential scaling.
	 * 
	 * <P>Acknowledgement
	 * <P>This program is based on a program written by J. B. Campbell (2) that computes values of the Bessel functions K
	 * of float argument and float order.  Modifications include the addition of non-scaled functions, parameterization
	 * of machine dependencies, and the use of more accurate approximations for SINH and SIN.
	 * 
	 * <P>References:<ol>
	 * <li>"On Temme's Algorithm for the Modified Bessel Functions of the Third Kind," Campbell, J. B., TOMS 6(4), Dec. 1980, pp. 581-586.</li>
	 * <li>"A FORTRAN IV Subroutine for the Modified Bessel Functions of the Third Kind of Real Order and Real Argument," Campbell, J. B., Report NRC/ERB-925, National Research Council, Canada.</li>
	 * </ol>
	 * 
	 * <P>Latest modification: May 30, 1989
	 * 
	 * <p>@author Modified by: W. J. Cody and L. Stoltz
	 * Applied Mathematics Division<br>
	 * Argonne National Laboratory<br>
	 * Argonne, IL  60439
	 * 
	 * @param x Non-negative argument for which K's or exponentially scaled K's (K*EXP(x)) are to be calculated.
	 * If K's are to be calculated, X must not be greater than XMAX_BESS_K (=705.342).
	 * @param alpha Order for which K's or exponentially scaled K's (K*EXP(X)) are to be calculated.
	 * @param expo - set true if exponentially scaled K's are to be calculated. Else, if unscaled K's are to be calculated.
	 */
	public static final double k(double x, double alpha, boolean expo) {
		if (Double.isNaN(x) || Double.isNaN(alpha)) return x + alpha;
		if (x < 0) return Double.NaN;

		if (alpha < 0) alpha = -alpha;
		int nb = 1 + (int) floor(alpha);
		alpha -= (nb - 1);
		double[] bk = new double[nb];
		int ncalc = k_internal(x, alpha, expo, bk);
		if (ncalc != nb) {
			if (ncalc < 0)
				System.err.println(String.format("bessel_k(%g): ncalc (=%ld) != nb (=%ld); alpha=%g. Arg. out of range?", x, ncalc, nb, alpha));
			else
				System.err.println(String.format("bessel_k(%g,nu=%g): precision lost in result", x, alpha+(double) nb-1));
		}
		return bk[nb - 1];
	}


	private static final int j_internal(double x, double alpha, double[] b) {
		/* ---------------------------------------------------------------------
		  Mathematical constants

		   PI2	  = 2 / PI
		   TWOPI1 = first few significant digits of 2 * PI
		   TWOPI2 = (2*PI - TWOPI1) to working precision, i.e.,
			    TWOPI1 + TWOPI2 = 2 * PI to extra precision.
		 --------------------------------------------------------------------- */
		final double pi2 = .636619772367581343075535;
		final double twopi1 = 6.28125;
		final double twopi2 =  .001935307179586476925286767;

		/*---------------------------------------------------------------------
		 *  Factorial(N)
		 *--------------------------------------------------------------------- */
		final double fact[] = { 1.,1.,2.,6.,24.,120.,720.,5040.,40320.,
				362880.,3628800.,39916800.,479001600.,6227020800.,87178291200.,
				1.307674368e12,2.0922789888e13,3.55687428096e14,6.402373705728e15,
				1.21645100408832e17,2.43290200817664e18,5.109094217170944e19,
				1.12400072777760768e21,2.585201673888497664e22,
				6.2044840173323943936e23 };

		/* Local variables */
		long nend, intx, nbmx, k, l, m, nstart;
		int nb = b.length, ncalc, j;

		double nu, twonu, capp, capq, pold, vcos, test, vsin;
		double p, s, t, z, alpem, halfx, aa, bb, cc, psave, plast;
		double tover, t1, alp2em, em, en, xc, xk, xm, psavel, gnu, xin, sum;

		/* Parameter adjustment */
		//--b;

		nu = alpha;
		twonu = nu + nu;

		/*-------------------------------------------------------------------
	      Check for out of range arguments.
	      -------------------------------------------------------------------*/
		if (nb <= 0 || x < 0 || nu < 0 || nu > 1) {
			/* Error return -- X, NB, or ALPHA is out of range : */
			b[0] = 0.;
			return min(nb, 0) - 1;
		}

		//if (nb > 0 && x >= 0. && 0. <= nu && nu < 1.) {

		ncalc = nb;
		/* Initialize result array to zero. */
		Arrays.fill(b, 0);
		if(x > xlrg_BESS_IJ) {
			//ML_ERROR(ME_RANGE, "J_bessel");
			/* indeed, the limit is 0,
			 * but the cutoff happens too early */
			return ncalc;
		}
		intx = (long) (x);

		/*===================================================================
		  Branch into  3 cases :
		  1) use 2-term ascending series for small X
		  2) use asymptotic form for large X when NB is not too large
		  3) use recursion otherwise
		  ===================================================================*/

		if (x < rtnsig_BESS) {
			/* ---------------------------------------------------------------
		     Two-term ascending series for small X.
		     --------------------------------------------------------------- */
			alpem = 1. + nu;

			halfx = (x > enmten_BESS) ? .5 * x :  0.;
			aa	  = (nu != 0.)	  ? pow(halfx, nu) / (nu * gamma_cody(nu)) : 1.;
			bb	  = (x + 1. > 1.)? -halfx * halfx : 0.;
			b[0] = aa + aa * bb / alpem;
			if (x != 0. && b[0] == 0.)
				ncalc = 0;

			if (nb != 1) {
				if (x <= 0.) {
					for (int n = 1; n < nb; ++n)
						b[n] = 0.;
				}
				else {
					/* ----------------------------------------------
			       Calculate higher order functions.
			       ---------------------------------------------- */
					if (bb == 0.)
						tover = (enmten_BESS + enmten_BESS) / x;
					else
						tover = enmten_BESS / bb;
					cc = halfx;
					for (int n = 1; n < nb; ++n) {
						aa /= alpem;
						alpem += 1.;
						aa *= cc;
						if (aa <= tover * alpem)
							aa = 0.;

						b[n] = aa + aa * bb / alpem;
						if (b[n] == 0. && ncalc >= n)
							ncalc = n;
					}
				}
			}
		} else if (x > 25. && nb <= intx + 1) {
			/* ------------------------------------------------------------
		       Asymptotic series for X > 25 (and not too large nb)
		       ------------------------------------------------------------ */
			xc = sqrt(pi2 / x);
			xin = 1 / (64 * x * x);
			if (x >= 130.)	m = 4;
			else if (x >= 35.) m = 8;
			else		m = 11;
			xm = 4. * (double) m;
			/* ------------------------------------------------
		       Argument reduction for SIN and COS routines.
		       ------------------------------------------------ */
			t = trunc(x / (twopi1 + twopi2) + .5);
			z = (x - t * twopi1) - t * twopi2 - (nu + .5) / pi2;
			vsin = sin(z);
			vcos = cos(z);
			gnu = twonu;
			for (int i = 0; i < 2; ++i) {
				s = (xm - 1. - gnu) * (xm - 1. + gnu) * xin * .5;
				t = (gnu - (xm - 3.)) * (gnu + (xm - 3.));
				t1= (gnu - (xm + 1.)) * (gnu + (xm + 1.));
				k = m + m;
				capp = s * t / fact[(int) k];
				capq = s * t1/ fact[(int) k + 1];
				xk = xm;
				for (; k >= 4; k -= 2) {/* k + 2(j-2) == 2m */
					xk -= 4.;
					s = (xk - 1. - gnu) * (xk - 1. + gnu);
					t1 = t;
					t = (gnu - (xk - 3.)) * (gnu + (xk - 3.));
					capp = (capp + 1. / fact[(int) k - 2]) * s * t  * xin;
					capq = (capq + 1. / fact[(int) k - 1]) * s * t1 * xin;

				}
				capp += 1.;
				capq = (capq + 1.) * (gnu * gnu - 1.) * (.125 / x);
				b[i] = xc * (capp * vcos - capq * vsin);
				if (nb == 1)
					return ncalc;

				/* vsin <--> vcos */ t = vsin; vsin = -vcos; vcos = t;
				gnu += 2.;
			}
			/* -----------------------------------------------
		       If  NB > 2, compute J(X,ORDER+I)	for I = 2, NB-1
		       ----------------------------------------------- */
			if (nb > 2)
				for (gnu = twonu + 2., j = 2; j < nb; j++, gnu += 2.)
					b[j] = gnu * b[j - 1] / x - b[j - 2];
		}
		else {
			/* rtnsig_BESS <= x && ( x <= 25 || intx+1 < *nb ) :
		       --------------------------------------------------------
		       Use recurrence to generate results.
		       First initialize the calculation of P*S.
		       -------------------------------------------------------- */
			nbmx = nb - intx;
			long n = intx + 1;
			en = (double)(n + n) + twonu;
			plast = 1.;
			p = en / x;
			/* ---------------------------------------------------
		       Calculate general significance test.
		       --------------------------------------------------- */
			test = ensig_BESS + ensig_BESS;
			boolean skip_to_L190 = false;
			if (nbmx >= 3) {
				/* ------------------------------------------------------------
			   Calculate P*S until N = NB-1.  Check for possible overflow.
			   ---------------------------------------------------------- */
				tover = enten_BESS / ensig_BESS;
				nstart = intx + 2;
				nend = nb - 1;
				en = (double) (nstart + nstart) - 2. + twonu;
				for (k = nstart; k <= nend; ++k) {
					n = k;
					en += 2.;
					pold = plast;
					plast = p;
					p = en * plast / x - pold;
					if (p > tover) {
						/* -------------------------------------------
								To avoid overflow, divide P*S by TOVER.
								Calculate P*S until ABS(P) > 1.
							-------------------------------------------*/
						tover = enten_BESS;
						p /= tover;
						plast /= tover;
						psave = p;
						psavel = plast;
						nstart = n + 1;
						do {
							++n;
							en += 2.;
							pold = plast;
							plast = p;
							p = en * plast / x - pold;
						} while (p <= 1.);

						bb = en / x;
						/* -----------------------------------------------
				   Calculate backward test and find NCALC,
				   the highest N such that the test is passed.
				   ----------------------------------------------- */
						test = pold * plast * (.5 - .5 / (bb * bb));
						test /= ensig_BESS;
						p = plast * tover;
						--n;
						en -= 2.;
						nend = min(nb,n);
						for (l = nstart; l <= nend; ++l) {
							pold = psavel;
							psavel = psave;
							psave = en * psavel / x - pold;
							if (psave * psavel > test) {
								ncalc = (int) (l - 1);
								//goto L190;
								skip_to_L190 = true;
								break;
							}
						}
						if (!skip_to_L190)
							ncalc = (int) (nend);
						//goto L190;
						skip_to_L190 = true;
						break;
					}
				}
				if (!skip_to_L190) {
					n = nend;
					en = (double) (n + n) + twonu;
					/* -----------------------------------------------------
				   Calculate special significance test for NBMX > 2.
				   -----------------------------------------------------*/
					test = max(test, sqrt(plast * ensig_BESS) * sqrt(p + p));
				}
			}
			/* ------------------------------------------------
		       Calculate P*S until significance test passes. */
			if (!skip_to_L190) {
				do {
					++n;
					en += 2.;
					pold = plast;
					plast = p;
					p = en * plast / x - pold;
				} while (p < test);
			}

			//L190:
			skip_to_L190 = false;
			/*---------------------------------------------------------------
		      Initialize the backward recursion and the normalization sum.
		      --------------------------------------------------------------- */
			++n;
			en += 2.;
			bb = 0.;
			aa = 1. / p;
			m = n / 2;
			em = (double)m;
			m = (n << 1) - (m << 2);/* = 2 n - 4 (n/2)
					       = 0 for even, 2 for odd n */
			if (m == 0)
				sum = 0.;
			else {
				alpem = em - 1. + nu;
				alp2em = em + em + nu;
				sum = aa * alpem * alp2em / em;
			}
			nend = n - nb;
			/* if (nend > 0) */
			/* --------------------------------------------------------
		       Recur backward via difference equation, calculating
		       (but not storing) b[N], until N = NB.
		       -------------------------------------------------------- */
			for (l = 1; l <= nend; ++l) {
				--n;
				en -= 2.;
				cc = bb;
				bb = aa;
				aa = en * bb / x - cc;
				m = m != 0 ? 0 : 2; /* m = 2 - m failed on gcc4-20041019 */
				if (m != 0) {
					em -= 1.;
					alp2em = em + em + nu;
					if (n == 1)
						break;

					alpem = em - 1. + nu;
					if (alpem == 0.)
						alpem = 1.;
					sum = (sum + aa * alp2em) * alpem / em;
				}
			}
			/*--------------------------------------------------
		      Store b[NB].
		      --------------------------------------------------*/
			b[(int) (n - 1)] = aa;
			boolean skip_to_L250 = false, skip_to_L240 = false;
			if (nend >= 0) {
				if (nb <= 1) {
					if (nu + 1. == 1.)
						alp2em = 1.;
					else
						alp2em = nu;
					sum += b[0] * alp2em;
					// goto L250;
					skip_to_L250 = true;
				}
				else {/*-- nb >= 2 : ---------------------------
				Calculate and store b[NB-1].
				----------------------------------------*/
					--n;
					en -= 2.;
					b[(int) (n - 1)] = en * aa / x - bb;
					//if (n == 1)
					//	goto L240;

					if (n != 1) {
						m = m != 0 ? 0 : 2; /* m = 2 - m failed on gcc4-20041019 */
						if (m != 0) {
							em -= 1.;
							alp2em = em + em + nu;
							alpem = em - 1. + nu;
							if (alpem == 0.)
								alpem = 1.;
							sum = (sum + b[(int) (n - 1)] * alp2em) * alpem / em;
						}
					} else {
						skip_to_L240 = true;
					}
				}
			}

			if (!skip_to_L240 && !skip_to_L250) {
				/* if (n - 2 != 0) */
				/* --------------------------------------------------------
			       Calculate via difference equation and store b[N],
			       until N = 2.
			       -------------------------------------------------------- */
				for (n = n-1; n >= 2; n--) {
					en -= 2.;
					b[(int) (n - 1)] = en * b[(int) n] / x - b[(int) (n + 1)];
					m = m != 0 ? 0 : 2; /* m = 2 - m failed on gcc4-20041019 */
					if (m != 0) {
						em -= 1.;
						alp2em = em + em + nu;
						alpem = em - 1. + nu;
						if (alpem == 0.)
							alpem = 1.;
						sum = (sum + b[(int) (n - 1)] * alp2em) * alpem / em;
					}
				}
				/* ---------------------------------------
			       Calculate b[1].
			       -----------------------------------------*/
				b[0] = 2. * (nu + 1.) * b[1] / x - b[2];
			}

			//L240:
			skip_to_L240 = false;
			if (!skip_to_L250) {
				em -= 1.;
				alp2em = em + em + nu;
				if (alp2em == 0.)
					alp2em = 1.;
				sum += b[0] * alp2em;
			}

			//L250:
			skip_to_L250 = false;
			/* ---------------------------------------------------
		       Normalize.  Divide all b[N] by sum.
		       ---------------------------------------------------*/
			/*	    if (nu + 1. != 1.) poor test */
			if(abs(nu) > 1e-15)
				sum *= (gamma_cody(nu) * pow(.5* x, -nu));

			aa = enmten_BESS;
			if (sum > 1.)
				aa *= sum;
			for (int nn = 0; nn < nb; ++nn) {
				if (abs(b[nn]) < aa)
					b[nn] = 0.;
				else
					b[nn] /= sum;
			}
		}
		//}
		return ncalc;
	}

	private static final int y_internal(double x, double alpha, double[] by) {
		/* ----------------------------------------------------------------------
		  Mathematical constants
		    FIVPI = 5*PI
		    PIM5 = 5*PI - 15
		 ----------------------------------------------------------------------*/
		final double fivpi = 15.707963267948966192;
		final double pim5	=   .70796326794896619231;

		/*----------------------------------------------------------------------
		      Coefficients for Chebyshev polynomial expansion of
		      1/gamma(1-x), abs(x) <= .5
		      ----------------------------------------------------------------------*/
		final double ch[] = { -6.7735241822398840964e-24,
				-6.1455180116049879894e-23,2.9017595056104745456e-21,
				1.3639417919073099464e-19,2.3826220476859635824e-18,
				-9.0642907957550702534e-18,-1.4943667065169001769e-15,
				-3.3919078305362211264e-14,-1.7023776642512729175e-13,
				9.1609750938768647911e-12,2.4230957900482704055e-10,
				1.7451364971382984243e-9,-3.3126119768180852711e-8,
				-8.6592079961391259661e-7,-4.9717367041957398581e-6,
				7.6309597585908126618e-5,.0012719271366545622927,
				.0017063050710955562222,-.07685284084478667369,
				-.28387654227602353814,.92187029365045265648 };

		/* Local variables */
		long k, na;
		int nb = by.length, ncalc, i;

		double alfa, div, ddiv, even, gamma, term, cosmu, sinmu,
		b, c, d, e, f, g, h, p, q, r, s, d1, d2, q0, pa,pa1, qa,qa1,
		en, en1, nu, ex,  ya,ya1, twobyx, den, odd, aye, dmu, x2, xna;

		en1 = ya = ya1 = 0;		/* -Wall */

		ex = x;
		nu = alpha;
		if (nb > 0 && 0. <= nu && nu < 1.) {
			if(ex < DBL_MIN || ex > xlrg_BESS_Y) {
				/* Warning is not really appropriate, give
				 * proper limit:
				 * ML_ERROR(ME_RANGE, "Y_bessel"); */
				ncalc = nb;
				if(ex > xlrg_BESS_Y)  by[0]= 0.; /*was ML_POSINF */
				else if(ex < DBL_MIN) by[0]=Double.NEGATIVE_INFINITY;
				for(i=0; i < nb; i++)
					by[i] = by[0];
				return ncalc;
			}
			xna = trunc(nu + .5);
			na = (long) xna;
			if (na == 1) {/* <==>  .5 <= *alpha < 1	 <==>  -5. <= nu < 0 */
				nu -= xna;
			}
			if (nu == -.5) {
				p = M_SQRT_2dPI / sqrt(ex);
				ya = p * sin(ex);
				ya1 = -p * cos(ex);
			} else if (ex < 3.) {
				/* -------------------------------------------------------------
			       Use Temme's scheme for small X
			       ------------------------------------------------------------- */
				b = ex * .5;
				d = -log(b);
				f = nu * d;
				e = pow(b, -nu);
				if (abs(nu) < M_eps_sinc)
					c = M_1_PI;
				else
					c = nu / sinpi(nu);

				/* ------------------------------------------------------------
			       Computation of sinh(f)/f
			       ------------------------------------------------------------ */
				if (abs(f) < 1.) {
					x2 = f * f;
					en = 19.;
					s = 1.;
					for (i = 1; i <= 9; ++i) {
						s = s * x2 / en / (en - 1.) + 1.;
						en -= 2.;
					}
				} else {
					s = (e - 1. / e) * .5 / f;
				}
				/* --------------------------------------------------------
			       Computation of 1/gamma(1-a) using Chebyshev polynomials */
				x2 = nu * nu * 8.;
				aye = ch[0];
				even = 0.;
				alfa = ch[1];
				odd = 0.;
				for (i = 3; i <= 19; i += 2) {
					even = -(aye + aye + even);
					aye = -even * x2 - aye + ch[i - 1];
					odd = -(alfa + alfa + odd);
					alfa = -odd * x2 - alfa + ch[i];
				}
				even = (even * .5 + aye) * x2 - aye + ch[20];
				odd = (odd + alfa) * 2.;
				gamma = odd * nu + even;
				/* End of computation of 1/gamma(1-a)
			       ----------------------------------------------------------- */
				g = e * gamma;
				e = (e + 1. / e) * .5;
				f = 2. * c * (odd * e + even * s * d);
				e = nu * nu;
				p = g * c;
				q = M_1_PI / g;
				c = nu * M_PI_2;
				if (abs(c) < M_eps_sinc)
					r = 1.;
				else
					r = sinpi(nu/2.) / c;

				r = PI * c * r * r;
				c = 1.;
				d = -b * b;
				h = 0.;
				ya = f + r * q;
				ya1 = p;
				en = 1.;

				while (abs(g / (1. + abs(ya))) +
						abs(h / (1. + abs(ya1))) > DBL_EPSILON) {
					f = (f * en + p + q) / (en * en - e);
					c *= (d / en);
					p /= en - nu;
					q /= en + nu;
					g = c * (f + r * q);
					h = c * p - en * g;
					ya += g;
					ya1+= h;
					en += 1.;
				}
				ya = -ya;
				ya1 = -ya1 / b;
			} else if (ex < thresh_BESS_Y) {
				/* --------------------------------------------------------------
			       Use Temme's scheme for moderate X :  3 <= x < 16
			       -------------------------------------------------------------- */
				c = (.5 - nu) * (.5 + nu);
				b = ex + ex;
				e = ex * M_1_PI * cospi(nu) / DBL_EPSILON;
				e *= e;
				p = 1.;
				q = -ex;
				r = 1. + ex * ex;
				s = r;
				en = 2.;
				while (r * en * en < e) {
					en1 = en + 1.;
					d = (en - 1. + c / en) / s;
					p = (en + en - p * d) / en1;
					q = (-b + q * d) / en1;
					s = p * p + q * q;
					r *= s;
					en = en1;
				}
				f = p / s;
				p = f;
				g = -q / s;
				q = g;
				//L220:
				en -= 1.;
				while (en > 0.) {
					r = en1 * (2. - p) - 2.;
					s = b + en1 * q;
					d = (en - 1. + c / en) / (r * r + s * s);
					p = d * r;
					q = d * s;
					e = f + 1.;
					f = p * e - g * q;
					g = q * e + p * g;
					en1 = en;
					//goto L220;
					en -= 1.;
				}
				f = 1. + f;
				d = f * f + g * g;
				pa = f / d;
				qa = -g / d;
				d = nu + .5 - p;
				q += ex;
				pa1 = (pa * q - qa * d) / ex;
				qa1 = (qa * q + pa * d) / ex;
				b = ex - M_PI_2 * (nu + .5);
				c = cos(b);
				s = sin(b);
				d = M_SQRT_2dPI / sqrt(ex);
				ya = d * (pa * s + qa * c);
				ya1 = d * (qa1 * s - pa1 * c);
			} else { /* x > thresh_BESS_Y */
				/* ----------------------------------------------------------
			       Use Campbell's asymptotic scheme.
			       ---------------------------------------------------------- */
				na = 0;
				d1 = trunc(ex / fivpi);
				i = (int) d1;
				dmu = ex - 15. * d1 - d1 * pim5 - (alpha + .5) * M_PI_2;
				if (i - (i / 2 << 1) == 0) {
					cosmu = cos(dmu);
					sinmu = sin(dmu);
				} else {
					cosmu = -cos(dmu);
					sinmu = -sin(dmu);
				}
				ddiv = 8. * ex;
				dmu = alpha;
				den = sqrt(ex);
				for (k = 1; k <= 2; ++k) {
					p = cosmu;
					cosmu = sinmu;
					sinmu = -p;
					d1 = (2. * dmu - 1.) * (2. * dmu + 1.);
					d2 = 0.;
					div = ddiv;
					p = 0.;
					q = 0.;
					q0 = d1 / div;
					term = q0;
					for (i = 2; i <= 20; ++i) {
						d2 += 8.;
						d1 -= d2;
						div += ddiv;
						term = -term * d1 / div;
						p += term;
						d2 += 8.;
						d1 -= d2;
						div += ddiv;
						term *= (d1 / div);
						q += term;
						if (abs(term) <= DBL_EPSILON) {
							break;
						}
					}
					p += 1.;
					q += q0;
					if (k == 1)
						ya = M_SQRT_2dPI * (p * cosmu - q * sinmu) / den;
					else
						ya1 = M_SQRT_2dPI * (p * cosmu - q * sinmu) / den;
					dmu += 1.;
				}
			}
			if (na == 1) {
				h = 2. * (nu + 1.) / ex;
				if (h > 1.) {
					if (abs(ya1) > Double.MAX_VALUE / h) {
						h = 0.;
						ya = 0.;
					}
				}
				h = h * ya1 - ya;
				ya = ya1;
				ya1 = h;
			}

			/* ---------------------------------------------------------------
			   Now have first one or two Y's
			   --------------------------------------------------------------- */
			by[0] = ya;
			ncalc = 1;
			if(nb > 1) {
				by[1] = ya1;
				if (ya1 != 0.) {
					aye = 1. + alpha;
					twobyx = 2. / ex;
					ncalc = 2;
					for (i = 2; i < nb; ++i) {
						if (twobyx < 1.) {
							if (abs(by[i - 1]) * twobyx >= Double.MAX_VALUE / aye)
								break; //goto L450;
						} else {
							if (abs(by[i - 1]) >= Double.MAX_VALUE / aye / twobyx)
								break; //goto L450;
						}
						by[i] = twobyx * aye * by[i - 1] - by[i - 2];
						aye += 1.;
						++(ncalc);
					}
				}
			}
			//L450:
			for (i = ncalc; i < nb; ++i)
				by[i] = Double.NEGATIVE_INFINITY;/* was 0 */

		} else {
			by[0] = 0.;
			ncalc = min(nb,0) - 1;
		}
		return ncalc;
	}

	private static final int i_internal(double x, double alpha, boolean expo, double[] bi) {
		/*-------------------------------------------------------------------
	      Mathematical constants
	      -------------------------------------------------------------------*/
		final double const__ = 1.585;

		/* Local variables */
		long nend, intx, nbmx, k, l, n, nstart;
		double pold, test,	p, em, en, empal, emp2al, halfx,
		aa, bb, cc, psave, plast, tover, psavel, sum, nu, twonu;
		int nb = bi.length, ncalc = 0, ize = expo ? 2 : 1;

		/*Parameter adjustments */
		//--bi;
		nu = alpha;
		twonu = nu + nu;

		/*-------------------------------------------------------------------
	      Check for X, NB, OR IZE out of range.
	      ------------------------------------------------------------------- */
		if (nb > 0 && x >= 0. &&	(0. <= nu && nu < 1.) /*&&
				(1 <= ize && ize <= 2) */ ) {

			ncalc = nb;
			if(ize == 1 && x > exparg_BESS) {
				//for(k=1; k <= nb; k++)
				//	bi[(int) k]=Double.POSITIVE_INFINITY; /* the limit *is* = Inf */
				Arrays.fill(bi, Double.POSITIVE_INFINITY);
				return ncalc;
			}
			if(ize == 2 && x > xlrg_BESS_IJ) {
				//for(k=1; k <= nb; k++)
				//	bi[(int) k]= 0.; /* The limit exp(-x) * I_nu(x) --> 0 : */
				Arrays.fill(bi, 0);
				return ncalc;
			}
			intx = (long) (x);/* fine, since *x <= xlrg_BESS_IJ <<< LONG_MAX */
			if (x >= rtnsig_BESS) { /* "non-small" x ( >= 1e-4 ) */
				/* -------------------------------------------------------------------
					Initialize the forward sweep, the P-sequence of Olver
				------------------------------------------------------------------- */
				nbmx = nb - intx;
				n = intx + 1;
				en = (double) (n + n) + twonu;
				plast = 1.;
				p = en / x;
				/* ------------------------------------------------
		       Calculate general significance test
		       ------------------------------------------------ */
				test = ensig_BESS + ensig_BESS;
				if (intx << 1 > nsig_BESS * 5) {
					test = sqrt(test * p);
				} else {
					test /= pow(const__, (double)intx);
				}
				boolean skip_to_L120 = false;
				if (nbmx >= 3) {
					/* --------------------------------------------------
						Calculate P-sequence until N = NB-1
						Check for possible overflow.
					------------------------------------------------ */
					tover = enten_BESS / ensig_BESS;
					nstart = intx + 2;
					nend = nb - 1;
					for (k = nstart; k <= nend; ++k) {
						n = k;
						en += 2.;
						pold = plast;
						plast = p;
						p = en * plast / x + pold;
						if (p > tover) {
							/* ------------------------------------------------
								To avoid overflow, divide P-sequence by TOVER.
								Calculate P-sequence until ABS(P) > 1.
	 							---------------------------------------------- */
							tover = enten_BESS;
							p /= tover;
							plast /= tover;
							psave = p;
							psavel = plast;
							nstart = n + 1;
							do {
								++n;
								en += 2.;
								pold = plast;
								plast = p;
								p = en * plast / x + pold;
							}
							while (p <= 1.);

							bb = en / x;
							/* ------------------------------------------------
								Calculate backward test, and find NCALC,
								the highest N such that the test is passed.
							------------------------------------------------ */
							test = pold * plast / ensig_BESS;
							test *= .5 - .5 / (bb * bb);
							p = plast * tover;
							--n;
							en -= 2.;
							nend = min(nb,n);
							boolean skip_to_L90 = false;
							for (l = nstart; l <= nend; ++l) {
								ncalc = (int) l;
								pold = psavel;
								psavel = psave;
								psave = en * psavel / x + pold;
								if (psave * psavel > test) {
									// goto L90;
									skip_to_L90 = true;
									break;
								}
							}
							if (!skip_to_L90)
								ncalc = (int) (nend + 1);
							//L90:
							skip_to_L90 = false;
							--(ncalc);
							//goto L120;
							skip_to_L120 = true;
							break;
						}
					}
					if (!skip_to_L120) {
						n = nend;
						en = (double)(n + n) + twonu;
						/*---------------------------------------------------
							Calculate special significance test for NBMX > 2.
						--------------------------------------------------- */
						test = max(test,sqrt(plast * ensig_BESS) * sqrt(p + p));
					}
				}
				if (!skip_to_L120) {
					/* --------------------------------------------------------
						Calculate P-sequence until significance test passed.
					-------------------------------------------------------- */
					do {
						++n;
						en += 2.;
						pold = plast;
						plast = p;
						p = en * plast / x + pold;
					} while (p < test);
				}

				//L120:
				skip_to_L120 = false;
				/* -------------------------------------------------------------------
					Initialize the backward recursion and the normalization sum.
				------------------------------------------------------------------- */
				++n;
				en += 2.;
				bb = 0.;
				aa = 1. / p;
				em = (double) n - 1.;
				empal = em + nu;
				emp2al = em - 1. + twonu;
				sum = aa * empal * emp2al / em;
				nend = n - nb;
				boolean skip_to_L220 = false, skip_to_L230 = false;
				if (nend < 0) {
					/* -----------------------------------------------------
						N < NB, so store BI[N] and set higher orders to 0..
					----------------------------------------------------- */
					bi[(int) (n - 1)] = aa;
					nend = -nend;
					for (l = 1; l <= nend; ++l) {
						bi[(int) (n + l - 1)] = 0.;
					}
				} else {
					if (nend > 0) {
						/* -----------------------------------------------------
							Recur backward via difference equation,
							calculating (but not storing) BI[N], until N = NB.
						--------------------------------------------------- */

						for (l = 1; l <= nend; ++l) {
							--n;
							en -= 2.;
							cc = bb;
							bb = aa;
							/* for x ~= 1500,  sum would overflow to 'inf' here,
							 * and the final bi[] /= sum would give 0 wrongly;
							 * RE-normalize (aa, sum) here -- no need to undo */
							if(nend > 100 && aa > 1e200) {
								/* multiply by  2^-900 = 1.18e-271 */
								cc	= ldexp(cc, -900);
								bb	= ldexp(bb, -900);
								sum = ldexp(sum,-900);
							}
							aa = en * bb / x + cc;
							em -= 1.;
							emp2al -= 1.;
							if (n == 1) {
								break;
							}
							if (n == 2) {
								emp2al = 1.;
							}
							empal -= 1.;
							sum = (sum + aa * empal) * emp2al / em;
						}
					} // end if (nend > 0)
					/* ---------------------------------------------------
						Store BI[NB]
					--------------------------------------------------- */
					bi[(int) (n - 1)] = aa;
					if (nb <= 1) {
						sum = sum + sum + aa;
						// goto L230;
						skip_to_L230 = true;
					} else {
						/* -------------------------------------------------
							Calculate and Store BI[NB-1]
						------------------------------------------------- */
						--n;
						en -= 2.;
						bi[(int) (n - 1)] = en * aa / x + bb;
						//if (n == 1) {
						//	goto L220;
						//}
						if (n != 1) {
							em -= 1.;
							if (n == 2)
								emp2al = 1.;
							else
								emp2al -= 1.;
		
							empal -= 1.;
							sum = (sum + bi[(int) (n - 1)] * empal) * emp2al / em;
						} else {
							// goto L220;
							skip_to_L220 = true;
						}
					}
				} // end if (nend < 0)
				if (!skip_to_L220 && !skip_to_L230) {
					nend = n - 2;
					if (nend > 0) {
						/* --------------------------------------------
							Calculate via difference equation
							and store BI[N], until N = 2.
						------------------------------------------ */
						for (l = 1; l <= nend; ++l) {
							--n;
							en -= 2.;
							bi[(int) (n - 1)] = en * bi[(int) n] / x + bi[(int) n + 1];
							em -= 1.;
							if (n == 2)
								emp2al = 1.;
							else
								emp2al -= 1.;
							empal -= 1.;
							sum = (sum + bi[(int) (n - 1)] * empal) * emp2al / em;
						}
					}
					/* ----------------------------------------------
						Calculate BI[1]
					-------------------------------------------- */
					bi[0] = 2. * empal * bi[1] / x + bi[2];
				}
				//L220:
				skip_to_L220 = false;
				if (!skip_to_L230) {
					sum = sum + sum + bi[0];
				}

				//L230:
				skip_to_L230 = false;
					/* ---------------------------------------------------------
		       Normalize.  Divide all BI[N] by sum.
		       --------------------------------------------------------- */
					if (nu != 0.)
						sum *= (gamma_cody(1. + nu) * pow(x * .5, -nu));
				if (ize == 1)
					sum *= exp(-(x));
				aa = enmten_BESS;
				if (sum > 1.)
					aa *= sum;
				for (n = 1; n <= nb; ++n) {
					if (bi[(int) (n - 1)] < aa)
						bi[(int) (n - 1)] = 0.;
					else
						bi[(int) (n - 1)] /= sum;
				}
				return ncalc;
			} else { /* small x  < 1e-4 */
				/* -----------------------------------------------------------
		       Two-term ascending series for small X.
		       -----------------------------------------------------------*/
				aa = 1.;
				empal = 1. + nu;
				//#ifdef IEEE_754
				/* No need to check for underflow */
				halfx = .5 * x;
				//#else
				//	if (*x > enmten_BESS) */
				//	halfx = .5 * *x;
				//	else
				//		halfx = 0.;
				//#endif
				if (nu != 0.)
					aa = pow(halfx, nu) / gamma_cody(empal);
				if (ize == 2)
					aa *= exp(-(x));
				bb = halfx * halfx;
				bi[0] = aa + aa * bb / empal;
				if (x != 0. && bi[0] == 0.)
					ncalc = 0;
				if (nb > 1) {
					if (x == 0.) {
						for (n = 2; n <= nb; ++n)
							bi[(int) (n - 1)] = 0.;
					} else {
						/* -------------------------------------------------
			       Calculate higher-order functions.
			       ------------------------------------------------- */
						cc = halfx;
						tover = (enmten_BESS + enmten_BESS) / x;
						if (bb != 0.)
							tover = enmten_BESS / bb;
						for (n = 2; n <= nb; ++n) {
							aa /= empal;
							empal += 1.;
							aa *= cc;
							if (aa <= tover * empal)
								bi[(int) (n - 1)] = aa = 0.;
							else
								bi[(int) (n - 1)] = aa + aa * bb / empal;
							if (bi[(int) (n - 1)] == 0. && ncalc > n)
								ncalc = (int) (n - 1);
						}
					}
				}
			}
		} else { /* argument out of range */
			ncalc = min(nb,0) - 1;
		}
		return ncalc;
	}

	private static final int k_internal(double x, double alpha, boolean expo, double[] bk) {
		/*---------------------------------------------------------------------
		 * Mathematical constants
		 *	A = LOG(2) - Euler's constant
		 *	D = SQRT(2/PI)
	     ---------------------------------------------------------------------*/
		final double a = .11593151565841244881;

		/*---------------------------------------------------------------------
	      P, Q - Approximation for LOG(GAMMA(1+ALPHA))/ALPHA + Euler's constant
	      Coefficients converted from hex to decimal and modified
	      by W. J. Cody, 2/26/82 */
		final double p[] = { .805629875690432845,20.4045500205365151,
				157.705605106676174,536.671116469207504,900.382759291288778,
				730.923886650660393,229.299301509425145,.822467033424113231 };
		final double q[] = { 29.4601986247850434,277.577868510221208,
				1206.70325591027438,2762.91444159791519,3443.74050506564618,
				2210.63190113378647,572.267338359892221 };
		/* R, S - Approximation for (1-ALPHA*PI/SIN(ALPHA*PI))/(2.D0*ALPHA) */
		final double r[] = { -.48672575865218401848,13.079485869097804016,
				-101.96490580880537526,347.65409106507813131,
				3.495898124521934782e-4 };
		final double s[] = { -25.579105509976461286,212.57260432226544008,
				-610.69018684944109624,422.69668805777760407 };
		/* T    - Approximation for SINH(Y)/Y */
		final double t[] = { 1.6125990452916363814e-10,
				2.5051878502858255354e-8,2.7557319615147964774e-6,
				1.9841269840928373686e-4,.0083333333333334751799,
				.16666666666666666446 };
		/*---------------------------------------------------------------------*/
		final double estm[] = { 52.0583,5.7607,2.7782,14.4303,185.3004, 9.3715 };
		final double estf[] = { 41.8341,7.1075,6.4306,42.511,1.35633,84.5096,20.};

		/* Local variables */
		long iend, m, ii, mplus1;
		double x2by4, twox, c, blpha, ratio, wminf = 0;
		double d1, d2, d3, f0, f1, f2, p0, q0, t1, t2, twonu;
		double dm, ex, bk1 = 0, bk2 = 0, nu;
		int nb = bk.length, ncalc, i, j, k, ize = expo ? 2 : 1;

		ii = 0; /* -Wall */

		ex = x;
		nu = alpha;
		ncalc = min(nb,0) - 2;
		if (nb > 0 && (0. <= nu && nu < 1.) /*&& (1 <= ize && ize <= 2)*/) {
			if(ex <= 0 || (ize == 1 && ex > xmax_BESS_K)) {
				if(ex <= 0) {
					if(ex < 0) throw new RuntimeException("K_bessel"); //ML_ERROR(ME_RANGE, "K_bessel");
					//for(i=0; i < nb; i++)
					//	bk[i] = Double.POSITIVE_INFINITY;
					Arrays.fill(bk, Double.POSITIVE_INFINITY);
				} else /* would only have underflow */
					//for(i=0; i < nb; i++)
					//	bk[i] = 0.;
					Arrays.fill(bk, 0);
				//*ncalc = *nb;
				return nb;
			}
			k = 0;
			if (nu < sqxmin_BESS_K) {
				nu = 0.;
			} else if (nu > .5) {
				k = 1;
				nu -= 1.;
			}
			twonu = nu + nu;
			iend = nb + k - 1;
			c = nu * nu;
			d3 = -c;
			if (ex <= 1.) {
				/* ------------------------------------------------------------
		       Calculation of P0 = GAMMA(1+ALPHA) * (2/X)**ALPHA
				      Q0 = GAMMA(1-ALPHA) * (X/2)**ALPHA
		       ------------------------------------------------------------ */
				d1 = 0.; d2 = p[0];
				t1 = 1.; t2 = q[0];
				for (i = 2; i <= 7; i += 2) {
					d1 = c * d1 + p[i - 1];
					d2 = c * d2 + p[i];
					t1 = c * t1 + q[i - 1];
					t2 = c * t2 + q[i];
				}
				d1 = nu * d1;
				t1 = nu * t1;
				f1 = log(ex);
				f0 = a + nu * (p[7] - nu * (d1 + d2) / (t1 + t2)) - f1;
				q0 = exp(-nu * (a - nu * (p[7] + nu * (d1-d2) / (t1-t2)) - f1));
				f1 = nu * f0;
				p0 = exp(f1);
				/* -----------------------------------------------------------
		       Calculation of F0 =
		       ----------------------------------------------------------- */
				d1 = r[4];
				t1 = 1.;
				for (i = 0; i < 4; ++i) {
					d1 = c * d1 + r[i];
					t1 = c * t1 + s[i];
				}
				/* d2 := sinh(f1)/ nu = sinh(f1)/(f1/f0)
				 *	   = f0 * sinh(f1)/f1 */
				if (abs(f1) <= .5) {
					f1 *= f1;
					d2 = 0.;
					for (i = 0; i < 6; ++i) {
						d2 = f1 * d2 + t[i];
					}
					d2 = f0 + f0 * f1 * d2;
				} else {
					d2 = sinh(f1) / nu;
				}
				f0 = d2 - nu * d1 / (t1 * p0);
				if (ex <= 1e-10) {
					/* ---------------------------------------------------------
			   X <= 1.0E-10
			   Calculation of K(ALPHA,X) and X*K(ALPHA+1,X)/K(ALPHA,X)
			   --------------------------------------------------------- */
					bk[0] = f0 + ex * f0;
					if (ize == 1) {
						bk[0] -= ex * bk[0];
					}
					ratio = p0 / f0;
					c = ex * Double.MAX_VALUE;
					if (k != 0) {
						/* ---------------------------------------------------
			       Calculation of K(ALPHA,X)
			       and  X*K(ALPHA+1,X)/K(ALPHA,X),	ALPHA >= 1/2
			       --------------------------------------------------- */
						ncalc = -1;
						if (bk[0] >= c / ratio) {
							return ncalc;
						}
						bk[0] = ratio * bk[0] / ex;
						twonu += 2.;
						ratio = twonu;
					}
					ncalc = 1;
					if (nb == 1)
						return ncalc;

					/* -----------------------------------------------------
			   Calculate  K(ALPHA+L,X)/K(ALPHA+L-1,X),
			   L = 1, 2, ... , NB-1
			   ----------------------------------------------------- */
					ncalc = -1;
					for (i = 1; i < nb; ++i) {
						if (ratio >= c)
							return ncalc;

						bk[i] = ratio / ex;
						twonu += 2.;
						ratio = twonu;
					}
					ncalc = 1;
					//goto L420;
					for (i = ncalc; i < nb; ++i) { /* i == *ncalc */
						//#ifndef IEEE_754
						//if (bk[i-1] >= DBL_MAX / bk[i])
						//	return;
						//#endif
						bk[i] *= bk[i-1];
						(ncalc)++;
					}
				} else {
					/* ------------------------------------------------------
			   10^-10 < X <= 1.0
			   ------------------------------------------------------ */
					c = 1.;
					x2by4 = ex * ex / 4.;
					p0 = .5 * p0;
					q0 = .5 * q0;
					d1 = -1.;
					d2 = 0.;
					bk1 = 0.;
					bk2 = 0.;
					f1 = f0;
					f2 = p0;
					do {
						d1 += 2.;
						d2 += 1.;
						d3 = d1 + d3;
						c = x2by4 * c / d2;
						f0 = (d2 * f0 + p0 + q0) / d3;
						p0 /= d2 - nu;
						q0 /= d2 + nu;
						t1 = c * f0;
						t2 = c * (p0 - d2 * f0);
						bk1 += t1;
						bk2 += t2;
					} while (abs(t1 / (f1 + bk1)) > DBL_EPSILON ||
							abs(t2 / (f2 + bk2)) > DBL_EPSILON);
					bk1 = f1 + bk1;
					bk2 = 2. * (f2 + bk2) / ex;
					if (ize == 2) {
						d1 = exp(ex);
						bk1 *= d1;
						bk2 *= d1;
					}
					wminf = estf[0] * ex + estf[1];
				}
			} else if (DBL_EPSILON * ex > 1.) {
				/* -------------------------------------------------
		       X > 1./EPS
		       ------------------------------------------------- */
				ncalc = nb;
				bk1 = 1. / (M_SQRT_2dPI * sqrt(ex));
				for (i = 0; i < nb; ++i)
					bk[i] = bk1;
				return ncalc;

			} else {
				/* -------------------------------------------------------
		       X > 1.0
		       ------------------------------------------------------- */
				twox = ex + ex;
				blpha = 0.;
				ratio = 0.;
				if (ex <= 4.) {
					/* ----------------------------------------------------------
			   Calculation of K(ALPHA+1,X)/K(ALPHA,X),  1.0 <= X <= 4.0
			   ----------------------------------------------------------*/
					d2 = trunc(estm[0] / ex + estm[1]);
					m = (long) d2;
					d1 = d2 + d2;
					d2 -= .5;
					d2 *= d2;
					for (i = 2; i <= m; ++i) {
						d1 -= 2.;
						d2 -= d1;
						ratio = (d3 + d2) / (twox + d1 - ratio);
					}
					/* -----------------------------------------------------------
			   Calculation of I(|ALPHA|,X) and I(|ALPHA|+1,X) by backward
			   recurrence and K(ALPHA,X) from the wronskian
			   -----------------------------------------------------------*/
					d2 = trunc(estm[2] * ex + estm[3]);
					m = (long) d2;
					c = abs(nu);
					d3 = c + c;
					d1 = d3 - 1.;
					f1 = DBL_MIN;
					f0 = (2. * (c + d2) / ex + .5 * ex / (c + d2 + 1.)) * DBL_MIN;
					for (i = 3; i <= m; ++i) {
						d2 -= 1.;
						f2 = (d3 + d2 + d2) * f0;
						blpha = (1. + d1 / d2) * (f2 + blpha);
						f2 = f2 / ex + f1;
						f1 = f0;
						f0 = f2;
					}
					f1 = (d3 + 2.) * f0 / ex + f1;
					d1 = 0.;
					t1 = 1.;
					for (i = 1; i <= 7; ++i) {
						d1 = c * d1 + p[i - 1];
						t1 = c * t1 + q[i - 1];
					}
					p0 = exp(c * (a + c * (p[7] - c * d1 / t1) - log(ex))) / ex;
					f2 = (c + .5 - ratio) * f1 / ex;
					bk1 = p0 + (d3 * f0 - f2 + f0 + blpha) / (f2 + f1 + f0) * p0;
					if (ize == 1) {
						bk1 *= exp(-ex);
					}
					wminf = estf[2] * ex + estf[3];
				} else {
					/* ---------------------------------------------------------
			   Calculation of K(ALPHA,X) and K(ALPHA+1,X)/K(ALPHA,X), by
			   backward recurrence, for  X > 4.0
			   ----------------------------------------------------------*/
					dm = trunc(estm[4] / ex + estm[5]);
					m = (long) dm;
					d2 = dm - .5;
					d2 *= d2;
					d1 = dm + dm;
					for (i = 2; i <= m; ++i) {
						dm -= 1.;
						d1 -= 2.;
						d2 -= d1;
						ratio = (d3 + d2) / (twox + d1 - ratio);
						blpha = (ratio + ratio * blpha) / dm;
					}
					bk1 = 1. / ((M_SQRT_2dPI + M_SQRT_2dPI * blpha) * sqrt(ex));
					if (ize == 1)
						bk1 *= exp(-ex);
					wminf = estf[4] * (ex - abs(ex - estf[6])) + estf[5];
				}
				/* ---------------------------------------------------------
		       Calculation of K(ALPHA+1,X)
		       from K(ALPHA,X) and  K(ALPHA+1,X)/K(ALPHA,X)
		       --------------------------------------------------------- */
				bk2 = bk1 + bk1 * (nu + .5 - ratio) / ex;
			}
			/*--------------------------------------------------------------------
		  Calculation of 'NCALC', K(ALPHA+I,X),	I  =  0, 1, ... , NCALC-1,
		  &	  K(ALPHA+I,X)/K(ALPHA+I-1,X),	I = NCALC, NCALC+1, ... , NB-1
		  -------------------------------------------------------------------*/
			ncalc = nb;
			bk[0] = bk1;
			if (iend == 0)
				return ncalc;

			j = 1 - k;
			if (j >= 0)
				bk[j] = bk2;

			if (iend == 1)
				return ncalc;

			m = min((long) (wminf - nu),iend);
			for (i = 2; i <= m; ++i) {
				t1 = bk1;
				bk1 = bk2;
				twonu += 2.;
				if (ex < 1.) {
					if (bk1 >= Double.MAX_VALUE / twonu * ex)
						break;
				} else {
					if (bk1 / ex >= Double.MAX_VALUE / twonu)
						break;
				}
				bk2 = twonu / ex * bk1 + t1;
				ii = i;
				++j;
				if (j >= 0) {
					bk[j] = bk2;
				}
			}

			m = ii;
			if (m == iend) {
				return ncalc;
			}
			ratio = bk2 / bk1;
			mplus1 = m + 1;
			ncalc = -1;
			for (i = (int) mplus1; i <= iend; ++i) {
				twonu += 2.;
				ratio = twonu / ex + 1./ratio;
				++j;
				if (j >= 1) {
					bk[j] = ratio;
				} else {
					if (bk2 >= Double.MAX_VALUE / ratio)
						return ncalc;

					bk2 *= ratio;
				}
			}
			ncalc = (int) max(1, mplus1 - k);
			if (ncalc == 1)
				bk[0] = bk2;
			if (nb == 1)
				return ncalc;

			//L420:
				for (i = ncalc; i < nb; ++i) { /* i == *ncalc */
					//#ifndef IEEE_754
					//if (bk[i-1] >= DBL_MAX / bk[i])
					//	return;
					//#endif
					bk[i] *= bk[i-1];
					(ncalc)++;
				}
		}
		return ncalc;
	}
}
