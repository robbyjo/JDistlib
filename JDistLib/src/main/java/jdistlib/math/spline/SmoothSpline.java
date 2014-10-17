/*
 * Roby Joehanes
 * 
 * Copyright 2007 Roby Joehanes
 * This file is distributed under the GNU General Public License version 3.0.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jdistlib.math.spline;

import static java.lang.Math.*;
import static java.util.Arrays.fill;
import static jdistlib.math.LinPack.dpbfa;
import static jdistlib.math.LinPack.dpbsl;
import static jdistlib.math.Constants.kInvGoldRatio;
import static jdistlib.math.Constants.M_LN2;
import static jdistlib.math.spline.SmoothSplineCriterion.*;

/**
 * <P>This class deals with smoothing of cubic B-Splines. The algorithm is explained
 * in the following paper:
 * 
 * <P>Silverman, B. W. (1985) Some aspects of the spline smoothing approach to
 * non-parametric regression curve-fitting. Journal of the Royal Statistical Society.
 * Series B (Methodological), 47(1): 1--52.<br>
 * JSTOR link: <a href="http://www.jstor.org/stable/2345542">http://www.jstor.org/stable/2345542</a>
 * 
 * <P>And the comments of Finbarr O'Sullivan at page 39--40 of the paper above.
 * 
 * <P>The code is in FORTRAN 77 and it is readily available in Netlib:<br>
 * <a href="http://www.netlib.org/gcv/sbart.f">sbart.f</a> in <a href="http://www.netlib.org/gcv/">GCV</a>
 * package.
 * 
 * <P>This code uses the idea explained in the following paper:
 * 
 * <P>de Boor, C. (1977) Package for calculating with B-Splines, SIAM Journal on
 * Numerical Analysis, 14(3): 441--472.<br>
 * JSTOR link: <a href="http://www.jstor.org/stable/2156696">http://www.jstor.org/stable/2156696</a>
 * 
 * <P>De Boor's code is also in FORTRAN 77 and it is also readily available in Netlib as
 * <a href="http://www.netlib.org/pppack/">PPPack</a> package.
 * 
 * <P>De Boor uses LINPACK for Cholesky decomposition and solver for band matrices (dpbfa and dpbsl).
 * LINPACK is also available in Netlib here:<br>
 * <a href="http://www.netlib.org/linpack/">http://www.netlib.org/linpack/</a>
 * 
 * <P>All routines above are available as public domain. They are also included in
 * <a href="http://www.r-project.org">R</a>, a popular statistical framework. In R, some
 * routines are translated into C and hand-polished. The codes included in R is in GPL.
 * My translation is also in GPL.
 * 
 * <P>Relevant methods:
 * <ul>
 * <li>fit = Essentially a call akin to R's <tt>smooth.spline</tt>. Returns a
 * <tt>SmoothSplineResult</tt> object that has all outputs you need for prediction.</li>
 * <li>predict = Essentially like R's <tt>predict.smooth.spline</tt>. Returns
 * the predicted value of the B-Spline</li>
 * </ul>
 * 
 * <P>Here's what's different here from the original code and/or R:
 * <ul>
 * <li>In the original codes, some of the routines are in single precision (real). In R,
 * they are already converted to double precision. The ones in here are all in double-precision.</li>
 * <li>The original codes and R's code contains a questionable constant value, namely 0.3330
 * (in sgram). The original paper states that it should be 1/3. So, I used that instead. (See kAThird)
 * Inevitably, the output of the original code and R loses some precision. Note that
 * the output starts to differ only at the fourth or fifth decimal place.</li>
 * <li>Removed superfluous parameters. FORTRAN 77 doesn't allow dynamic arrays. So, all
 * intermediary / buffer arrays have to be passed from the caller. This leads to code
 * ugliness. I created arrays just where needed without seriously affecting the
 * performance. Also, array lengths / dimensions are no longer passed here since Java allows
 * <tt>.length</tt> for all arrays.</li>
 * <li>Some matrices are better put as double-dimension arrays (most notably sg0, sg1, sg2, sg3,
 * which presents Sigma matrix, and hs0, hs1, hs2, hs3, which represents X'WX matrix)</li>
 * <li>Some loop unrolling is reversed (most notably in sgram). Some of the loops are unrolled
 * to avoid costly if() statements (e.g. in sinerp). Some are left intact (e.g. in stxwx)</li>
 * <li>I changed the derivative array to ROW major. FORTRAN uses column major array convention
 * and it's a pain to pass the array around.</li>
 * <li>Variable / parameter / method renaming. FORTRAN 77's harsh 6-character names no longer
 * applies in Java. So, I put better names around. But I keep the original comments intact,
 * unless it's either superfluous or no longer applies.</li>
 * <li>I uses R's default for epsilon, tolerance, the lower- and upper-bounds for smoothing
 * parameter search, and the maximum number of iterations.</li> 
 * </ul>
 * 
 * <P>Here's the gotchas:
 * <ul>
 * <li>I have tested this routine quite a lot, but far from extensively. I think it should
 * be relatively free of bugs / errors. But, let me know if you do find some discrepancies.</li>
 * <li>Also, despite the testing, some branches of the code have never gotten tested. I put some
 * markers at the code. I've never encountered errors so far.</li>
 * <li>Note that the original comment is still intact. The call parameters are different
 * now since some have been deleted.</li>
 * </ul>
 * 
 * @author Roby Joehanes
 *
 */
public class SmoothSpline
{
	public static final double
		kDefaultTolerance = 1e-4, // Taken from R
		kDefaultEpsilon = 2e-8, // Taken from R
		kDefaultSmoothingParamLowerBound = -1.5, // Taken from R
		kDefaultSmoothingParamUpperBound = 1.5; // Taken from R
	public static final int kDefaultMaxNumIterations = 500; // Taken from R
	private static final double
		kLog2Of1_4 = log(1.4) / M_LN2,
		kLog2Of1_42 = log(10.0/7) / M_LN2,
		kstxwx_eps = 1e-10, // I don't know why O'Sullivan used different epsilon values?
		ksslvrg_eps = 1e-11,
		kBigValue = 1e100,
		kAThird = 1.0/3.0; // The original code and R use 0.3330f instead
	private static final int kInsertionTreshold = 4;

	private static final int calcNumInnerKnots(int n)
	{
		if (n < 50)
			return n;
		if (n < 200)
			return (int) floor(50 * pow(2, (n-50)/150.0));
		if (n < 800)
			return (int) floor(100 * pow(2, kLog2Of1_4 * (n-200)/600.0) );
		if (n < 3200)
			return (int) floor(140 * pow(2, kLog2Of1_42 * (n-800)/2400.0) );
		return (int) floor(200 + pow(n - 3200, 0.2));
	}

	/**
	 * Form a knot. Original name: sknotl
	 * @param x
	 * @return
	 */
	private static final double[] formKnots(double[] x)
	{
		int
			n = x.length,
			nk = calcNumInnerKnots(n),
			nMin1 = n - 1,
			nkMin1 = nk - 1;
		double[] result = new double[nk + 6];
		result[0] = result[1] = result[2] = x[0];
		result[nk+3] = result[nk+4] = result[nk+5] = x[nMin1];
	
		for (int i = 0; i < nk; i++)
			result[i + 3] = x[(i * nMin1) / nkMin1];
		return result;
	}
	
	/**
	 * A Cubic B-spline Smoothing routine.
	 * 
	 * @author Finbarr O'Sullivan
	 * 
	 * <P>Ported from Fortran by Roby Joehanes. Original routine name: sbart
	 * 
	 * <P>The algorithm minimises:<br><br>
	 * 
	 * (1/n) * sum ws(i)^2 * (ys(i)-sz(i))^2 + lambda* int ( s"(x) )^2 dx<br><br>
	 * 
	 * lambda is a function of the spar which is assumed to be between 0 and 1
	 * <pre>
	 * INPUT
	 * -----
	 * penalt	A penalty > 1 to be used in the gcv criterion
	 * dofoff	either `df.offset' for GCV or `df' (to be matched).
	 * n		number of data points
	 * ys(n)	vector of length n containing the observations
	 * ws(n)	vector containing the weights given to each data point
	 *          NB: the code alters the values here.
	 * xs(n)	vector containing the ordinates of the observations
	 * ssw		`centered weighted sum of y^2'
	 * nk		number of b-spline coefficients to be estimated nk <= n+2
	 * knot(nk+4)	vector of knot points defining the cubic b-spline basis.
	 * 				To obtain full cubic smoothing splines one might
	 * 				have (provided the xs-values are strictly increasing)
	 * spar		penalised likelihood smoothing parameter
	 * ispar	indicating if spar is supplied (ispar=1) or to be estimated
	 * lspar, uspar lower and upper values for spar search;  0.,1. are good values
	 * tol, eps	used in Golden Search routine
	 * isetup	setup indicator [initially 0
	 * icrit	indicator saying which cross validation score is to be computed
	 * 			0: none ;  1: GCV ;  2: CV ;  3: 'df matching'
	 * ld4		the leading dimension of abd (ie ld4=4)
	 * ldnk		the leading dimension of p2ip (not referenced)
	 * 
	 * 
	 * OUTPUT
	 * ------
	 * coef(nk)	vector of spline coefficients
	 * sz(n)	vector of smoothed z-values
	 * lev(n)	vector of leverages
	 * crit		either ordinary or generalized CV score
	 * spar         if ispar != 1
	 * lspar         == lambda (a function of spar and the design)
	 * iter		number of iterations needed for spar search (if ispar != 1)
	 * ier		error indicator
	 * 			ier = 0 ___  everything fine
	 * 			ier = 1 ___  spar too small or too big problem in cholesky decomposition
	 * 
	 * Working arrays/matrix
	 * xwy			X'Wy
	 * hs0,hs1,hs2,hs3	the diagonals of the X'WX matrix
	 * sg0,sg1,sg2,sg3	the diagonals of the Gram matrix SIGMA
	 * abd (ld4,nk)		[ X'WX + lambda*SIGMA ] in diagonal form
	 * p1ip(ld4,nk)		inner products between columns of L inverse
	 * p2ip(ldnk,nk)	all inner products between columns of L inverse where  L'L = [X'WX + lambda*SIGMA]  NOT REFERENCED
	 */
	private static final SmoothSplineResult fitCubicSpline (double penalty, double dfOffset,
			double[] xs, double[] ys, double[] ws, double ssw, double[] knots,
			SmoothSplineCriterion criterion, double spar, double lspar,
			double uspar, double tol, double eps, int maxIteration)
	{
		int
			n = xs.length,
			nk = knots.length - 4;
		boolean hasError = false;
		assert (n == ys.length && n == ws.length); // sanity check

		SmoothSplineResult result = new SmoothSplineResult();
		result.mCoefficients = new double[nk];
		result.mSmoothedValues = new double[n];
		result.mLeverage = new double[n];
		result.mEstimatedDF = dfOffset;
		result.mCriterion = criterion;
		result.mKnots = knots;
		double
			ratio = 1,
			sigma[][] = null,
			xwy[] = null,
			XtWX[][] = null;
		// Compute SIGMA, X' W X, X' W z, trace ratio, s0, s1.
		// SIGMA	-> sg0,sg1,sg2,sg3
		// X' W X	-> hs0,hs1,hs2,hs3
		// X' W Z	-> xwy
		// trevor fixed this 4/19/88
		// Note: sbart, i.e. stxwx() and sslvrg() {mostly, not always!}, use
		// 	 the square of the weights; the following rectifies that
		double[] newWs = new double[n];
		for (int i = 0; i < n; ++i)
			if (ws[i] > 0.)
				newWs[i] = sqrt(ws[i]);

		// SIGMA[i,j] := Int  B''(i,t) B''(j,t) dt  {B(k,.) = k-th B-spline}
		sigma = calculateSigma(knots);
		XtWX = calculateXtWX(xs, ys, newWs, knots);
		xwy = XtWX[4];
		double[][] newHs = new double[4][];
		for (int i = 0; i < 4; i++)
			newHs[i] = XtWX[i];
		XtWX = newHs;
		{
			// Compute ratio :=  tr(X' W X) / tr(SIGMA)
			double
				t1 = 0,
				t2 = 0;
			for (int i = 3 - 1; i < (nk - 3); ++i)
			{
				t1 += XtWX[0][i];
				t2 += sigma[0][i];
			}
			ratio = t1 / t2;
		}

		boolean isSparSupplied = !(Double.isNaN(spar) || Double.isInfinite(spar));
		if (isSparSupplied) { // Value of spar supplied 
			result.mSmoothingParameter = spar;
			result.mLambda = ratio * pow(16, result.mSmoothingParameter * 6 - 2);
			result.mHasFactorizationProblems = calculateCVScore(result, penalty, xs, ys, newWs, ssw, knots, criterion, xwy, XtWX, sigma);
			return result;
		}

		// ELSE ---- spar not supplied --> compute it !
		result.mIterNo = 0;

		double
			a = lspar,
			b = uspar,
			v = a + kInvGoldRatio * (b - a),
			w = v,
			x = v,
			e = 0;
		result.mSmoothingParameter = x;
		result.mLambda = ratio * pow(16, result.mSmoothingParameter * 6 - 2);
		result.mHasFactorizationProblems = calculateCVScore(result, penalty, xs, ys, newWs, ssw, knots, criterion, xwy, XtWX, sigma);

		double
			fx = result.mFitCVScore,
			fv = fx,
			fw = fx,
			d = 0;
		//boolean
		//	Fparabol = false,
		//	tracing = ispar < 0;
		while (!hasError)
		{
			double
				xm = (a + b) * .5,
				tol1 = eps * abs(x) + tol / 3.0,
				tol2 = tol1 * 2.0;
			++result.mIterNo;
			if (abs(x - xm) <= tol2 - (b - a) * .5 || result.mIterNo > maxIteration)
			    break;

			// is golden-section necessary?
			boolean isGoldenSect = abs(e) <= tol1 ||
				// if had Inf then go to golden-section
				fx >= kBigValue || fv >= kBigValue || fw >= kBigValue;

			// Fit Parabola
			if (!isGoldenSect)
			{
				double
					r = (x - w) * (fx - fv),
					q = (x - v) * (fx - fw),
					p = (x - v) * q - (x - w) * r;
				q = (q - r) * 2.;
				if (q > 0.)
				    p = -p;
				q = abs(q);
				r = e;
				e = d;
	
				// is parabola acceptable?  Otherwise do golden-section
				isGoldenSect = abs(p) >= abs(.5 * q * r) || q == 0.
					|| p <= q * (a - x) || p >= q * (b - x);

				// Parabolic Interpolation step
				if (!isGoldenSect)
				{
					d = p / q;
					double u = x + d;

					// f must not be evaluated too close to ax or bx
					if (u - a < tol2 || b - u < tol2)
						d = xm >= x ? abs(tol1) : - abs(tol1);
				}
			}

			// a golden-section step
			if (isGoldenSect)
			{
				e = x >= xm ? a - x : b - x;
				d = kInvGoldRatio * e;
			}

			// f must not be evaluated too close to x
			double u = x + ((abs(d) >= tol1) ? d : (d >= 0 ? abs(tol1) : -abs(tol1)));

			result.mSmoothingParameter = u;
			result.mLambda = ratio * pow(16, result.mSmoothingParameter * 6 - 2);
			hasError = calculateCVScore(result, penalty, xs, ys, newWs, ssw, knots, criterion, xwy, XtWX, sigma);
			double fu = result.mFitCVScore;

			//  update  a, b, v, w, and x
			if (fu <= fx) {
			    if (u >= x) a = x; else b = x;

			    v = w; fv = fw;
			    w = x; fw = fx;
			    x = u; fx = fu;
			}
			else {
			    if (u < x)  a = u; else b = u;
			    if (fu <= fw || w == x) {
			    	v = w; fv = fw;
			    	w = u; fw = fu;
			    } else if (fu <= fv || v == x || v == w) {
			    	v = u; fv = fu;
			    }
			}
		}
		result.mSmoothingParameter = x;
		result.mFitCVScore = fx;
		result.mHasFactorizationProblems = hasError;
		double sum = 0;
		for (int i = 0; i < n; i++)
			sum += result.mLeverage[i];
		result.mEstimatedDF = sum;
		sum = 0;
		for (int i = 0; i < n; i++)
		{
			double temp = ys[i] - result.mSmoothedValues[i];
			sum += ws[i] * temp * temp;
		}
		result.mPenalizedCriterion = sum;
		return result;
	}
	
	/**
	 * <P>Calculation of the cubic B-spline smoothness prior for "usual" interior knot setup.
	 * <P>sgm[0-3](nb)    Symmetric matrix whose (i,j)'th element contains the integral of
	 * B"(i,.) B"(j,.) , i=1,2 ... nb and j=i,...nb.<BR>
	 * Only the upper four diagonals are computed.
	 * 
	 * <P>Original subroutine name: sgram
	 * 
	 * @param tb
	 */
	private static final double[][] calculateSigma(double[] tb)
	{
		int
			nb = tb.length - 4,
			ileft = 0;
		double
			sg[][] = new double[4][nb],
			yw1[] = new double[4],
			yw2[] = new double[4],
			wpt,
			vnikx[][] = new double[3][4];

		for (int i = 0; i < nb; ++i)
		{
			// Calculate a linear approximation to the
			// second derivative of the non-zero B-splines
			// over the interval [tb(i),tb(i+1)].
			ileft = determineInterval(tb, tb[i], ileft, null);

			// Left end second derivatives
			calculateBSplineDerivatives(tb, tb[i], ileft, vnikx);

			// Put values into yw1
			for (int ii = 0; ii < 4; ++ii)
				yw1[ii] = vnikx[2][ii];

			// Right end second derivatives
			calculateBSplineDerivatives(tb, tb[i + 1], ileft, vnikx);

			ileft++;
			// Slope*(length of interval) in Linear Approximation to B"
			for (int ii = 0; ii < 4; ++ii)
				yw2[ii] = vnikx[2][ii] - yw1[ii];

			wpt = tb[i + 1] - tb[i];
			// Calculate Contributions to the sigma vectors
			int maxIdx = min(ileft, 4);
			for (int ii = 0; ii < maxIdx; ++ii)
				for (int jj = ii; jj < maxIdx; ++jj)
					sg[jj-ii][ileft + ii - maxIdx] += wpt * (yw1[ii] * yw1[jj] +
						(yw2[ii] * yw1[jj] + yw2[jj] * yw1[ii]) * 0.5 + yw2[ii] * yw2[jj] * kAThird);
		}
		return sg;
	}

	/**
	 * Calculate X'WX and X'WY. Original routine name: stxwx
	 * @param x
	 * @param y
	 * @param w
	 * @param knots
	 * @return
	 */
	private static final double[][] calculateXtWX(double[] x, double[] y, double[] w,  double[] knots)
	{
		int
			k = x.length,
			n = knots.length - 4,
			ileft;
	
		// Local variables
		int mflag[] = new int[1];
		double
			vnikx[] = new double[4],
			dvnikx[][] = new double[][] { vnikx },
			hs0[] = new double[n],
			hs1[] = new double[n],
			hs2[] = new double[n],
			hs3[] = new double[n],
			xwy[] = new double[n],
			retval[][] = new double[][] { hs0, hs1, hs2, hs3, xwy };
	
		// Compute X' W^2 X -> hs0,hs1,hs2,hs3  and X' W^2 Z -> y
		// Note that here the weights w(i) == sqrt(wt[i])  where wt[] where original weights
		ileft = 0;
		for (int i = 0; i < k; ++i) {
			ileft = determineInterval(knots, x[i], ileft, mflag);
			if (mflag[0] == 1) {
				if (x[i] <= knots[ileft] + kstxwx_eps)
					ileft = n - 1;
				else
					return retval;
			}
			calculateBSplineDerivatives(knots, x[i], ileft, dvnikx);
			int j = ileft - 3;
			double
				w_i = w[i],
				z_i = y[i],
				vnikx_i = vnikx[0],
				temp = w_i * w_i * vnikx_i;
			xwy[j] += temp * z_i;
			hs0[j] += temp * vnikx_i;
			hs1[j] += temp * vnikx[1];
			hs2[j] +=temp * vnikx[2];
			hs3[j] += temp * vnikx[3];
			j++;
			vnikx_i = vnikx[1];
			temp = w_i * w_i * vnikx_i;
			xwy[j] += temp * z_i;
			hs0[j] += temp * vnikx_i;
			hs1[j] += temp * vnikx[2];
			hs2[j] += temp * vnikx[3];
			j++;
			vnikx_i = vnikx[2];
			temp = w_i * w_i * vnikx_i;
			xwy[j] += temp * z_i;
			hs0[j] += temp * vnikx_i;
			hs1[j] += temp * vnikx[3];
			j++;
			vnikx_i = vnikx[3];
			temp = w_i * w_i * vnikx_i;
			xwy[j] += temp * z_i;
			hs0[j] += temp * vnikx_i;
		}
		return retval;
	}
	
	/**
	 * Purpose :  Computes Inner Products between columns of L^{-1}
	 * where L = abd is a Banded Matrix with 3 subdiagonals
	 * 
	 * <P>The algorithm works in two passes:<br>
	 * Pass 1 computes (cj,ck) k=j,j-1,j-2,j-3 ;  j=nk, .. 1<br>
	 * Pass 2 computes (cj,ck) k <= j-4  (If flag == 1 ).<br>
	 * A refinement of Elden's trick is used.
	 * 
	 * <P>Original routine name: sinerp. Pass 2 is unused. So, I took it out
	 * 
	 * @param abd Band matrix
	 * @return the inner product matrix
	 */
	private static final double[][] calculateInnerProduct(double[][] abd)
	{
		int
			nk = abd[0].length,
			i = nk - 1;
		double
			p1ip[][] = new double[abd.length][nk],
			c0 = 1.0 / abd[3][i],
			c1,c2,c3,
			wjm1 = c0 * c0,
			wjm21, wjm22, wjm31, wjm32, wjm33;

		p1ip[3][i] = wjm1;
		i--;
		c0 = 1.0 / abd[3][i];
		c3 = -abd[2][i+1] * c0;
		wjm21 = wjm1;
		wjm22 = p1ip[2][i] = c3 * wjm1;
		wjm1 = p1ip[3][i] = c0*c0 + c3*c3 * wjm1;
		i--;
		c0 = 1.0 / abd[3][i];
		c2 = -abd[1][i+2] * c0;
		c3 = -abd[2][i+1] * c0;
		wjm31 = wjm21;
		wjm32 = wjm22;
		wjm33 = p1ip[1][i] = c2 * wjm21 + c3 * wjm22;
		p1ip[2][i] = c2 * wjm22 + c3 * wjm1;
		p1ip[3][i] = c0*c0 + c2*c2 * wjm21 + 2*c2*c3 * wjm22 + c3*c3 * wjm1;
		wjm21 = wjm1;
		wjm22 = p1ip[2][i];
		wjm1 = p1ip[3][i];
		for (--i; i >= 0; i--)
		{
			c0 = 1.0 / abd[3][i];
			c1 = -abd[0][i+3] * c0;
			c2 = -abd[1][i+2] * c0;
			c3 = -abd[2][i+1] * c0;
			p1ip[0][i] = c1 * wjm31 + c2 * wjm32 + c3 * wjm33;
			p1ip[1][i] = c1 * wjm32 + c2 * wjm21 + c3 * wjm22;
			p1ip[2][i] = c1 * wjm33 + c2 * wjm22 + c3 * wjm1;
			p1ip[3][i] = c0*c0 + c1*c1 * wjm31 + 2*c1*c2 * wjm32
				+ 2*c1*c3 * wjm33 + c2*c2 * wjm21 + 2*c2*c3 * wjm22 + c3*c3 * wjm1;
			wjm31 = wjm21;
			wjm32 = wjm22;
			wjm33 = p1ip[1][i];
			wjm21 = wjm1;
			wjm22 = p1ip[2][i];
			wjm1 = p1ip[3][i];
		}
		return p1ip;
	}

	/**
	 * Estimate cross validation score
	 * 
	 * <P> Original routine name: sslvrg
	 * @param result
	 * @param penalty
	 * @param x
	 * @param y
	 * @param w
	 * @param ssw
	 * @param knot
	 * @param criterion
	 * @param xwy
	 * @param hs
	 * @param sg
	 * @return
	 */
	private static final boolean calculateCVScore(SmoothSplineResult result, double penalty, double[] x,
		double[] y, double[] w, double ssw, double[] knot, SmoothSplineCriterion criterion,
		double[] xwy, double[][] hs, double[][] sg)
	{
		// Local variables
		double
			temp,
			coef[] = result.mCoefficients,
			sz[] = result.mSmoothedValues,
			lev[] = result.mLeverage;

		int
			n = x.length,
			nk = knot.length - 4,
			mflag[] = new int[1];
		double
			dvnikx[][] = new double[1][4],
			vnikx[] = dvnikx[0],
			abd[][] = new double[4][nk];

		// Purpose :
		//       Compute smoothing spline for smoothing parameter lambda
		//       and compute one of three `criteria' (OCV , GCV , "df match").
		// See comments in ./sbart.f from which this is called

		int ileft = 0;
		if (hs != null && sg != null)
		{
			// compute the coefficients coef() of estimated smooth
			int
				nkMin1 = nk - 1,
				nkMin2 = nk - 2,
				nkMin3 = nk - 3;
			double lambda = result.mLambda;
			System.arraycopy(xwy, 0, coef, 0, nk);
			abd[3][nkMin1] = hs[0][nkMin1] + lambda * sg[0][nkMin1];
			abd[3][nkMin2] = hs[0][nkMin2] + lambda * sg[0][nkMin2];
			abd[3][nkMin3] = hs[0][nkMin3] + lambda * sg[0][nkMin3];
			abd[2][nkMin1] = hs[1][nkMin2] + lambda * sg[1][nkMin2];
			abd[2][nkMin2] = hs[1][nkMin3] + lambda * sg[1][nkMin3];
			abd[1][nkMin1] = hs[2][nkMin3] + lambda * sg[2][nkMin3];
			for (int i = 0; i < nkMin3; ++i)
				for (int j = 0; j < 4; ++j)
					abd[3-j][i+j] = hs[j][i] + lambda * sg[j][i];
			// factorize banded matrix abd:
			int retVal = dpbfa(abd, 3);
			if (retVal != 0)
				return true;
			// solve linear system (from factorize abd):
			dpbsl(abd, 3, coef);
			// Value of smooth at the data points
		}
		for (int i = 0; i < n; ++i)
			// bvalue_(double[] t, double[] bcoef, double x, int jderiv)
			// bvalue(t,lent,bcoef,n,k,x,jderiv)
			// bvalue(knot,lenkno,coef, nk,4,xv,0)
			sz[i] = evaluateBSplineDerivatives(knot, coef, x[i], 0);
		// Compute the criterion function if requested
		if (criterion == NO_CRITERION)
			return false;
		// --- Ordinary or Generalized CV or "df match" ---
		//     Get Leverages First
		double[][] p1ip = calculateInnerProduct(abd);
		for (int i = 0; i < n; ++i) {
			double xv = x[i];
			ileft = determineInterval(knot, xv, ileft, mflag);
			if (mflag[0] == -1) {
				ileft = 3;
				xv = knot[3] + ksslvrg_eps;
			} else if (mflag[0] == 1) {
				ileft = nk - 1;
				xv = knot[nk] - ksslvrg_eps;
			}
			calculateBSplineDerivatives(knot, xv, ileft, dvnikx);
			int
				j = ileft - 3,
				jp1 = j+1,
				jp2 = j+2;
			double
				b0 = vnikx[0],
				b1 = vnikx[1],
				b2 = vnikx[2],
				b3 = vnikx[3],
				wi = w[i];
			lev[i] = (p1ip[3][j] * b0 * b0 + 2 * p1ip[2][j] * b0 * b1
					+ 2 * p1ip[1][j] * b0 * b2 + 2 * p1ip[0][j] * b0 * b3
					+ p1ip[3][jp1] * b1 * b1 + 2 * p1ip[2][jp1] * b1 * b2
					+ 2 * p1ip[1][jp1] * b1 * b3 + p1ip[3][jp2] * b2 * b2
					+ 2 * p1ip[2][jp2] * b2 * b3 + p1ip[3][ileft] * b3 * b3) * wi * wi; 
		}

		// Evaluate Criterion
		switch (criterion)
		{
			case GCV: // Generalized CV
			{
				double
					rss = ssw,
					df = 0,
					sumw = 0.;
				// w(i) are sqrt( wt[i] ) weights scaled such
				// that sumw =  number of observations with w(i) > 0
				for (int i = 0; i < n; ++i) {
					temp = (y[i] - sz[i]) * w[i];
					rss += temp * temp;
					df += lev[i];
					temp = w[i];
					sumw += temp * temp;
					//System.out.println((i+1) + " ==> " + y[i] + sCm + rss + sCm + df + sCm + sumw);
				}
				temp = 1 - (result.mEstimatedDF + penalty * df) / sumw;
				result.mFitCVScore = (rss / sumw) / (temp * temp);
			}
			break;
			case CV: // Ordinary CV
			{
				double sum = 0;
				for (int i = 0; i < n; ++i) {
					temp = (y[i] - sz[i]) * w[i] / (1 - lev[i]);
					sum += temp * temp;
				}
				result.mFitCVScore = sum / n;
			}
			break;
			case DF_MATCH: // df matching
			{
				double sum = 0;
				for (int i = 0; i < n; ++i)
					sum += lev[i];
				temp = result.mEstimatedDF - sum;
				result.mFitCVScore = temp * temp + 3;
			}
			break;
			default:
		}
		return false;
	}

	/**
	 * Calculates value and deriv.s of all b-splines which do not vanish at x<br>
	 * calls bsplvb
	 * 
	 * <P>Ripped from PPPACK. Original routine name = bsplvd
	 * 
	 * <pre>
	 * ******  i n p u t  ******
	 *  t     the knot array, of length left+k (at least)
	 *  k     the order of the b-splines to be evaluated
	 *  x     the point at which these values are sought
	 *  left  an int indicating the left endpoint of the interval of
	 *        interest. the  k  b-splines whose support contains the interval
	 *               (t(left), t(left+1))
	 *        are to be considered.
	 *  a s s u m p t i o n  - - -  it is assumed that
	 *               t(left) < t(left+1)
	 *        division by zero will result otherwise (in  b s p l v b ).
	 *        also, the output is as advertised only if
	 *               t(left) <= x <= t(left+1) .
	 *  nderiv   an int indicating that values of b-splines and their
	 *        derivatives up to but not including the  nderiv-th  are asked
	 *        for. ( nderiv  is replaced internally by the int in (1,k)
	 *        closest to it.)
	 *
	 * ******  w o r k   a r e a  ******
	 *  a     an array of order (k,k), to contain b-coeff.s of the derivat-
	 *        ives of a certain order of the  k  b-splines of interest.
	 * 
	 * ******  o u t p u t  ******
	 *  dbiatx   an array of order (k,nderiv). its entry  (i,m)  contains
	 *        value of  (m-1)st  derivative of  (left-k+i)-th  b-spline of
	 *        order  k  for knot sequence  t , i=m,...,k; m=1,...,nderiv.
	 * 
	 * ******  m e t h o d  ******
	 *  values at  x  of all the relevant b-splines of order k,k-1,...,
	 *  k+1-nderiv  are generated via  bsplvb  and stored temporarily
	 *  in  dbiatx .  then, the b-coeffs of the required derivatives of the
	 *  b-splines of interest are generated by differencing, each from the
	 *  preceding one of lower order, and combined with the values of b-
	 *  splines of corresponding order in  dbiatx  to produce the desired
	 *  values.
	 * </pre>
	 */
	private static final void calculateBSplineDerivatives(double[] t, double x, int left, double[][] dbiatx)
	{
		// dbiatx is nderiv x k instead of k x nderiv
		int
			k = dbiatx[0].length,
			nderiv = dbiatx.length;

		int mhigh = max(min(nderiv,k),1); // mhigh is usually equal to nderiv.
		calculateBSplineInitialDerivatives(t, k - mhigh, true, x, left, dbiatx[0]);
		if (mhigh == 1)
			return;
		// the first column of  dbiatx  always contains the b-spline values
		// for the current order. these are stored in column k+1-current
		// order  before  bsplvb  is called to put values for the next
		// higher order on top of it.
		int ideriv = mhigh;
		for (int m = 2; m <= mhigh; ++m) {
			for (int j = ideriv, jp1mid = 0; j <= k; ++j, ++jp1mid)
				dbiatx[ideriv-1][j-1] = dbiatx[0][jp1mid];
			--ideriv;
			calculateBSplineInitialDerivatives(t, k - ideriv, false, x, left, dbiatx[0]);
		}
	
		// at this point,  b(left-k+i, k+1-j)(x) is in  dbiatx(i,j) for
		// i=j,...,k and j=1,...,mhigh ('=' nderiv). in particular, the
		// first column of  dbiatx  is already in final form. to obtain cor-
		// responding derivatives of b-splines in subsequent columns, gene-
		// rate their b-repr. by differencing, then evaluate at  x.
	
		double a[][] = new double[k][k];
		for (int i = 0; i < k; ++i)
			a[i][i] = 1;
		// at this point, a(.,j) contains the b-coeffs for the j-th of the
		// k  b-splines of interest here.
	
		int kp1 = k + 1;
		for (int m = 2; m <= mhigh; ++m) {
			int kp1mm = kp1 - m;
	
			// for j=1,...,k, construct b-coeffs of  (m-1)st  derivative of
			// b-splines from those for preceding derivative by differencing
			// and store again in  a(.,j) . the fact that  a(i,j) = 0  for
			// i < j  is used.sed.
			for (int ldummy = 1, il = left+1, j = k; ldummy <= kp1mm; ++ldummy, --il, --j) {
				double factor = kp1mm / (t[il + kp1mm - 1] - t[il - 1]);
				// the assumption that t(left) < t(left+1) makes denominator in  factor  nonzero.
				for (int l = 0; l < j; ++l)
					a[l][j-1] = (a[l][j-1] - a[l][j-2]) * factor;
			}
	
			// for i=1,...,k, combine b-coeffs a(.,i) with b-spline values
			// stored in dbiatx(.,m) to get value of  (m-1)st  derivative of
			// i-th b-spline (of interest here) at  x , and store in
			// dbiatx(i,m). storage of this value over the value of a b-spline
			// of order m there is safe since the remaining b-spline derivative
			// of the same order do not use this value due to the fact that  a(j,i) = 0  for j < i.
			for (int i = 1; i <= k; ++i) {
				double sum = 0;
				for (int j = max(i,m); j <= k; ++j)
					sum += a[i-1][j-1] * dbiatx[m-1][j-1];
				dbiatx[m-1][i-1] = sum;
			}
		}
	}
	
	/**
	 * Calculates the value of all possibly nonzero b-splines at  x  of order
	 * jout  =  dmax( jhigh , (j+1)*(index-1) ) with knot sequence  t .
	 * 
	 * <P>Ripped from PPPACK. Original name: bsplvb
	 * 
	 * <pre>
	 * ******  i n p u t  ******
	 * t.....knot sequence, of length  left + jout  , assumed to be nondecreasing.
	 * a s s u m p t i o n  :  t(left)  <  t(left + 1)
	 * d i v i s i o n  b y  z e r o  will result if  t(left) = t(left+1)
	 * 
	 * jhigh, index.....ints which determine the order  jout = max(jhigh, (j+1)*(index-1))
	 *        of the b-splines whose values at  x  are to be returned.
	 *        index  is used to avoid recalculations when several columns of
	 *        the triangular array of b-spline values are needed
	 *        (e.g., in  bvalue  or in  bsplvd ). precisely,
	 *
	 *        if  index = 1 ,
	 *        the calculation starts from scratch and the entire triangular
	 *        array of b-spline values of orders 1,2,...,jhigh  is generated
	 *        order by order , i.e., column by column .
	 *        
	 *        if  index = 2 ,
	 *        only the b-spline values of order  j+1, j+2, ..., jout  are ge-
	 *        nerated, the assumption being that  biatx , j , deltal , deltar
	 *        are, on entry, as they were on exit at the previous call.
	 *         in particular, if  jhigh = 0, then  jout = j+1, i.e., just
	 *        the next column of b-spline values is generated.
	 * 
	 *  w a r n i n g . . .  the restriction   jout <= jmax (= 20)  is
	 *        imposed arbitrarily by the dimension statement for  deltal and
	 *        deltar  below, but is  n o w h e r e  c h e c k e d  for.
	 * 
	 * x.....the point at which the b-splines are to be evaluated.
	 * left.....an int chosen (usually) so that t(left) <= x <= t(left+1).
	 * 
	 * ******  o u t p u t  ******
	 *  biatx.....array of length  jout , with  biatx(i)  containing the value
	 *        at  x  of the polynomial of order  jout  which agrees with
	 *        the b-spline  b(left-jout+i,jout,t)  on the interval (t(left),
	 *        t(left+1)) .
	 * 
	 * ******  m e t h o d  ******
	 *  the recurrence relation
	 *                       x - t(i)               t(i+j+1) - x
	 *     b(i,j+1)(x)  =  ----------- b(i,j)(x) + --------------- b(i+1,j)(x)
	 *                     t(i+j)-t(i)             t(i+j+1)-t(i+1)
	 *  is used (repeatedly) to generate the
	 *  (j+1)-vector  b(left-j,j+1)(x),...,b(left,j+1)(x)
	 *  from the j-vector  b(left-j+1,j)(x),...,b(left,j)(x),
	 *  storing the new values in  biatx  over the old.  the facts that
	 *            b(i,1) = 1         if  t(i) <= x < t(i+1)
	 *  and that
	 *            b(i,j)(x) = 0  unless  t(i) <= x < t(i+j)
	 *  are used. the particular organization of the calculations follows
	 *  algorithm (8)  in chapter x of the text.
	 * Arguments
	 *     dimension     t(left+jout), biatx(jout)
	 *     -----------------------------------
	 * current fortran standard makes it impossible to specify the length of
	 *  t  and of  biatx  precisely without the introduction of otherwise
	 *  superfluous additional arguments.
	 * </pre>
	 */
	private static final void calculateBSplineInitialDerivatives(double[] t, int jhigh, boolean isInit, double x, int left, double[] biatx)
	{
		if (isInit)
		{
			biatx[0] = 1;
			if (jhigh < 1)
				return;
		}
		double
			deltal[] = new double[jhigh + 1],
			deltar[] = new double[jhigh + 1];

		for (int j = 0; j < jhigh; j++)
		{
			deltar[j] = t[left + j + 1] - x;
			deltal[j] = x - t[left - j];
			double saved = 0;
			for (int i = 0; i <= j; ++i) {
				double term = biatx[i] / (deltar[i] + deltal[j - i]);
				biatx[i] = saved + deltar[i] * term;
				saved = deltal[j - i] * term;
			}
			biatx[j+1] = saved;
		}
	}
	
	/**
	 * <P>computes  `left' := max( i ; 1 <= i <= n   &&  xt[i] <= x )  .
	 * 
	 * <P>RJ's note: I ripped this from CMLIB. Original name: interv
	 * 
	 * <pre>
	 *
	 ******  i n p u t  ******
	 *
	 *  xt	numeric vector of length  n , assumed to be nondecreasing
	 *  x	the point whose location with respect to the sequence  xt  is
	 *      to be determined.
	 *  mflag =: all_inside	    {logical} indicating if result should be coerced
	 *		to lie inside {1, n-1}
	 *  ilo   typically the result of the last call to findInterval(.)
	 *        `ilo' used to be a static variable (in Fortran) which is not
	 *	desirable in R anymore (threads!).
	 *	Instead, you *should* use a reasonable value, in the first call.
	 *
	 ******  o u t p u t  ******
	 *
	 *  left, mflag  both integers, whose value is
	 *   0     -1      if            x <  xt[1]
	 *   i      0      if  xt[i]  <= x <  xt[i+1]
	 *   n      1      if  xt[n]  <= x
	 *
	 * in particular,  mflag = 0 is the 'usual' case.  mflag != 0
	 * indicates that  x  lies outside the halfopen interval
	 * xt[1] <= y < xt[n] . the asymmetric treatment of the
	 * interval is due to the decision to make all pp functions cont-
	 * inuous from the right.
	 *
	 *   Note that if all_inside, left is 1 instead of 0 and n-1 instead of n;
	 *   and if rightmost_closed and x == xt[n],  left is    n-1 instead of n.
	 *
	 *
	 ******  m e t h o d  ******
	 *
	 *  the program is designed to be efficient in the common situation that
	 *  it is called repeatedly, with  x  taken from an increasing or decreasing
	 *  sequence. this will happen, e.g., when a pp function is to be graphed.
	 *  The first guess for  left  is therefore taken to be the value returned at
	 *  the previous call and stored in the  l o c a l   variable  ilo .
	 *  a first check ascertains that  ilo < n (this is necessary since the
	 *  present call may have nothing to do with the previous call).
	 *
	 *  then, if  xt[ilo] <= x < xt[ilo+1], we set  left = ilo
	 *  and are done after just three comparisons.
	 *  otherwise, we repeatedly double the difference  istep = ihi - ilo
	 *  while also moving  ilo  and  ihi  in the direction of  x , until
	 *                      xt[ilo] <= x < xt[ihi] ,
	 *  after which we use bisection to get, in addition, ilo+1 = ihi .
	 *  left = ilo  is then returned.
	 *  </pre>
	 */
	private static final int determineInterval(double[] xt, double x, int ilo, int[] mflag)
	{
		int ihi, n = xt.length;
		if (mflag == null)
			mflag = new int[1];

		if(ilo < 0)
		{
			if (x < xt[0])
			{
				mflag[0] = -1;
				return 0; // left_boundary
			}
			ilo = 0;
		}
		ihi = ilo + 1;
		if (ihi >= n) {
			if (x >= xt[n - 1]) // right_boundary
			{
				mflag[0] = 1;
				return n - 1;
			}
			if (n <= 1) // x < xt[1]
			{
				mflag[0] = -1;
				return 0; // left_boundary
			}
			ilo = n - 2;
			ihi = n - 1;
		}

		if (x < xt[ihi])
		{
			if (x >= xt[ilo]) // `lucky': same interval as last time
				return ilo;

			// **** now x < xt[ilo] .	decrease  ilo  to capture  x 
			for(int istep = 1; ; istep *= 2) {
				ihi = ilo;
				ilo = ihi - istep;
				if (ilo <= 0)
				{
					if (x < xt[0])
					{
						mflag[0] = -1;
						return 0;
					}
					ilo = 0;
					break;
				}
				if (x >= xt[ilo - 1])
					break;
			}
		}
		else {
			// **** now x >= xt[ihi] .	increase  ihi  to capture  x
			for(int istep = 1; ; istep *= 2) {
				ilo = ihi;
				ihi = ilo + istep;
				if (ihi >= n)
				{
					if (x >= xt[n - 1]) //right_boundary;
					{
						mflag[0] = 1;
						return n - 1;
					}
					ihi = n- 1;
					break;
				}
				if (x < xt[ihi])
					break;
			}
		}

		//* **** now xt[ilo] <= x < xt[ihi] . narrow the interval.
		for(;;) {
			int middle = (ilo + ihi) / 2;
			if (middle == ilo)
				return ilo;
			// note. it is assumed that middle = ilo in case ihi = ilo+1 .
			if (x >= xt[middle])
				ilo = middle;
			else
				ihi = middle;
		}
	}
	
	/**
	 * <P>Calculates value at  x  of  jderiv-th derivative of spline from B-repr.
	 * The spline is taken to be continuous from the right.
	 *
	 * <P>calls  interv
	 * 
	 * <P>Ripped from PPPACK. Original name: bvalue
	 * 
	 * <pre>
	 * ******  i n p u t ******
	 *  t, bcoef, n, k......forms the b-representation of the spline  f  to
	 *        be evaluated. specifically,
	 *  t.....knot sequence, of length  n+k, assumed nondecreasing.
	 *  bcoef.....b-coefficient sequence, of length  n .
	 *  n.....length of  bcoef  and dimension of s(k,t),
	 *        a s s u m e d  positive .
	 *  k.....order of the spline .
	 *
	 *  w a r n i n g . . .   the restriction  k <= kmax (=20)  is imposed
	 *        arbitrarily by the dimension statement for  aj, dm, dm  below,
	 *        but is  n o w h e r e  c h e c k e d  for.
	 *
	 *  x.....the point at which to evaluate .
	 *  jderiv.....integer giving the order of the derivative to be evaluated
	 *        a s s u m e d  to be zero or positive.
	 *
	 * ******  o u t p u t  ******
	 *  bvalue.....the value of the (jderiv)-th derivative of  f  at  x .
	 *
	 * ******  m e t h o d  ******
	 *     the nontrivial knot interval  (t(i),t(i+1))  containing  x  is lo-
	 *  cated with the aid of  interv(). the  k  b-coeffs of  f  relevant for
	 *  this interval are then obtained from  bcoef (or taken to be zero if
	 *  not explicitly available) and are then differenced  jderiv  times to
	 *  obtain the b-coeffs of  (d^jderiv)f  relevant for that interval.
	 *  precisely, with  j = jderiv, we have from x.(12) of the text that
	 *
	 *     (d^j)f  =  sum ( bcoef(.,j)*b(.,k-j,t) )
	 *
	 *  where
	 *                   / bcoef(.),                     ,  j .eq. 0
	 *                   /
	 *    bcoef(.,j)  =  / bcoef(.,j-1) - bcoef(.-1,j-1)
	 *                   / ----------------------------- ,  j > 0
	 *                   /    (t(.+k-j) - t(.))/(k-j)
	 * 
	 *     then, we use repeatedly the fact that
	 * 
	 *    sum ( a(.)*b(.,m,t)(x) )  =  sum ( a(.,x)*b(.,m-1,t)(x) )
	 *  with
	 *                 (x - t(.))*a(.) + (t(.+m-1) - x)*a(.-1)
	 *    a(.,x)  =    ---------------------------------------
	 *                 (x - t(.))      + (t(.+m-1) - x)
	 * 
	 *  to write  (d^j)f(x)  eventually as a linear combination of b-splines
	 *  of order  1 , and the coefficient for  b(i,1,t)(x)  must then
	 *  be the desired number  (d^j)f(x). (see x.(17)-(19) of text).
	 * 
	 * Arguments
	 *     dimension t(n+k)
	 *  current fortran standard makes it impossible to specify the length of
	 *  t  precisely without the introduction of otherwise superfluous
	 *  additional arguments.
	 *  </pre>
	 */
	private static final double evaluateBSplineDerivatives(double[] t, double[] bcoef, double x, int jderiv)
	{
		// Local variables
		int
			i = 1,
			n = bcoef.length,
			k = t.length - n,
			mflag[] = new int[1];
		double
			aj[] = new double[k],
			dm[] = new double[k-1],
			dp[] = new double[k-1];

		if (jderiv >= k)
			return 0;

		// *** find  i	s.t.  1 <= i < n+k  and	 t(i) < t(i+1) and
		//     t(i) <= x < t(i+1) . if no such i can be found,	x  lies
		//     outside the support of the spline  f and bvalue = 0.
		// {this case is handled in the calling R code}
		//     (the asymmetry in this choice of	 i  makes  f  rightcontinuous)
		if (x != t[n] || t[n] != t[n + k - 1])
		{
			i = determineInterval(t, x, 0, mflag) + 1;
			if (mflag[0] != 0)
				return 0; // Should never happen
		} else
			i = n;

		// *** if k = 1 (and jderiv = 0), bvalue = bcoef(i).
		int km1 = k - 1;
		if (km1 <= 0)
			return bcoef[i - 1];

		// store the k b-spline coefficients relevant for the knot interval
		// (t(i),t(i+1)) in aj(1),...,aj(k) and compute dm(j) = x - t(i+1-j),
		// dp(j) = t(i+j) - x, j=1,...,k-1 . set any of the aj not obtainable
		// from input to zero. set any t.s not obtainable equal to t(1) or
		// to t(n+k) appropriately.
		int
			jcmin = 1,
			imk = i - k;
		if (imk >= 0) {
			for (int j = 0; j < km1; ++j)
				dm[j] = x - t[i - j - 1];
		} else {
			// TODO This branch is not tested
			jcmin = 1 - imk;
			for (int j = 0; j < i; ++j)
				dm[j] = x - t[i - j + 1];
			for (int j = i - 1; j < km1; ++j)
				dm[j] = dm[i - 1];
		}

		int
			jcmax = k - 1,
			nmi = n - i;
		if (nmi >= 0) {
			for (int j = 0; j < km1; ++j)
				dp[j] = t[i + j] - x;
		} else {
			// TODO This branch is not tested
			jcmax = k + nmi - 1;
			for (int j = 0; j <= jcmax; ++j)
				dp[j] = t[i + j] - x;
			for (int j = jcmax; j < k; ++j)
				dp[j] = dp[jcmax];
		}

		for (int jc = jcmin - 1; jc <= jcmax; ++jc)
			aj[jc] = bcoef[imk + jc];

		// *** difference the coefficients  jderiv  times.
		for (int j = 0; j < jderiv; ++j) {
			int kmj = k - j - 1;
			for (int jj = 0, ilo = kmj - 1; jj < kmj; ++jj, --ilo)
				aj[jj] = (aj[jj + 1] - aj[jj]) / (dm[ilo] + dp[jj]) * kmj;
		}

		// *** compute value at  x  in (t(i),t(i+1)) of jderiv-th derivative,
		//     given its relevant b-spline coeffs in aj(1),...,aj(k-jderiv).
		for (int j = jderiv; j < km1; ++j) {
			int kmj = k - j - 2;
			for (int jj = 0, ilo = kmj; jj <= kmj; ++jj, --ilo)
				aj[jj] = (aj[jj + 1] * dm[ilo] + aj[jj] * dp[jj])  / (dm[ilo] + dp[jj]);
		}
		return aj[0];
	}

	public static final double predict(SmoothSplineResult result, double val, int deriv)
	{	return predict(result.mKnots, result.mCoefficients, result.mXMin, result.mXMax, val, deriv); }

	public static final double predict(double[] knots, double[] coefs, double xmin, double xmax, double val, int deriv)
	{
		double
			range = xmax - xmin,
			normalizedValue = (val - xmin) / range;

		if (normalizedValue >= 0 && normalizedValue <= 1)
			return evaluateBSplineDerivatives(knots, coefs, normalizedValue, deriv) / (deriv > 0 ? pow(range, deriv) : 1);
		// Either xs < 0 or xs > 1
		double neg = normalizedValue < 0 ? 0 : 1;
		if (deriv == 0)
		{
			double
				intercept = evaluateBSplineDerivatives(knots, coefs, neg, 0),
				slope = evaluateBSplineDerivatives(knots, coefs, neg, 1);
			return intercept + slope * (normalizedValue - neg);
		}
		else if (deriv == 1)
			return evaluateBSplineDerivatives(knots, coefs, neg, 1);
		return 0;
	}

	private static final void normalizeWeights(double[] weights)
	{
		// Normalize weights
		double sumW = 0;
		int
			n = weights.length,
			numNonZero = 0;
		for (int i = 0; i < n; i++)
		{
			double wt = weights[i];
			if (wt < 0)
				throw new RuntimeException("Negative weight is not allowed!");
			else
				if (wt > 0)
				{
					sumW += wt;
					numNonZero++;
				}
		}
		if (numNonZero == 0)
			throw new RuntimeException("All zero weight is not allowed!");
		sumW = numNonZero / sumW;
			for (int i = 0; i < n; i++)
				weights[i] *= sumW;
	}

	public static final SmoothSplineResult fitDFMatch(double[] x, double[] y, double df)
	{
		return fit(x,y,null,DF_MATCH, 1.0, df, Double.NaN,
				kDefaultSmoothingParamLowerBound, kDefaultSmoothingParamUpperBound,
				kDefaultTolerance, kDefaultMaxNumIterations);
	}

	public static final SmoothSplineResult fitDFMatch(double[] x, double[] y, double[] weights, double df)
	{
		return fit(x,y,weights,DF_MATCH, 1.0, df, Double.NaN,
				kDefaultSmoothingParamLowerBound, kDefaultSmoothingParamUpperBound,
				kDefaultTolerance, kDefaultMaxNumIterations);
	}

	public static final SmoothSplineResult fit(double[] x, double[] y)
	{
		return fit(x,y, null, SmoothSplineCriterion.NO_CRITERION, 1, 0, Double.NaN,
			kDefaultSmoothingParamLowerBound, kDefaultSmoothingParamUpperBound,
			kDefaultTolerance, kDefaultMaxNumIterations);
	}

	public static final SmoothSplineResult fit(double[] x, double[] y, double[] weights)
	{
		return fit(x,y,weights, SmoothSplineCriterion.NO_CRITERION, 1, 0, Double.NaN,
			kDefaultSmoothingParamLowerBound, kDefaultSmoothingParamUpperBound,
			kDefaultTolerance, kDefaultMaxNumIterations);
	}

	public static final SmoothSplineResult fit(double[] x, double[] y, double[] weights,
		SmoothSplineCriterion criterion, double penalty, double df)
	{
		return fit(x,y,weights,criterion, penalty, df, Double.NaN,
			kDefaultSmoothingParamLowerBound, kDefaultSmoothingParamUpperBound,
			kDefaultTolerance, kDefaultMaxNumIterations);
	}

	/**
	 * Fit a smooth spline
	 * @param x array of n observations
	 * @param y array of n observations
	 * @param weights array of n observations. Null for default weights.
	 * @param criterion one of NO_CRITERION, GCV, CV, and DF_MATCHING
	 * @param penalty Must be 0 < penalty <= 1. Default to 1
	 * @param df Supply if you want DF matching. Else, it will be used for DF offset.
	 * @param smoothingParam Put Double.NaN to estimate it.
	 */
	public static final SmoothSplineResult fit(double[] x, double[] y, double[] weights,
		SmoothSplineCriterion criterion, double penalty, double df, double smoothingParam)
	{
		return fit(x,y,weights,criterion, penalty, df, smoothingParam,
				kDefaultSmoothingParamLowerBound, kDefaultSmoothingParamUpperBound,
				kDefaultTolerance, kDefaultMaxNumIterations);
	}

	/**
	 * Fit a smooth spline
	 * @param x array of n observations
	 * @param y array of n observations
	 * @param weights array of n observations. Null for default weights.
	 * @param criterion one of NO_CRITERION, GCV, CV, and DF_MATCHING
	 * @param penalty Must be 0 < penalty <= 1. Default to 1
	 * @param df Supply if you want DF matching. Else, it will be used for DF offset.
	 * @param smoothingParam Put Double.NaN to estimate it.
	 * @param smoothingParamLBound Will be ignored if smoothingParam is not NaN. Default = -1.5
	 * @param smoothingParamUBound Will be ignored if smoothingParam is not NaN. Default = +1.5
	 * @param tolerance Tolerance threshold. Default is 0.0001.
	 * @param maxNumIterations maximum number of iterations. Default is 500.
	 * @return
	 */
	public static final SmoothSplineResult fit(double[] x, double[] y, double[] weights,
		SmoothSplineCriterion criterion, double penalty, double df, double smoothingParam,
		double smoothingParamLBound, double smoothingParamUBound, double tolerance, int maxNumIterations)
	{
		int n = x.length;
		if (y == null)
		{
			y = x;
			x = new double[n];
			for (int i = 0; i < n; i++)
				x[i] = i+1;
		}
		if (weights == null)
		{
			weights = new double[n];
			fill(weights, 1.0);
		}
		else
			normalizeWeights(weights);
		assert(n == y.length && n == weights.length);
		double
			sortedArray[][] = new double[3][n],
			sumOfSquare = 0; // ssw
		// Copy the array first. We don't want to clobber the data
		System.arraycopy(x, 0, sortedArray[0], 0, n);
		System.arraycopy(y, 0, sortedArray[1], 0, n);
		System.arraycopy(weights, 0, sortedArray[2], 0, n);
		sort2DByRow(sortedArray, 0); // Sort!
		int
			dupeIdx[] = findDuplicateIndices(sortedArray[0]),
			numUniqueElements = dupeIdx[n-1] + 1;
		double[][] sortedUniqueArray;
		if (numUniqueElements < n)
		{
			sortedUniqueArray = new double[3][numUniqueElements];
			double[]
				oldXBar = sortedArray[0],
				oldYBar = sortedArray[1],
				oldWBar = sortedArray[2],
				newXBar = sortedUniqueArray[0],
				newYBar = sortedUniqueArray[1],
				newWBar = sortedUniqueArray[2],
				tempBar = new double[numUniqueElements];
			for (int i = 0; i < n; i++)
			{
				int idx = dupeIdx[i];
				double
					wt = oldWBar[i],
					yi = oldYBar[i],
					wtyi = wt*yi;
				newXBar[idx] = oldXBar[i];
				newWBar[idx] += wt;
				newYBar[idx] += wtyi;
				tempBar[idx] += wtyi * yi;
			}
			for (int i = 0; i < numUniqueElements; i++)
			{
				double
					wi = newWBar[i],
					yi = newYBar[i];
				if (wi > 0)
					newYBar[i] = yi = yi / wi;
				sumOfSquare += tempBar[i] - wi * yi * yi;
			}
		}
		else
			sortedUniqueArray = sortedArray;

		double
			xbar[] = sortedUniqueArray[0],
			ybar[] = sortedUniqueArray[1],
			wbar[] = sortedUniqueArray[2],
			xmin = xbar[0],
			xmax = xbar[numUniqueElements-1],
			range = xmax - xmin;

		for (int i = 0; i < numUniqueElements; i++)
			xbar[i] = (xbar[i] - xmin) / range;
		double[] knots = formKnots(xbar);

		SmoothSplineResult result = fitCubicSpline(penalty, df, xbar, ybar, wbar, sumOfSquare, knots,
			criterion, smoothingParam, smoothingParamLBound, smoothingParamUBound,
			tolerance, kDefaultEpsilon, maxNumIterations);
		
		result.mXMax = xmax;
		result.mXMin = xmin;
		if (criterion == CV)
		{
			double
				sum = 0,
				sumW = 0;
			for (int i = 0; i < n; i++)
			{
				int idx = dupeIdx[i];
				double
					wi = weights[i],
					wbi = wbar[idx];
				if (wbi <= 0)
					wbi = 1;
				double temp = (y[i] - result.mSmoothedValues[idx]) / (1 - (result.mLeverage[idx] * wi) / wbi);
				sum += wi * temp * temp;
				sumW += wi;
			}
			result.mCVScore = sum / sumW;
		} else
		{
			double
				dfOffset = criterion == DF_MATCH ? 0 : df,
				sum = 0,
				sumW = 0,
				denom = 1 - (dfOffset + penalty * result.mEstimatedDF) / n;
			denom *= denom; // square it;
			for (int i = 0; i < n; i++)
			{
				double
					wi = weights[i],
					temp = y[i] - result.mSmoothedValues[dupeIdx[i]];
				sum += wi * temp * temp;
				sumW += wi;
			}
			result.mCVScore = (sum / sumW) / denom;
		}

		return result;
	}

	/**
	 * <P>Find duplicate indices, assuming arr is sorted.
	 * It's similar to R's <tt>match</tt> function, but it
	 * further assumes that <tt>arr</tt> is sorted.<br>
	 * Example:
	 * <br>Input: {1, 2, 2, 2, 3, 3, 4, 5, 6, 6}
	 * <br>Output: {0, 1, 1, 1, 2, 2, 3, 4, 5, 5}
	 * 
	 * <P>So, the number of unique elements can be obtained
	 * by (result[arr.length-1] + 1).
	 * @param arr
	 * @return
	 */
	static final int[] findDuplicateIndices(double[] arr)
	{
		int
			n = arr.length,
			idx = 0,
			result[] = new int[n];
		double lastX = arr[0];
		for (int i = 1; i < n; i++)
		{
			double curX = arr[i];
			if (curX != lastX)
			{
				idx++;
				lastX = curX;
			}
			result[i] = idx;
		}
		return result;
	}

	private static final void sort2DByRow(double[][] data, int rowNo)
	{
		int maxData = data[0].length - 1;
		quickSortByRow(data, rowNo, 0, maxData);
		insertionSortByRow(data, rowNo, 0, maxData);
	}

	/**
	 * Three median quick sort
	 *
	 * @param data       the data
	 * @param rowNo   the column index of which the table is compared against
	 * @param lo         left boundary of array partition
	 * @param hi         right boundary of array partition
	 */
	private static final void quickSortByRow(double[][] data, int rowNo, int lo, int hi)
	{
		int left, right, midpoint;

		if ((hi - lo) <= kInsertionTreshold)
			return;

		midpoint = (hi + lo)/2;
		if (data[rowNo][lo] > data[rowNo][midpoint])
			swapRow(data, lo, midpoint);
		if (data[rowNo][lo] > data[rowNo][hi])
			swapRow(data, lo, hi);
		if (data[rowNo][midpoint] > data[rowNo][hi])
			swapRow(data, midpoint, hi);

		right = hi - 1;
		swapRow(data, midpoint, right);
		left = lo;
		double
			dataRow[] = data[rowNo],
			pivotElement = dataRow[right];
		for(;;)
		{
			while(dataRow[++left] < pivotElement){}
			while(dataRow[--right] > pivotElement){}
			if (right < left)
				break;
			swapRow(data, left, right);
		}
		swapRow(data, left, hi-1);
		quickSortByRow(data, rowNo, lo, right);
		quickSortByRow(data, rowNo, left+1, hi);
	}

	/**
	 * Insertion Sort
	 * 
	 * @param data       the data
	 * @param rowNo   the row index of which the table is compared against
	 * @param lo         left boundary of array partition
	 * @param hi         right boundary of array partition
	 */
	private static final void insertionSortByRow(double[][] data, int rowNo, int lo, int hi)
	{
		double element_i, element_j;
		int j, jp1, numRows = data.length;
		double[] col_i = new double[numRows];
		for (int i = 1; i <= hi; i++)
		{
			for (int r = 0; r < numRows; r++)
				col_i[r] = data[r][i];
			element_i = data[rowNo][i];
			for (j = i - 1; j >= lo; j--)
			{
				element_j = data[rowNo][j];
				if (element_j <= element_i)
					break;
				jp1 = j+1;
				for (int r = 0; r < numRows; r++)
					data[r][jp1] = data[r][j]; // a[j+1] = a[j]
			}
			jp1 = j+1;
			for (int r = 0; r < numRows; r++)
				data[r][jp1] = col_i[r];
		}
	}

	private static final void swapRow(double[][] data, int i, int j)
	{
		int numRows = data.length;
		for (int rowNo = 0; rowNo < numRows; rowNo++)
		{
			double
				curRow[] = data[rowNo],
				temp = curRow[i];
			curRow[i] = curRow[j];
			curRow[j] = temp;
		}
	}

	static void testBSPLVD()
	{
		// Correct output:
		// 0.00  1.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000
		// 0.25  0.5625000  0.3750000  0.0625000  0.0000000  0.0000000  0.0000000  0.0000000
		// 0.50  0.2500000  0.5000000  0.2500000  0.0000000  0.0000000  0.0000000  0.0000000
		// 0.75  0.0625000  0.3750000  0.5625000  0.0000000  0.0000000  0.0000000  0.0000000
		// 1.00  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000  0.0000000  0.0000000
		// 1.25  0.0000000  0.0000000  0.7656250  0.2239583  0.0104167  0.0000000  0.0000000
		// 1.50  0.0000000  0.0000000  0.5625000  0.3958333  0.0416667  0.0000000  0.0000000
		// 1.75  0.0000000  0.0000000  0.3906250  0.5156250  0.0937500  0.0000000  0.0000000
		// 2.00  0.0000000  0.0000000  0.2500000  0.5833333  0.1666667  0.0000000  0.0000000
		// 2.25  0.0000000  0.0000000  0.1406250  0.5989583  0.2604167  0.0000000  0.0000000
		// 2.50  0.0000000  0.0000000  0.0625000  0.5625000  0.3750000  0.0000000  0.0000000
		// 2.75  0.0000000  0.0000000  0.0156250  0.4739583  0.5104167  0.0000000  0.0000000
		// 3.00  0.0000000  0.0000000  0.0000000  0.3333333  0.6666667  0.0000000  0.0000000
		// 3.25  0.0000000  0.0000000  0.0000000  0.1875000  0.7916667  0.0208333  0.0000000
		// 3.50  0.0000000  0.0000000  0.0000000  0.0833333  0.8333333  0.0833333  0.0000000
		// 3.75  0.0000000  0.0000000  0.0000000  0.0208333  0.7916667  0.1875000  0.0000000
		// 4.00  0.0000000  0.0000000  0.0000000  0.0000000  0.6666667  0.3333333  0.0000000
		// 4.25  0.0000000  0.0000000  0.0000000  0.0000000  0.5104167  0.4739583  0.0156250
		// 4.50  0.0000000  0.0000000  0.0000000  0.0000000  0.3750000  0.5625000  0.0625000
		// 4.75  0.0000000  0.0000000  0.0000000  0.0000000  0.2604167  0.5989583  0.1406250
		// 5.00  0.0000000  0.0000000  0.0000000  0.0000000  0.1666667  0.5833333  0.2500000
		// 5.25  0.0000000  0.0000000  0.0000000  0.0000000  0.0937500  0.5156250  0.3906250
		// 5.50  0.0000000  0.0000000  0.0000000  0.0000000  0.0416667  0.3958333  0.5625000
		// 5.75  0.0000000  0.0000000  0.0000000  0.0000000  0.0104167  0.2239583  0.7656250
		// 6.00  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  1.0000000

		// Taken from Example 1 of:
		// de Boor (1977) Package for Calculating with B-Splines, SIAM J. Numerical Analysis 14(3), June 1977
		double
			t[] = new double[] {0,0,0,1,1,3,4,6,6,6},
			values[][] = new double[1][3];
		int
			n = 7,
			k = 3,
			nPoint = 25,
			iLeft = 0;
		double
			xl = t[k - 1],
			dx = (t[n] - xl) / (nPoint - 1);
		for (int i = 0; i < nPoint; i++)
		{
			double
				x = xl + i * dx,
				allValues[] = new double[n];
			iLeft = determineInterval(t, x, iLeft, null);
			if (iLeft >= n)
				iLeft = n - 1;
			calculateBSplineDerivatives(t, x, iLeft, values);
			int imkp1 = iLeft - k + 1;
			for (int j = 0; j < k; j++)
			{
				allValues[imkp1+j] = values[0][j];
				values[0][j] = 0;
			}
			System.out.print(String.format("%5.2f  %5.7f", x, allValues[0])); //$NON-NLS-1$
			for (int j = 1; j < n; j++)
				System.out.print(String.format("  %5.7f",allValues[j])); //$NON-NLS-1$
			System.out.println();
		}
	}

	static void testSmoothSplines()
	{
		/*
		 * xi and fi are taken from Pang, Tao (2006) "An Introduction to Computational Physics", 2nd ed.
		 * speed, dist, and y18 are taken from R's car example
		 */
		double
			xi[] = new double[] { 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
				0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1},
			fi[] = new double[] { 0.420650268, 0.869057123, 0.243074468, 0.352583416,
				0.869472216, 0.219531331, 0.663085387, 0.476098157, 0.781728422, 0.509583544,
				0.570627762, 0.540800479, 0.233658831, 0.103976142, 0.527013717, 0.519541602,
				0.935703005, 0.360399664, 0.237154223, 0.851028235, 0.231539401},
			speed[] = new double[] {4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13,
				13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18,
				19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25},
			dist[] = new double[] {2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46,
				26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68,
				32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85},
			y18[] = new double[] {1, 2, 3, 5, 4, 7, 6, 5, 4, 3, 4, 6, 8, 10, 10, 10, 10, 10};
		assert (xi.length == fi.length);
		assert (speed.length == dist.length);
		SmoothSplineResult result;
		//*
		result = fit(fi, xi, null, DF_MATCH, 1, 3);
		//Minimizing criterion: DF_MATCH
		//Smoothing parameter: 1.018851345458852
		//Lambda: 0.020315773036788925
		//Num. iterations: 11
		//Equivalent degrees of freedom (DF): 3.0003649975970452
		//Penalized criterion: 1.8626611778254198
		//GCV: 0.1517015406647021
		System.out.println(result);
		System.out.println(predict(result, 0.05, 0)); //0.6374198108495056
		System.out.println(predict(result, 0.90, 0)); //0.4909193517530635
		System.out.println(predict(result, 0.95, 0)); //0.5055859812416792
		System.out.println();

		result = fit(fi, xi, null, GCV, 1, 0);
		//Minimizing criterion: GCV
		//Smoothing parameter: 1.4999418748090076
		//Lambda: 60.7881102827463
		//Num. iterations: 26
		//Equivalent degrees of freedom (DF): 2.0009149997606093
		//Penalized criterion: 1.9085673029717933
		//GCV: 0.13324690261258523
		System.out.println(result);
		System.out.println(predict(result, 0.05, 0)); //0.5514480705995526 vs 0.5514477
		System.out.println(predict(result, 0.90, 0)); //0.45455352986272307 vs 0.4545536
		System.out.println(predict(result, 0.95, 0)); //0.4488745767544037 vs 0.4488747
		System.out.println();

		result = fit(fi, xi, null, CV, 1, 0);
		//Minimizing criterion: CV
		//Smoothing parameter: 1.4999579810285835
		//Lambda: 60.80439975985657
		//Num. iterations: 30
		//Equivalent degrees of freedom (DF): 2.0009147864018386
		//Penalized criterion: 1.9085673187134673
		//PRESS: 0.14641910796023755
		System.out.println(result);
		System.out.println(predict(result, 0.05, 0)); //0.551448070184632
		System.out.println(predict(result, 0.90, 0)); //0.45455351361617224
		System.out.println(predict(result, 0.95, 0)); //0.4488745540299733
		System.out.println();
		//*/

		result = fit(speed, dist, null, GCV, 1, 0);
		//Minimizing criterion: GCV
		//Smoothing parameter: 0.7802768994560999
		//Lambda: 0.11121547691795662
		//Num. iterations: 11
		//Equivalent degrees of freedom (DF): 2.6352697498668305
		//Penalized criterion: 4187.779014409647
		//GCV: 244.1043967439251
		System.out.println(result);

		result = fit(speed, dist, null, DF_MATCH, 1, 10);
		//Minimizing criterion: DF_MATCH
		//Smoothing parameter: 0.3541654525300812
		//Lambda: 9.281738283403245E-5
		//Num. iterations: 9
		//Equivalent degrees of freedom (DF): 9.99857714398653
		//Penalized criterion: 1680.6894921712935
		//GCV: 263.9022507178775
		System.out.println(result);

		result = fit(y18, null, null, GCV, 1, 0);
		System.out.println(result);
	}

//	public static void main(String argv[])
//	{
//		//testBSPLVD();
//		testSmoothSplines();
//	}
}
