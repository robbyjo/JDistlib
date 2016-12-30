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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package net.sourceforge.jdistlib.math;

import static java.lang.Math.abs;
import static java.lang.Math.cos;
import static java.lang.Math.exp;
import static java.lang.Math.hypot;
import static java.lang.Math.log;
import static java.lang.Math.PI;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

import net.sourceforge.jdistlib.util.Utilities;

/**
 * <P>A Java object that represents polynomials as arrays of numerical coefficients.
 * The single variable is simple (not a function). 
 * The lowest (0) -degree term is at index 0 and the
 * higher-degree terms follow to the right. No power is left out
 * even if its coefficient is 0.
 * 
 * <P>Also, this class does not represent polynomials with negative powers
 * (e.g. x^-1) and all coefficients have to be real numbers.
 * 
 * @author Roby Joehanes
 */
public class Polynomial
{
	private static final double
		kDegree = 94 * PI / 180.0,
		kMRE = 2 * sqrt(2) * Constants.DBL_EPSILON,
		kCosR = cos(kDegree),
		kSinR = sin(kDegree);
	private static final Polynomial mPrivateInstance = new Polynomial();
	// "Global variables" of Jenkins-Traub algorithm. This is a result of bad coding.
	private class JenkinsTraubGlobal
	{
		int nn;
		double sr, si, tr, ti, pvr, pvi, omp, relstp;
		double[] pr, pi, hr, hi, qpr, qpi, qhr, qhi, shr, shi;
	}

	protected double[] mCoefficients;

	/**
	 * Private instance
	 */
	private Polynomial() {}

	public Polynomial(int degree)
	{	mCoefficients = new double[degree + 1];
	}

	/*
	 * A copy constructor. May be dangerous to use the input data directly
	 * since we could construct other Polys from the same data. Then altering
	 * one would alter all!
	 */
	public Polynomial(double... coeffs)
	{	
		this(coeffs.length - 1);	// create a new coeffs array
		System.arraycopy(coeffs, 0, mCoefficients, 0, coeffs.length);
	}

	/**
	 * Get the coefficients
	 * @return
	 */
	public double[] getCoefficients()
	{	return mCoefficients;}

	/**
	 * Get the degree of the polynomial
	 * @return
	 */
	public int getDegree()
	{
		// Makes sure that this polynomial in the compact form, that is:
		// The polynomial may not have the coefficient of its highest degree as 0.
		// Otherwise, the result will be wrong. Example:
		// If the polynomial object contains these coefficients: [2, 3, 1, 0]
		// That means: 0 x^3 + 1 x^2 + 3 x^1 + 2 x^0
		// getDegree() must return 2 instead of 3.
		compact();
		return mCoefficients.length - 1;
	}

	/**
	 * Set coefficient of this polynomial
	 * @param power
	 * @param value
	 */
	public void setCoefficient(int power, double value)
	{	mCoefficients[power] = value; }

	public static final Polynomial chooseHigherDegreePolynomial(Polynomial p1, Polynomial p2)
	{	return (p1.getDegree() >= p2.getDegree()) ? p1 : p2;	}

	public static final Polynomial chooseLowerDegreePolynomial(Polynomial p1, Polynomial p2)
	{	return (p1.getDegree() >= p2.getDegree()) ? p2 : p1;	}

	/**
	 * Add another polynomial to this polynomial and store the result into a new instance of QPolynomial
	 * @param poly
	 */
	public Polynomial plus(Polynomial poly)
	{
		Polynomial result = clone();
		result.plusEquals(poly);
		return result;
	}

	/**
	 * Add another polynomial into this polynomial and let the result overwrite this polynomial
	 * (this = this + poly)
	 * @param poly
	 */
	public void plusEquals(Polynomial poly)
	{
		double[]
			hiPoly = chooseHigherDegreePolynomial(this, poly).getCoefficients().clone(),
			// clone only because _this_ polynomial may not be the one of higher degree
		 	loPoly = chooseLowerDegreePolynomial(this, poly).getCoefficients();

		int minDegree = loPoly.length;
		for (int i = 0; i < minDegree; i++)
			hiPoly[i] += loPoly[i];
		mCoefficients = hiPoly;
		compact(); // just in case that the addition may cause coeffs at higher degree become zero
	}

	/**
	 * Subtract another polynomial from this polynomial and store the result into a new instance of QPolynomial
	 * @param poly
	 */
	public Polynomial minus(Polynomial poly)
	{
		Polynomial result = clone();
		result.minusEquals(poly);
		return result;
	}

	/**
	 * Subtract another polynomial from this polynomial, in place (this = this - poly)
	 * @param poly
	 */
	public void minusEquals(Polynomial poly)
	{
		plusEquals(poly.timesScalar(-1.0));
	}

	/**
	 * Multiply this polynomial with a constant and store the result into a new instance
	 * of QPolynomial
	 * @param c
	 */
	public Polynomial timesScalar(double c)
	{
		Polynomial result = clone();
		result.timesScalarEquals(c);
		return result;
	}

	/**
	 * Multiply this polynomial with a constant, in place
	 * @param c
	 */
	public void timesScalarEquals(double c)
	{
		if (c == 0.0) // short-circuit multiply by 0
			mCoefficients = new double[] {0.0};
		else
		{
			double[] result = mCoefficients;
			int degree = mCoefficients.length;
			for (int i = 0; i < degree; i++)
				result[i] *= c;
		}
	}

	/**
	 * Multiply this polynomial with another polynomial and store the result into a new instance of QPolynomial
	 * @param poly
	 */
	public Polynomial times(Polynomial poly)
	{
		Polynomial result = clone();
		result.timesEquals(poly);
		return result;
	}

	/**
	 * Multiply this polynomial with another polynomial, in place
	 * @param poly
	 */
	public void timesEquals(Polynomial poly)
	{
		int
			thisDegree = mCoefficients.length -1,
			thatDegree = poly.mCoefficients.length - 1,	
			resultDegree = thisDegree + thatDegree + 1;

		// resultDegree is the degree of the new poly, and we add 1 for the constant term. CN 11.14.05
		double[]
			productCoeffs = new double[resultDegree],
			theseCoeffs = mCoefficients,
			thoseCoeffs = poly.getCoefficients();

		for (int i = 0; i <= thisDegree; i++)	
			for (int j = 0; j <= thatDegree; j++)
				productCoeffs[i + j] += theseCoeffs[i] * thoseCoeffs[j]; 

		mCoefficients = productCoeffs;
		compact(); // force compaction just in case we have zero coefficients at highest degrees
	}

	/**
	 * Compute the derivative of this polynomial and store the result into a new instance of QPolynomial
	 * @return the derivative
	 */
	public Polynomial differentiate()
	{
		int degree = mCoefficients.length;
		double[] result = new double[degree - 1]; // reduce the degree by one
		for (int i = 1; i <= degree; i++)
			result[i - 1] = mCoefficients[i] * i;
		return new Polynomial(result);
	}

	/**
	 * Compute the integral of this polynomial.
	 * The last constant is c
	 * @param c The constant at the last term (i.e. x^0)
	 * @return the integrated polynomial
	 */
	public Polynomial integrate(double c)
	{
		int degree = mCoefficients.length - 1;
		double[] result = new double[degree + 2]; // increase the degree & length by one
		for (int i = 0; i <= degree; i++)
			result[i + 1] = mCoefficients[i]/(i + 1);
		result[0] = c;
		return new Polynomial(result);
	}

	/**
	 * Compute the integral of this polynomial.
	 * The last constant defaults to ZERO
	 * @return the integrated polynomial
	 */
	public Polynomial integrate()
	{	return integrate(0); }

	/**
	 * Compute a definite integral bounded by (a, b)
	 * @param a
	 * @param b
	 * @return
	 */
	public double integrate(double a, double b)
	{
		Polynomial poly = integrate();
		return poly.evaluate(b) - poly.evaluate(a); 
	}

	/**
	 * Evaluate this polynomial at x = x0
	 * @param x0
	 * @return The result
	 */
	public double evaluate(double x0)
	{
		double
			sum = 0.0,
			power = 1.0;
		int degree = getDegree();
		for (int i = 0; i <= degree; i++)
		{
			sum += mCoefficients[i] * power;
			power *= x0;
		}
		return sum;
	}
	
	/**
	 * Evaluate this polynomial based on a precomputed vector of powers of the variable.
	 * To be used in PolynomialMatrix. CN 10.31.05
	 * @param powersOfX[].
	 * @return a scalar, the vector product
	 */
	public double evaluate(double powersOfX[])
	{
		double sum = 0.0;
		int degree = getDegree();
		
		for (int i = 0; i <= degree; i++)
			sum += mCoefficients[i] * powersOfX[i];
		return sum;
	}

	/**
	 * Compacts the representation of this polynomial. That is,
	 * we don't want the highest coefficient to be zero
	 * For example: If the polynomial object contains these coefficients: [2, 3, 1, 0],
	 * which means: 0 x^3 + 1 x^2 + 3 x^1 + 2 x^0 --
	 * compact() method will delete the coefficient at x^3, like this:<br>
	 * 1 x^2 + 3 x^1 + 2 x^0<br>
	 * Thus, the QPolynomial will have this array instead: [2, 3, 1]
	 */
	public void compact()
	{
		int	coeffLength = mCoefficients.length;
		// Check the coefficients at index 1 and above (i.e. the ones that are not at x^0)
		for (int i = coeffLength - 1; mCoefficients[i] == 0 && i > 0; i--)
			coeffLength--;
		if (coeffLength != mCoefficients.length)
		{
			double[] result = new double[coeffLength];
			System.arraycopy(mCoefficients, 0, result, 0, coeffLength);
			mCoefficients = result;
		}
	}

	/**
	 * Clone this polynomial
	 */
	@Override
	public Polynomial clone()
	{	return new Polynomial(mCoefficients.clone()); }

	/**
	 * Construct a string representation of this polynomial
	 * @param varName
	 * @return
	 */
	public String toString(String varName)
	{
		compact(); // make sure we're printing the compact form of this polynomials
		StringBuffer buf = new StringBuffer();
		int	curDegree = mCoefficients.length - 1;
		if (curDegree > 0)
		{
			for (int i = curDegree; i >= 0; i--)
			{
				double coeff = mCoefficients[i];
				if (coeff == 0)
					continue;
				if (coeff > 0)
				{
					if (buf.length() > 0)
						buf.append(" + "); //$NON-NLS-1$
				}
				else
					buf.append(" - "); //$NON-NLS-1$
				coeff = Math.abs(coeff);
				if (coeff != 1 || i == 0)
					buf.append(coeff);
				if (i > 1)
					buf.append(varName + "^" + i); //$NON-NLS-1$
				else if (i > 0)
					buf.append(varName);
			}
		}
		else // (curDegree == 0) // We must print the constant if the degree is zero.
			buf.append(mCoefficients[0]);
		return buf.toString().trim();
	}

	/**
	 * Can we simplify this polynomial? If the lowest coefficients are zero,
	 * the polynomial looks like x^n * simpler_polynomial. So we want to
	 * factor that out first.
	 * @return coefficients of the simpler polynomials
	 */
	public Polynomial simplify()
	{
		int
			lowestNonZeroIndex = 0,
			degree = getDegree();
		double[] coeff = mCoefficients;
		for (int i = 0; i <= degree; i++)
		{
			if (mCoefficients[i] == 0)
				lowestNonZeroIndex = i+1;
			else
				break;
		}
		// If we can simplify it, write new coefficient
		if (lowestNonZeroIndex > 0)
		{
			degree = getDegree() - lowestNonZeroIndex;
			coeff = new double[degree + 1];
			System.arraycopy(mCoefficients, lowestNonZeroIndex, coeff, 0, degree + 1);
			return new Polynomial(coeff);
		}
		return this;
	}

	/**
	 * Returns the root. result[0] is the real part. result[1] is the imaginary part
	 * @return
	 */
	public double[][] findRoots()
	{
		compact();
		return findRoots(mCoefficients, null);
	}

	/**
	 * Get a string representation of this polynomial with x as the variable name
	 */
	@Override
	public String toString()
	{	return toString("x"); } //$NON-NLS-1$

	/**
	 * Jenkins-Traub algorithm for finding root of polynomials.
	 * The lowest (0) -degree term is at index 0 and the
	 * higher-degree terms follow to the right. No power is left out.
	 * @param realCoef Real coefficients
	 * @param imCoef Imaginary coefficients
	 * @return result[0] will be the real part, result[1] will be the imaginary part of the roots.
	 */
	public static final double[][] findRoots(double[] realCoef, double[] imCoef)
	{
		int n = realCoef.length;
		if (imCoef == null)
			imCoef = new double[n];
		else
		{
			assert(n == imCoef.length);
			imCoef = Utilities.rev(imCoef);
		}
		realCoef = Utilities.rev(realCoef);

		// degree is n - 1
		if (realCoef[0] == 0 && imCoef[0] == 0) // The coefficent of the highest polynomial can NOT be zero
			return null;
		JenkinsTraubGlobal globalVar = mPrivateInstance.new JenkinsTraubGlobal();
		boolean conv = false;
		double
			result[][] = new double[2][n-1],
			realRoot[] = result[0],
			imRoot[] = result[1],
			xx = Constants.M_1_SQRT_2,
			yy = -xx,
			zs[] = new double[2]; // zr and zi
		globalVar.nn = n - 1;
		int d1 = globalVar.nn - 1;

		while (realCoef[globalVar.nn] == 0. && imCoef[globalVar.nn] == 0.)
			globalVar.nn--;
		globalVar.nn++;
		if (globalVar.nn == 1)
			return result;
		double[][] tmp = new double[10][globalVar.nn];
		globalVar.pr = tmp[0];
		globalVar.pi = tmp[1];
		globalVar.hr = tmp[2];
		globalVar.hi = tmp[3];
		globalVar.qpr = tmp[4];
		globalVar.qpi = tmp[5];
		globalVar.qhr = tmp[6];
		globalVar.qhi = tmp[7];
		globalVar.shr = tmp[8];
		globalVar.shi = tmp[9];

		for (int i = 0; i < globalVar.nn; i++)
		{
			globalVar.pr[i] = realCoef[i];
			globalVar.pi[i] = imCoef[i];
			globalVar.shr[i] = Math.hypot(globalVar.pr[i], globalVar.pi[i]);
		}

		double bnd = findRootsPolynomialScale(globalVar.shr);
		if (bnd != 1)
			for (int i=0; i < globalVar.nn; i++)
			{
				globalVar.pr[i] *= bnd;
				globalVar.pi[i] *= bnd;
			}

		// start the algorithm for one zero
		while (globalVar.nn > 2)
		{
			// calculate bnd, a lower bound on the modulus of the zeros.
			for (int i=0 ; i < globalVar.nn ; i++)
				globalVar.shr[i] = hypot(globalVar.pr[i], globalVar.pi[i]);
			bnd = findRootsPolynomialCauchy(globalVar.nn, globalVar.shr, globalVar.shi);

			// outer loop to control 2 major passes with different sequences of shifts
			for (int i1 = 1; i1 <= 2; i1++)
			{
				// first stage calculation, no shift
				findRootsNoShift(globalVar, 5);

				// inner loop to select a shift
				for (int i2 = 1; i2 <= 9; i2++)
				{
					// shift is chosen with modulus bnd and amplitude rotated by 94 degrees
					// from the previous shift

					double xxx= kCosR * xx - kSinR * yy;
					yy = kSinR * xx + kCosR * yy;
					xx = xxx;
					globalVar.sr = bnd * xx;
					globalVar.si = bnd * yy;
					// second stage calculation, fixed shift
					conv = findRootsFixShift(globalVar, i2 * 10, zs);
					if (conv)
						break;
				}
				if (conv)
					break;
			}

			// the zerofinder has failed on two major passes return empty handed
			if (!conv)
				return null;

			/* the second stage jumps directly to the third stage iteration.
			 * if successful, the zero is stored and the polynomial deflated.
			 */
			int d_n = d1+2 - globalVar.nn;
			realRoot[d_n] = zs[0];
			imRoot[d_n] = zs[1];
			--globalVar.nn;
			for (int i=0; i < globalVar.nn ; i++)
			{
				globalVar.pr[i] = globalVar.qpr[i];
				globalVar.pi[i] = globalVar.qpi[i];
			}
	    } // end while

		// calculate the final zero and return
		double[] out = new double[2];
		findRootsCdivid(-globalVar.pr[1], -globalVar.pi[1], globalVar.pr[0], globalVar.pi[0], out);
		realRoot[d1] = out[0];
		imRoot[d1] = out[1];

	    return result;
	}

	// Adapted from R (GPL)
	private static final double findRootsPolynomialScale(double[] shr)
	{
		// find largest and smallest moduli of coefficients.
		int n = shr.length;
		double
			high = sqrt(Constants.DBL_MAX),
			lo = Constants.DBL_MIN / Constants.DBL_EPSILON,
			max_ = 0.,
			min_ = Constants.DBL_MAX;
		for (int i = 0; i < n; i++) {
			double x = shr[i];
			if (x > max_) max_ = x;
			if (x != 0. && x < min_)
				min_ = x;
		}

		// scale only if there are very large or very small components.
		if (min_ < lo || max_ > high) {
			double
				x = lo / min_,
				sc;
			if (x <= 1.)
				sc = 1. / (sqrt(max_) * sqrt(min_));
			else {
				sc = x;
				if (Constants.DBL_MAX / sc > max_)
					sc = 1.0;
			}
			int ell = (int) (log(sc) / log(2.0) + 0.5);
			return pow(2.0, ell);
	    }
		return 1.0;
	}

	// Adapted from R (GPL)
	private static final double findRootsPolynomialCauchy(int n, double[] pot, double[] q)
	{
	    // Computes a lower bound on the moduli of the zeros of a polynomial pot[1:nn] is the modulus of the coefficients.
		int n1 = n - 1;
		pot[n1] = -pot[n1];

		// compute upper estimate of bound.
		double x = exp((log(-pot[n1]) - log(pot[0])) / n1);

		// if newton step at the origin is better, use it.
		if (pot[n1-1] != 0.0) {
			double xm = -pot[n1] / pot[n1-1];
			if (xm < x)
				x = xm;
		}

		// chop the interval (0,x) until f <= 0.
		for(;;) {
			double
				xm = x * 0.1,
				f = pot[0];
			for (int i = 1; i < n; i++)
				f = f * xm + pot[i];
			if (f <= 0.0)
				break;
			x = xm;
		}
		double dx = x;

		// do Newton iteration until x converges to two decimal places.
		while (abs(dx / x) > 0.005)
		{
			q[0] = pot[0];
			for(int i = 1; i < n; i++)
				q[i] = q[i-1] * x + pot[i];
			double
				f = q[n1],
				delf = q[0];
			for(int i = 1; i < n1; i++)
				delf = delf * x + q[i];
			dx = f / delf;
			x -= dx;
		}
		return x;
	}

	/**
	 * Computes the derivative polynomial as the initial
	 * polynomial and computes l1 no-shift h polynomials.<br>
	 * Adapted from R (GPL)
	 */
	private static final void findRootsNoShift(JenkinsTraubGlobal globalVar, int l1)
	{
		int
			nn = globalVar.nn,
			n = nn - 1,
			nm1 = n - 1;
		double[]
			out = new double[2],
			hr = globalVar.hr,
			hi = globalVar.hi,
			pr = globalVar.pr,
			pi = globalVar.pi;

		for (int i=0; i < n; i++)
		{
			double xni = nn - i - 1;
			hr[i] = xni * pr[i] / n;
			hi[i] = xni * pi[i] / n;
	    }

		for (int jj = 1; jj <= l1; jj++)
		{
			if (hypot(hr[n-1], hi[n-1]) <= Constants.DBL_EPSILON * 10.0 * hypot(pr[n-1], pi[n-1])) {
				// If the constant term is essentially zero, shift h coefficients.

				for (int i = 1; i <= nm1; i++)
				{
					int j = nn - i;
					hr[j-1] = hr[j-2];
					hi[j-1] = hi[j-2];
				}
				hr[0] = hi[0] = 0;
			}
			else {
				findRootsCdivid(-pr[nn-1], -pi[nn-1], hr[n-1], hi[n-1], out);
				double
					tr = globalVar.tr = out[0],
					ti = globalVar.ti = out[1];
				for (int i = 1; i <= nm1; i++) {
					int j = nn - i;
					double
						t1 = hr[j-2],
						t2 = hi[j-2];
					hr[j-1] = tr * t1 - ti * t2 + pr[j-1];
					hi[j-1] = tr * t2 + ti * t1 + pi[j-1];
				}
				hr[0] = pr[0];
				hi[0] = pi[0];
			}
		}
	}

	/*
	 * Computes l2 fixed-shift h polynomials and tests for convergence.
	 * initiates a variable-shift iteration and returns with the
	 * approximate zero if successful.
	 */
	private static final boolean findRootsFixShift(JenkinsTraubGlobal globalVar, int l2, double[] zs)
	{
		/*  l2	  - limit of fixed shift steps
		 *  zr,zi - approximate zero if convergence (result TRUE)
		 *
		 * Return value indicates convergence of stage 3 iteration
		 *
		 * Uses global (sr,si), nn, pr[], pi[], .. (all args of polyev() !)
		 */

		double svsi, svsr, oti, otr;
		double[]
			pr = globalVar.pr,
			pi = globalVar.pi,
			hr = globalVar.hr,
			hi = globalVar.hi,
			qpr = globalVar.qpr,
			qpi = globalVar.qpi,
			shr = globalVar.shr,
			shi = globalVar.shi,
			out = new double[2];
		int
			nn = globalVar.nn,
			n = nn - 1;

		// evaluate p at s.
		findRootsPolynomialEval(nn, globalVar.sr, globalVar.si, pr, pi, qpr, qpi, out);
		globalVar.pvr = out[0];
		globalVar.pvi = out[1];

		boolean
			test = true,
			pasd = false,
			// calculate first t = -p(s)/h(s).
			bool = findRootsCalct(globalVar);

		// main loop for one second stage step.
		for (int j=1; j<=l2; j++)
		{
			otr = globalVar.tr;
			oti = globalVar.ti;

			// compute next h polynomial and new t.
			findRootsNextH(globalVar, bool);
			bool = findRootsCalct(globalVar);
			zs[0] = globalVar.sr + globalVar.tr;
			zs[1] = globalVar.si + globalVar.ti;

			// test for convergence unless stage 3 has
			// failed once or this is the last h polynomial.

			if (!bool && test && j != l2) {
				if (hypot(globalVar.tr - otr, globalVar.ti - oti) >= hypot(zs[0], zs[1]) * 0.5)
					pasd = false;
				else if (! pasd)
					pasd = true;
				else {
					// the weak convergence test has been passed twice, start the third stage
					// iteration, after saving the current h polynomial and shift.
					System.arraycopy(hr, 0, shr, 0, n);
					System.arraycopy(hi, 0, shi, 0, n);
					svsr = globalVar.sr;
					svsi = globalVar.si;
					if (findRootsVariableShift(globalVar, 10, zs))
						return true;

					// the iteration failed to converge. Turn off testing and restore h, s, pv and t.
					test = false;
					System.arraycopy(shr, 0, hr, 0, n);
					System.arraycopy(shi, 0, hi, 0, n);
					globalVar.sr = svsr;
					globalVar.si = svsi;
					findRootsPolynomialEval(nn, globalVar.sr, globalVar.si, pr, pi, qpr, qpi, out);
					globalVar.pvr = out[0];
					globalVar.pvi = out[1];
					bool = findRootsCalct(globalVar);
				}
			}
		}

		// attempt an iteration with final h polynomial from second stage.
		return(findRootsVariableShift(globalVar, 10, zs));
	}

	/*
	 * evaluates a polynomial  p  at  s	 by the horner recurrence
	 * placing the partial sums in q and the computed value in v_.<br>
	 * Adapted from R (GPL)
	 */
	private static final void findRootsPolynomialEval(int n, double s_r, double s_i, double[] p_r, double[] p_i, double[] q_r, double[] q_i, double[] out)
	{
		q_r[0] = p_r[0];
		q_i[0] = p_i[0];
		double
			v_r = q_r[0],
			v_i = q_i[0];
		for (int i = 1; i < n; i++)
		{
			double t = v_r * s_r - v_i * s_i + p_r[i];
			q_i[i] = v_i = v_r * s_i + v_i * s_r + p_i[i];
			q_r[i] = v_r = t;
		}
		out[0] = v_r;
		out[1] = v_i;
	}

	private static final boolean findRootsCalct(JenkinsTraubGlobal globalVar)
	{
		// computes	 t = -p(s)/h(s).
		// bool   - logical, set true if h(s) is essentially zero.
		int n = globalVar.nn - 1;
		double out[] = new double[2];

		// evaluate h(s).
		findRootsPolynomialEval(n, globalVar.sr, globalVar.si, globalVar.hr, globalVar.hi, globalVar.qhr, globalVar.qhi, out);
		double
			hvr = out[0],
			hvi = out[1];

		boolean bool = hypot(hvr, hvi) <= Constants.DBL_EPSILON * 10.0 * hypot(globalVar.hr[n-1], globalVar.hi[n-1]);
		if (!bool)
		{
			findRootsCdivid(-globalVar.pvr, -globalVar.pvi, hvr, hvi, out);
			globalVar.tr = out[0];
			globalVar.ti = out[1];
		} else
			globalVar.tr = globalVar.ti = 0.;
		return bool;
	}

	/* carries out the third stage iteration.
	 */
	private static final boolean findRootsVariableShift(JenkinsTraubGlobal globalVar, int l3, double[] zs)
	{
		/*  l3	    - limit of steps in stage 3.
		 *  zr,zi   - on entry contains the initial iterate;
		 *	      if the iteration converges it contains
		 *	      the final iterate on exit.
		 * Returns TRUE if iteration converges
		 *
		 * Assign and uses  GLOBAL sr, si
		 */
		boolean bool, b = false;
		double r1, r2, mp, ms, tp, out[] = new double[2];

		globalVar.sr = zs[0];
		globalVar.si = zs[1];

		// main loop for stage three
		for (int i = 1; i <= l3; i++)
		{
			// evaluate p at s and test for convergence.
			findRootsPolynomialEval(globalVar.nn, globalVar.sr, globalVar.si, globalVar.pr, globalVar.pi, globalVar.qpr, globalVar.qpi, out);
			globalVar.pvr = out[0];
			globalVar.pvi = out[1];
			mp = hypot(globalVar.pvr, globalVar.pvi);
			ms = hypot(globalVar.sr, globalVar.si);
			if (mp <=  20. * findRootsErrorEval(globalVar.nn, globalVar.qpr, globalVar.qpi, ms, mp))
			{
			    zs[0] = globalVar.sr;
			    zs[1] = globalVar.si;
			    return true;
			}

			// polynomial value is smaller in value than a bound on the error in evaluating p, terminate the iteration.
			if (i != 1)
			{
				if (!b && mp >= globalVar.omp && globalVar.relstp < .05)
				{
					// iteration has stalled. probably a cluster of zeros. do 5 fixed shift
					// steps into the cluster to force one zero to dominate. */
					tp = globalVar.relstp;
					b = true;
					if (globalVar.relstp < Constants.DBL_EPSILON)
						tp = Constants.DBL_EPSILON;
					r1 = sqrt(tp);
					r2 = globalVar.sr * (r1 + 1.) - globalVar.si * r1;
					globalVar.si = globalVar.sr * r1 + globalVar.si * (r1 + 1.);
					globalVar.sr = r2;
					findRootsPolynomialEval(globalVar.nn, globalVar.sr, globalVar.si, globalVar.pr, globalVar.pi, globalVar.qpr, globalVar.qpi, out);
					globalVar.pvr = out[0];
					globalVar.pvi = out[1];
					for (int j = 1; j <= 5; ++j)
					{
						bool = findRootsCalct(globalVar);
						findRootsNextH(globalVar, bool);
					}
					mp = Constants.DBL_MAX;
				}
				else {
					// exit if polynomial value increases significantly.
					if (mp * .1 > globalVar.omp)
						return false;
				}
			}
			globalVar.omp = mp;
			// calculate next iterate.

			bool = findRootsCalct(globalVar);
			findRootsNextH(globalVar, bool);
			bool = findRootsCalct(globalVar);
			if (!bool)
			{
				globalVar.relstp = hypot(globalVar.tr, globalVar.ti) / hypot(globalVar.sr, globalVar.si);
				globalVar.sr += globalVar.tr;
				globalVar.si += globalVar.ti;
			}
		}
	    return false;
	}

	// Adapted from R (GPL)
	private static final void findRootsNextH(JenkinsTraubGlobal globalVar, boolean bool)
	{
		/* calculates the next shifted h polynomial.
		 * bool :	if TRUE  h(s) is essentially zero
		 */
		int n = globalVar.nn - 1;

		if (!bool) {
			for (int j=1; j < n; j++)
			{
				double
					t1 = globalVar.qhr[j - 1],
					t2 = globalVar.qhi[j - 1];
				globalVar.hr[j] = globalVar.tr * t1 - globalVar.ti * t2 + globalVar.qpr[j];
				globalVar.hi[j] = globalVar.tr * t2 + globalVar.ti * t1 + globalVar.qpi[j];
			}
			globalVar.hr[0] = globalVar.qpr[0];
			globalVar.hi[0] = globalVar.qpi[0];
		}
		else {
			// if h(s) is zero replace h with qh.
			for (int j=1; j < n; j++) {
				globalVar.hr[j] = globalVar.qhr[j-1];
				globalVar.hi[j] = globalVar.qhi[j-1];
			}
			globalVar.hr[0] = globalVar.hi[0] = 0.;
		}
	}

	// Adapted from R (GPL)
	private static final double findRootsErrorEval(int n, double[] qr, double[] qi, double ms, double mp)
	{
		//	bounds the error in evaluating the polynomial by the horner recurrence.
		// qr,qi	 - the partial sum vectors
		// ms	 - modulus of the point
		// mp	 - modulus of polynomial value

		double e = hypot(qr[0], qi[0]) * kMRE / (Constants.DBL_EPSILON + kMRE);
		for (int i=0; i < n; i++)
			e = e*ms + hypot(qr[i], qi[i]);

		return e * (Constants.DBL_EPSILON + kMRE) - mp * kMRE;
	}

	// Adapted from R (GPL)
	private static final void findRootsCdivid(double ar, double ai, double br, double bi, double[] out)
	{
		// complex division c = a/b, i.e., (cr +i*ci) = (ar +i*ai) / (br +i*bi), avoiding overflow.
		double d, r, cr, ci;

		if (br == 0. && bi == 0.) // division by zero, c = infinity.
			cr = ci = Double.POSITIVE_INFINITY;
		else if (abs(br) >= abs(bi))
		{
			r = bi / br;
			d = br + r * bi;
			cr = (ar + ai * r) / d;
			ci = (ai - ar * r) / d;
		}
		else
		{
			r = br / bi;
			d = bi + r * br;
			cr = (ar * r + ai) / d;
			ci = (ai * r - ar) / d;
		}
		out[0] = cr;
		out[1] = ci;
	}
	
	/*
	 * Tries to divide this poly by another. Returns 0 if the quotient is
	 * not a scalar value. This is for a special purpose and here we are not
	 * interested in factoring into polynomials. CN 11.6.10
	 */
	public double findScalarQuotient(Polynomial divisorPoly)
	{
		int
			thatDegree = divisorPoly.mCoefficients.length - 1,
			thisDegree = mCoefficients.length - 1;
		
		if (thisDegree != thatDegree) // else they won't divide
		{	return 0;}

		double quotient = 0.0;
		for (int i = 0; i <= thisDegree; i++)
		{
			double
				divisor = divisorPoly.mCoefficients[i],
				dividend = mCoefficients[i];
			if (divisor * dividend != 0.0) // one must be nonzero
			{ return 0;}
			if (divisor == 0.0) // avoid divX0
			{	return 0;}		
			if (quotient == 0.0) // quotient not yet established
			{ quotient = dividend/divisor;}
			else if (divisor * quotient != dividend)
			{ return 0;}
		}
		return quotient;
	}

	static final void testRootPoly()
	{
		double
			quad[] = new double[] {1,2,1},
			choose[] = new double[] {1, 8, 28, 56, 70, 56, 28, 8, 1};
		double[][] result;
		result = findRoots(quad, null);
		System.out.println(result);
		result = findRoots(choose, null);
		System.out.println(result);
		quad = new double[] {5, -6, 1};
		result = findRoots(quad, null);
		System.out.println(result);
	}

	// Test drive
	public static void main(String[] args)
	{
		Polynomial
			poly1 = new Polynomial(1,-3,-3,1),
			poly2 = new Polynomial(1, 2, 1),
			poly3 = poly2.times(poly1),
			poly4 = poly2.integrate();
		System.out.println(poly1);
		System.out.println(poly1.differentiate());
		System.out.println(poly1.differentiate().integrate());
		System.out.println(poly2);
		System.out.println(poly3);
		System.out.println(poly3.getDegree());
		System.out.println(poly4);
		System.out.println(poly4.differentiate());
	}
}

