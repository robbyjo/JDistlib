/*
 * Roby Joehanes
 * 
 * Copyright 2007 Roby Joehanes
 * This file is distributed under the GNU General Public License version 2.0.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * 
 */
package jdistlib.math;

import static java.lang.Math.*;
import static jdistlib.math.Constants.*;

/**
 * Function optimization routines. Currently only Brent's minimization routine.
 * Maybe I'll add Nelder-Meade or other fancier methods.
 * 
 * @author Roby Joehanes
 *
 */
public class Optimization {
	/**
	 * Brent's minimization function with default tolerance (1e-10)
	 * @param f
	 * @param ax
	 * @param bx
	 * @return the x at which f(x) is the minimum value
	 */
	public static final strictfp double optimize(UnivariateFunction f, double ax, double bx)
	{	return optimize(f, ax, bx, 1e-10, 1000); }

	/**
	 * Richard Brent's function minimization routine.<br>
	 * <a href="http://en.wikipedia.org/wiki/Brent%27s_method">Wikipedia's link</a><br>
	 * Adapted from <a href="http://www.netlib.org/go/fmin.f">Netlib's fmin</a>.<br>
	 * Also looked at Numerical Methods in C, 2nd ed. Chapter 10 section 1 and 2.
	 * 
	 * @param f the function to minimize
	 * @param ax lower bound
	 * @param bx upper bound
	 * @param tol tolerance
	 * @param maxiter the maximum number of iterations
	 */
	public static final strictfp double optimize(UnivariateFunction f, double ax, double bx, double tol, int maxiter)
	{
	    // Local variables
	    double
	    	a = ax, b = bx, v = a + kInvGoldRatio * (b - a), w = v, x = v,
	    	d = 0.0, e = 0.0, fu, fv, fw, fx, tol3;
	    int iterNo = 0;

	    fv = fw = fx = f.eval(x);
	    tol3 = tol / 3.0;

	    // main loop starts here
	    for(;;) {
	    	iterNo++;
			double
				xm = (a + b) * 0.5,
				tol1 = SQRT_DBL_EPSILON * abs(x) + tol3,
				t2 = tol1 * 2.0;

			// check stopping criterion
			if (abs(x - xm) <= t2 - (b - a) * .5)
				break;
			double
				p = 0.0, q = 0.0, r = 0.0;
			if (abs(e) > tol1) { // fit parabola	
				r = (x - w) * (fx - fv);
				q = (x - v) * (fx - fw);
				p = (x - v) * q - (x - w) * r;
				q = (q - r) * 2.0;
				if (q > 0.0) p = -p; else q = -q;
				r = e;
				e = d;
			}
	
			if (abs(p) >= abs(q * 0.5 * r) || p <= q * (a - x) || p >= q * (b - x)) { // a golden-section step 
				e = x < xm ? b - x : a - x;
				d = kInvGoldRatio * e;
			}
			else { // a parabolic-interpolation step
				d = p / q;
				double u = x + d;
				// f must not be evaluated too close to ax or bx
				if (u - a < t2 || b - u < t2)
					d = x >= xm ? -tol1 : tol1;
			}

			// f must not be evaluated too close to x
			double u = abs(d) >= tol1 ? x + d : (d > 0 ? x + tol1 : x - tol1);	
			fu = f.eval(u);

			// update  a, b, v, w, and x
			if (fu <= fx) {
				if (u < x) b = x; else a = x;
				v = w; w = x; x = u;
				fv = fw; fw = fx; fx = fu;
			} else {
				if (u < x) a = u; else b = u;
				if (fu <= fw || w == x) {
					v = w; fv = fw;
					w = u; fw = fu;
				} else if (fu <= fv || v == x || v == w) {
					v = u; fv = fu;
				}
			}
			if (iterNo >= maxiter) {
				System.err.println("Warning: Convergence failure in optimize function!");
				break;
			}
		}
		// end of main loop
	    return x;
	}

	/** <pre>************************************************************************
	 *			    C math library
	 * function ZEROIN - obtain a function zero within the given range
	 *
	 * Output
	 *	Zeroin returns an estimate for the root with accuracy
	 *	4*EPSILON*abs(x) + tol
	 *
	 * Algorithm
	 *	G.Forsythe, M.Malcolm, C.Moler, Computer methods for mathematical
	 *	computations. M., Mir, 1980, p.180 of the Russian edition
	 *
	 *	The function makes use of the bisection procedure combined with
	 *	the linear or quadric inverse interpolation.
	 *	At every step program operates on three abscissae - a, b, and c.
	 *	b - the last and the best approximation to the root
	 *	a - the last but one approximation
	 *	c - the last but one or even earlier approximation than a that
	 *		1) |f(b)| <= |f(c)|
	 *		2) f(b) and f(c) have opposite signs, i.e., b and c confine
	 *		   the root
	 *	At every step Zeroin selects one of the two new approximations, the
	 *	former being obtained by the bisection procedure and the latter
	 *	resulting in the interpolation (if a,b, and c are all different
	 *	the quadric interpolation is utilized, otherwise the linear one).
	 *	If the latter (i.e. obtained by the interpolation) point is
	 *	reasonable (i.e. lies within the current interval [b,c] not being
	 *	too close to the boundaries) it is accepted. The bisection result
	 *	is used in the other case. Therefore, the range of uncertainty is
	 *	ensured to be reduced at least by the factor 1.6
	 *
	 ************************************************************************
	 *
	 * NOTE:  uniroot() --> do_zeroin2()  --- in  ../main/optimize.c
	 *					      ~~~~~~~~~~~~~~~~~~
	 * </pre>
	 * @param f The function whose zero is sought
	 * @param ax Root will be sought for within a range [ax,bx]
	 * @param bx
	 * @param tol Acceptable tolerance for the root value. May be specified as 0.0 to cause the program to find the root as accurate as possible.
	 * @param maxiter Max. iterations
	 * @return the x where f(x) == 0
	 */
	public static final strictfp double zeroin(UnivariateFunction f, double ax, double bx, double tol, int maxiter) {
		double maxit = maxiter+1, a = ax, b = bx, c = a, fa = f.eval(ax), fb = f.eval(bx), fc = fa;
		// First test if we have found a root at an endpoint
		if (fa == 0) return a;
		if (fb == 0) return b;
		while(maxit-- > 0) // Main iteration loop
		{
			double prev_step = b-a; // Distance from the last but one to the last approximation
			double tol_act;	// Actual tolerance
			double p, q; // Interpolation step is calculated in the form p/q; division operations is delayed until the last moment
			double new_step; // Step at this iteration

			if( abs(fc) < abs(fb) ) {
				a = b;  b = c;  c = a;	//Swap data for b to be the best approximation
				fa=fb;  fb=fc;  fc=fa;
			}
			tol_act = 2*DBL_EPSILON*abs(b) + tol/2;
			new_step = (c-b)/2;

			if( abs(new_step) <= tol_act || fb == 0.0 )
				return b; // Acceptable approx. is found

			// Decide if the interpolation can be tried
			if( abs(prev_step) >= tol_act // If prev_step was large enough
					&& abs(fa) > abs(fb) ) { // and was in true direction, interpolation may be tried
				double t1,cb,t2;
				cb = c-b;
				if( a==c ) { // If we have only two distinct points linear interpolation can only be applied
					t1 = fb/fa;
					p = cb*t1;
					q = 1.0 - t1;
				} else { // Quadric inverse interpolation
					q = fa/fc;  t1 = fb/fc;	 t2 = fb/fa;
					p = t2 * ( cb*q*(q-t1) - (b-a)*(t1-1.0) );
					q = (q-1.0) * (t1-1.0) * (t2-1.0);
				}
				if( p > 0 )	// p was calculated with the opposite sign; make p positive
					q = -q;
				else // and assign possible minus to q
					p = -p;

				if( p < (0.75*cb*q-abs(tol_act*q)/2) // If b+p/q falls in [b,c]
						&& p < abs(prev_step*q/2) )	// and isn't too large
					new_step = p/q;	 // it is accepted
				// If p/q is too large then the bisection procedure can reduce [b,c] range to more extent
			}

			if( abs(new_step) < tol_act) { // Adjust the step to be not less than tolerance
				if( new_step > 0 )
					new_step = tol_act;
				else
					new_step = -tol_act;
			}
			a = b;	fa = fb; // Save the previous approx.
			b += new_step;	fb = f.eval(b); // Do step to a new approxim.
			if( (fb > 0 && fc > 0) || (fb < 0 && fc < 0) ) {
				c = a;  fc = fa; // Adjust c for it to have a sign opposite to that of b
			}

		}
		System.err.println("Warning: Convergence failure in zeroin function!");
		return b;
	}

	public static void main(String[] args) {
		UnivariateFunction f = new UnivariateFunction() {
			public double eval(double x) {
				x = x - 1/3.0; 
				return x * x;
			}
			public void setObjects(Object... obj) {}
			public void setParameters(double... params) {}
		};
		System.out.println(optimize(f, 0, 1, 0.0001, 1000));
		System.out.println(optimize(f, 0, 1, 0, 1000));

		f = new UnivariateFunction() {
			public double eval(double x) {
				return x * (x * x - 1) + 0.5;
			}
			public void setObjects(Object... obj) {}
			public void setParameters(double... params) {}
		};
		System.out.println(zeroin(f, -2, 2, 0.0001, 1000));
		System.out.println(zeroin(f, -2, 2, 1e-10, 1000));
		System.out.println(zeroin(f, -2, 2, 0, 1000));
	}
}
