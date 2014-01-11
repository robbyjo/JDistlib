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
 *  <http://www.gnu.org/licenses/>.
 */
package jdistlib.math;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static jdistlib.math.Constants.DBL_EPSILON;
import static jdistlib.math.Constants.DBL_MIN;
import static jdistlib.math.Constants.DBL_MAX;
import jdistlib.Normal;

public class Integrate {
	//static double c_b6 = 0.;
	//static double c_b7 = 1.;

	/**
	***begin prologue  dqagi
	***date written   800101   (yymmdd)
	***revision date  830518   (yymmdd)
	***category no.  h2a3a1,h2a4a1
	***keywords  automatic integrator, infinite intervals,
	            general-purpose, transformation, extrapolation,
	            globally adaptive
	***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
	          de doncker,elise,appl. math. & progr. div. -k.u.leuven
	***purpose  the routine calculates an approximation result to a given
	           integral   i = integral of f over (bound,+infinity)
	           or i = integral of f over (-infinity,bound)
	           or i = integral of f over (-infinity,+infinity)
	           hopefully satisfying following claim for accuracy
	           abs(i-result) <= max(epsabs,epsrel*abs(i)).
	***description

	       integration over infinite intervals
	       standard fortran subroutine

	       parameters
	        on entry
	           f      - double precision
	                    function subprogram defining the integrand
	                    function f(x). the actual name for f needs to be
	                    declared e x t e r n a l in the driver program.

	           bound  - double precision
	                    finite bound of integration range
	                    (has no meaning if interval is doubly-infinite)

	           inf    - int
	                    indicating the kind of integration range involved
	                    inf = 1 corresponds to  (bound,+infinity),
	                    inf = -1            to  (-infinity,bound),
	                    inf = 2             to (-infinity,+infinity).

	           epsabs - double precision
	                    absolute accuracy requested
	           epsrel - double precision
	                    relative accuracy requested
	                    if  epsabs <= 0
	                    and epsrel < max(50*rel.mach.acc.,0.5d-28),
	                    the routine will end with ier = 6.


	        on return
	           result - double precision
	                    approximation to the integral

	           abserr - double precision
	                    estimate of the modulus of the absolute error,
	                    which should equal or exceed abs(i-result)

	           neval  - int
	                    number of integrand evaluations

	           ier    - int
	                    ier = 0 normal and reliable termination of the
	                            routine. it is assumed that the requested
	                            accuracy has been achieved.
	                  - ier > 0 abnormal termination of the routine. the
	                            estimates for result and error are less
	                            reliable. it is assumed that the requested
	                            accuracy has not been achieved.
	           error messages
	                    ier = 1 maximum number of subdivisions allowed
	                            has been achieved. one can allow more
	                            subdivisions by increasing the value of
	                            limit (and taking the according dimension
	                            adjustments into account). however, if
	                            this yields no improvement it is advised
	                            to analyze the integrand in order to
	                            determine the integration difficulties. if
	                            the position of a local difficulty can be
	                            determined (e.g. singularity,
	                            discontinuity within the interval) one
	                            will probably gain from splitting up the
	                            interval at this point and calling the
	                            integrator on the subranges. if possible,
	                            an appropriate special-purpose integrator
	                            should be used, which is designed for
	                            handling the type of difficulty involved.
	                        = 2 the occurrence of roundoff error is
	                            detected, which prevents the requested
	                            tolerance from being achieved.
	                            the error may be under-estimated.
	                        = 3 extremely bad integrand behaviour occurs
	                            at some points of the integration
	                            interval.
	                        = 4 the algorithm does not converge.
	                            roundoff error is detected in the
	                            extrapolation table.
	                            it is assumed that the requested tolerance
	                            cannot be achieved, and that the returned
	                            result is the best which can be obtained.
	                        = 5 the integral is probably divergent, or
	                            slowly convergent. it must be noted that
	                            divergence can occur with any other value
	                            of ier.
	                        = 6 the input is invalid, because
	                            (epsabs <= 0 and
	                             epsrel < max(50*rel.mach.acc.,0.5d-28))
	                             or limit < 1 or leniw < limit*4.
	                            result, abserr, neval, last are set to
	                            zero. exept when limit or leniw is
	                            invalid, iwork(1), work(limit*2+1) and
	                            work(limit*3+1) are set to zero, work(1)
	                            is set to a and work(limit+1) to b.

	        dimensioning parameters
	           limit - int
	                   dimensioning parameter for iwork
	                   limit determines the maximum number of subintervals
	                   in the partition of the given integration interval
	                   (a,b), limit >= 1.
	                   if limit < 1, the routine will end with ier = 6.

	           lenw  - int
	                   dimensioning parameter for work
	                   lenw must be at least limit*4.
	                   if lenw < limit*4, the routine will end
	                   with ier = 6.

	           last  - int
	                   on return, last equals the number of subintervals
	                   produced in the subdivision process, which
	                   determines the number of significant elements
	                   actually in the work arrays.

	        work arrays
	           iwork - int
	                   vector of dimension at least limit, the first
	                   k elements of which contain pointers
	                   to the error estimates over the subintervals,
	                   such that work(limit*3+iwork(1)),... ,
	                   work(limit*3+iwork(k)) form a decreasing
	                   sequence, with k = last if last <= (limit/2+2), and
	                   k = limit+1-last otherwise

	           work  - double precision
	                   vector of dimension at least lenw
	                   on return
	                   work(1), ..., work(last) contain the left
	                    end points of the subintervals in the
	                    partition of (a,b),
	                   work(limit+1), ..., work(limit+last) contain
	                    the right end points,
	                   work(limit*2+1), ...,work(limit*2+last) contain the
	                    integral approximations over the subintervals,
	                   work(limit*3+1), ..., work(limit*3)
	                    contain the error estimates.

	***routines called  dqagie
	***end prologue  dqagi
	*/
	static final void dqagi(UnivariateFunction f, double bound, int inf,
		    double epsabs, double epsrel,
		    double result, double abserr, int neval, int ier,
		    int limit)
	{
	    ier = 6;
	    neval = 0;
	    result = 0.;
	    abserr = 0.;
	    if (limit < 1) return;

	    dqagie(f, bound, inf, epsabs, epsrel, limit);

	    return;
	} /* dqagi */

	/**begin prologue  dqagie
	***date written   800101   (yymmdd)
	***revision date  830518   (yymmdd)
	***category no.  h2a3a1,h2a4a1
	***keywords  automatic integrator, infinite intervals,
	            general-purpose, transformation, extrapolation,
	            globally adaptive
	***author  piessens,robert,appl. math & progr. div - k.u.leuven
	          de doncker,elise,appl. math & progr. div - k.u.leuven
	***purpose  the routine calculates an approximation result to a given
	           integral   i = integral of f over (bound,+infinity)
	           or i = integral of f over (-infinity,bound)
	           or i = integral of f over (-infinity,+infinity),
	           hopefully satisfying following claim for accuracy
	           abs(i-result) <= max(epsabs,epsrel*abs(i))
	***description

	integration over infinite intervals
	standard fortran subroutine

	           f      - double precision
	                    function subprogram defining the integrand
	                    function f(x). the actual name for f needs to be
	                    declared e x t e r n a l in the driver program.

	           bound  - double precision
	                    finite bound of integration range
	                    (has no meaning if interval is doubly-infinite)

	           inf    - double precision
	                    indicating the kind of integration range involved
	                    inf = 1 corresponds to  (bound,+infinity),
	                    inf = -1            to  (-infinity,bound),
	                    inf = 2             to (-infinity,+infinity).

	           epsabs - double precision
	                    absolute accuracy requested
	           epsrel - double precision
	                    relative accuracy requested
	                    if  epsabs <= 0
	                    and epsrel < max(50*rel.mach.acc.,0.5d-28),
	                    the routine will end with ier = 6.

	           limit  - int
	                    gives an upper bound on the number of subintervals
	                    in the partition of (a,b), limit >= 1

	        on return
	           result - double precision
	                    approximation to the integral

	           abserr - double precision
	                    estimate of the modulus of the absolute error,
	                    which should equal or exceed abs(i-result)

	           neval  - int
	                    number of integrand evaluations

	           ier    - int
	                    ier = 0 normal and reliable termination of the
	                            routine. it is assumed that the requested
	                            accuracy has been achieved.
	                  - ier > 0 abnormal termination of the routine. the
	                            estimates for result and error are less
	                            reliable. it is assumed that the requested
	                            accuracy has not been achieved.
	           error messages
	                    ier = 1 maximum number of subdivisions allowed
	                            has been achieved. one can allow more
	                            subdivisions by increasing the value of
	                            limit (and taking the according dimension
	                            adjustments into account). however,if
	                            this yields no improvement it is advised
	                            to analyze the integrand in order to
	                            determine the integration difficulties.
	                            if the position of a local difficulty can
	                            be determined (e.g. singularity,
	                            discontinuity within the interval) one
	                            will probably gain from splitting up the
	                            interval at this point and calling the
	                            integrator on the subranges. if possible,
	                            an appropriate special-purpose integrator
	                            should be used, which is designed for
	                            handling the type of difficulty involved.
	                        = 2 the occurrence of roundoff error is
	                            detected, which prevents the requested
	                            tolerance from being achieved.
	                            the error may be under-estimated.
	                        = 3 extremely bad integrand behaviour occurs
	                            at some points of the integration
	                            interval.
	                        = 4 the algorithm does not converge.
	                            roundoff error is detected in the
	                            extrapolation table.
	                            it is assumed that the requested tolerance
	                            cannot be achieved, and that the returned
	                            result is the best which can be obtained.
	                        = 5 the integral is probably divergent, or
	                            slowly convergent. it must be noted that
	                            divergence can occur with any other value
	                            of ier.
	                        = 6 the input is invalid, because
	                            (epsabs <= 0 and
	                             epsrel < max(50*rel.mach.acc.,0.5d-28),
	                            result, abserr, neval, last, rlist(1),
	                            elist(1) and iord(1) are set to zero.
	                            alist(1) and blist(1) are set to 0
	                            and 1 respectively.

	           alist  - double precision
	                    vector of dimension at least limit, the first
	                     last  elements of which are the left
	                    end points of the subintervals in the partition
	                    of the transformed integration range (0,1).

	           blist  - double precision
	                    vector of dimension at least limit, the first
	                     last  elements of which are the right
	                    end points of the subintervals in the partition
	                    of the transformed integration range (0,1).

	           rlist  - double precision
	                    vector of dimension at least limit, the first
	                     last  elements of which are the integral
	                    approximations on the subintervals

	           elist  - double precision
	                    vector of dimension at least limit,  the first
	                    last elements of which are the moduli of the
	                    absolute error estimates on the subintervals

	           iord   - int
	                    vector of dimension limit, the first k
	                    elements of which are pointers to the
	                    error estimates over the subintervals,
	                    such that elist(iord(1)), ..., elist(iord(k))
	                    form a decreasing sequence, with k = last
	                    if last <= (limit/2+2), and k = limit+1-last
	                    otherwise

	           last   - int
	                    number of subintervals actually produced
	                    in the subdivision process

	***routines called  dqelg,dqk15i,dqpsrt
	***end prologue  dqagie


	           the dimension of rlist2 is determined by the value of
	           limexp in subroutine dqelg.

	           list of major variables
	           -----------------------

	          alist     - list of left end points of all subintervals
	                      considered up to now
	          blist     - list of right end points of all subintervals
	                      considered up to now
	          rlist(i)  - approximation to the integral over
	                      (alist(i),blist(i))
	          rlist2    - array of dimension at least (limexp+2),
	                      containing the part of the epsilon table
	                      wich is still needed for further computations
	          elist(i)  - error estimate applying to rlist(i)
	          maxerr    - pointer to the interval with largest error
	                      estimate
	          errmax    - elist(maxerr)
	          erlast    - error on the interval currently subdivided
	                      (before that subdivision has taken place)
	          area      - sum of the integrals over the subintervals
	          errsum    - sum of the errors over the subintervals
	          errbnd    - requested accuracy max(epsabs,epsrel*
	                      abs(result))
	          *****1    - variable for the left subinterval
	          *****2    - variable for the right subinterval
	          last      - index for subdivision
	          nres      - number of calls to the extrapolation routine
	          numrl2    - number of elements currently in rlist2. if an
	                      appropriate approximation to the compounded
	                      integral has been obtained, it is put in
	                      rlist2(numrl2) after numrl2 has been increased
	                      by one.
	          small     - length of the smallest interval considered up
	                      to now, multiplied by 1.5
	          erlarg    - sum of the errors over the intervals larger
	                      than the smallest interval considered up to now
	          extrap    - logical variable denoting that the routine
	                      is attempting to perform extrapolation. i.e.
	                      before subdividing the smallest interval we
	                      try to decrease the value of erlarg.
	          noext     - logical variable denoting that extrapolation
	                      is no longer allowed (true-value)

	           machine dependent constants
	           ---------------------------

	          epmach is the largest relative spacing.
	          uflow is the smallest positive magnitude.
	          oflow is the largest positive magnitude.
	*/
	static IntegrationResult dqagie(UnivariateFunction f, double bound, int inf, double epsabs, double epsrel, int limit) {
		double[] alist = new double[limit+1];
		double[] blist = new double[limit+1];
		double[] rlist = new double[limit+1];
		double[] elist = new double[limit+1];
		int[] iord = new int[limit+1];
		IntegrationResult result = new IntegrationResult();
		result.f = f;

		/* Local variables */
		double area, dres;
		int ksgn;
		double boun;
		int nres;
		double area1, area2, area12;
		int k;
		double small = 0.0, erro12;
		int ierro;
		double a1, a2, b1, b2, defab1 = 0, defab2 = 0, oflow;
		int ktmin, nrmax;
		double uflow;
		boolean noext, extrap;
		int iroff1, iroff2, iroff3;
		double res3la[] = new double[3], error1 = 0, error2 = 0;
		int id;
		double rlist2[] = new double [52];
		int numrl2;
		double deabs = 0, epmach, erlarg = 0.0, abseps = 0, correc = 0.0, errbnd, resabs = 0;
		int jupbnd;
		double erlast, errmax;
		int maxerr;
		double reseps = 0;
		double ertest = 0.0, errsum;
		double[] temp1 = new double[1], temp2 = new double[1], temp3 = new double[1];
		int[] temp4 = new int[1], temp5 = new int[1];


		/* ***first executable statement  dqagie */

		/* Function Body */
		epmach = DBL_EPSILON;

		/*           test on validity of parameters */
		/*           ----------------------------- */

		result.ier = 0;
		result.neval = 0;
		result.last = 0;
		result.result = 0.;
		result.abserr = 0.;
		alist[1] = 0.;
		blist[1] = 1.;
		rlist[1] = 0.;
		elist[1] = 0.;
		iord[1] = 0;
		if (epsabs <= 0. && (epsrel < max(epmach * 50., 5e-29))) result.ier = 6;
		if (result.ier == 6) return result;

		/*           first approximation to the integral */
		/*           ----------------------------------- */

		/*         determine the interval to be mapped onto (0,1).
	           if inf = 2 the integral is computed as i = i1+i2, where
	           i1 = integral of f over (-infinity,0),
	           i2 = integral of f over (0,+infinity). */

		boun = bound;
		if (inf == 2) {
			boun = 0.;
		}
		temp1[0] = result.abserr; temp2[0] = deabs; temp3[0] = resabs;
		result.result = dqk15i(f, boun, inf, 0., 1., temp1, temp2, temp3);
		result.abserr = temp1[0]; deabs = temp2[0]; resabs = temp3[0];

		/*           test on accuracy */

		result.last = 1;
		rlist[1] = result.result;
		elist[1] = result.abserr;
		iord[1] = 1;
		dres = abs(result.result);
		errbnd = max(epsabs, epsrel * dres);
		if (result.abserr <= epmach * 100. * deabs && result.abserr > errbnd) result.ier = 2;
		if (limit == 1) result.ier = 1;
		if (result.ier != 0 || (result.abserr <= errbnd && result.abserr != resabs) || result.abserr == 0.) {
			//goto L130;
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}

		/*           initialization */
		/*           -------------- */

		uflow = DBL_MIN;
		oflow = DBL_MAX;
		rlist2[0] = result.result;
		errmax = result.abserr;
		maxerr = 1;
		area = result.result;
		errsum = result.abserr;
		result.abserr = oflow;
		nrmax = 1;
		nres = 0;
		ktmin = 0;
		numrl2 = 2;
		extrap = false;
		noext = false;
		ierro = 0;
		iroff1 = 0;
		iroff2 = 0;
		iroff3 = 0;
		ksgn = -1;
		if (dres >= (1. - epmach * 50.) * deabs) {
			ksgn = 1;
		}

		/*           main do-loop */
		/*           ------------ */

		for (result.last = 2; result.last <= limit; ++(result.last)) {

			/*           bisect the subinterval with nrmax-th largest error estimate. */

			a1 = alist[maxerr];
			b1 = (alist[maxerr] + blist[maxerr]) * .5;
			a2 = b1;
			b2 = blist[maxerr];
			erlast = errmax;

			temp1[0] = error1; temp2[0] = resabs; temp3[0] = defab1;
			area1 = dqk15i(f, boun, inf, a1, b1, temp1, temp2, temp3);
			error1 = temp1[0]; resabs = temp2[0]; defab1 = temp3[0];

			temp1[0] = error2; temp2[0] = resabs; temp3[0] = defab2;
			area2 = dqk15i(f, boun, inf, a2, b2, temp1, temp2, temp3);
			error2 = temp1[0]; resabs = temp2[0]; defab2 = temp3[0];

			/*           improve previous approximations to integral
		     and error and test for accuracy. */

			area12 = area1 + area2;
			erro12 = error1 + error2;
			errsum = errsum + erro12 - errmax;
			area = area + area12 - rlist[maxerr];
			if (!(defab1 == error1 || defab2 == error2)) {
				if (abs(rlist[maxerr] - area12) <= abs(area12) * 1e-5 &&
						erro12 >= errmax * .99) {
					if (extrap)
						++iroff2;
					else /* if (! extrap) */
						++iroff1;
				}
				if (result.last > 10 && erro12 > errmax)
					++iroff3;
			}

			rlist[maxerr] = area1;
			rlist[result.last] = area2;
			errbnd = max(epsabs, epsrel * abs(area));

			/*           test for roundoff error and eventually set error flag. */

			if (iroff1 + iroff2 >= 10 || iroff3 >= 20)
				result.ier = 2;
			if (iroff2 >= 5)
				ierro = 3;

			/*           set error flag in the case that the number of
		     subintervals equals limit. */

			if (result.last == limit)
				result.ier = 1;

			/*           set error flag in the case of bad integrand behaviour
		     at some points of the integration range. */

			if (max(abs(a1), abs(b2)) <= (epmach * 100. + 1.) * (abs(a2) + uflow * 1e3)) {
				result.ier = 4;
			}

			/*           append the newly-created intervals to the list. */

			if (error2 <= error1) {
				alist[result.last] = a2;
				blist[maxerr] = b1;
				blist[result.last] = b2;
				elist[maxerr] = error1;
				elist[result.last] = error2;
			}
			else {
				alist[maxerr] = a2;
				alist[result.last] = a1;
				blist[result.last] = b1;
				rlist[maxerr] = area2;
				rlist[result.last] = area1;
				elist[maxerr] = error2;
				elist[result.last] = error1;
			}

			/*           call subroutine dqpsrt to maintain the descending ordering
		     in the list of error estimates and select the subinterval
		     with nrmax-th largest error estimate (to be bisected next). */

			//dqpsrt(limit, last, &maxerr, &errmax, elist, iord, &nrmax); // FIXME
			temp1[0] = errmax; temp4[0] = maxerr; temp5[0] = nrmax;
			dqpsrt(limit, result.last, temp4, temp1, elist, iord, temp5);
			errmax = temp1[0]; maxerr = temp4[0]; nrmax = temp5[0];
			if (errsum <= errbnd) {
				// goto L115;
				result.result = 0.;
				for (k = 1; k <= result.last; ++k)
					result.result += rlist[k];
				result.abserr = errsum;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
			if (result.ier != 0)	    break;
			if (result.last == 2) { /* L80: */
				small = .375;
				erlarg = errsum;
				ertest = errbnd;
				rlist2[1] = area; continue;
			}
			if (noext) 	    continue;

			erlarg -= erlast;
			if (abs(b1 - a1) > small) {
				erlarg += erro12;
			}
			if (!extrap) {

				/*           test whether the interval to be bisected next is the
		     smallest interval. */

				if (abs(blist[maxerr] - alist[maxerr]) > small) {
					continue;
				}
				extrap = true;
				nrmax = 2;
			}

			if (ierro != 3 && erlarg > ertest) {

				/*	    the smallest interval has the largest error.
		    before bisecting decrease the sum of the errors over the
		    larger intervals (erlarg) and perform extrapolation. */

				id = nrmax;
				jupbnd = result.last;
				if (result.last > limit / 2 + 2) {
					jupbnd = limit + 3 - result.last;
				}
				boolean cont = false;
				for (k = id; k <= jupbnd; ++k) {
					maxerr = iord[nrmax];
					errmax = elist[maxerr];
					if (abs(blist[maxerr] - alist[maxerr]) > small) {
						cont = true;
						break;
						//goto L90;
					}
					++nrmax;
					/* L50: */
				}
				if (cont) continue;
			}
			/*           perform extrapolation.  L60: */
			++numrl2;
			rlist2[numrl2 - 1] = area;
			temp4[0] = numrl2; temp5[0] = nres; temp1[0] = abseps;
			reseps = dqelg(temp4, rlist2, temp1, res3la, temp5);
			//dqelg(&numrl2, rlist2, &reseps, &abseps, res3la, &nres); // FIXME

			++ktmin;
			if (ktmin > 5 && result.abserr < errsum * .001) result.ier = 5;
//			if (abseps >= abserr) {
//				goto L70;
//			}
			if (abseps < result.abserr) {
				ktmin = 0;
				result.abserr = abseps;
				result.result = reseps;
				correc = erlarg;
				ertest = max(epsabs, epsrel * abs(reseps));
				if (result.abserr <= ertest)
					break;
			}

			/*            prepare bisection of the smallest interval. */

			//L70:
			if (numrl2 == 1) noext = true;
			if (result.ier == 5) break;
			maxerr = iord[1];
			errmax = elist[maxerr];
			nrmax = 1;
			extrap = false;
			small *= .5;
			erlarg = errsum;
			//L90: ;
		} // end for (last = 2; last <= limit; ++(last))

		/* L100:     set final result and error estimate. */
		/*	     ------------------------------------ */

		if (result.abserr == oflow) {
			// goto L115;
			result.result = 0.;
			for (k = 1; k <= result.last; ++k)
				result.result += rlist[k];
			result.abserr = errsum;
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}
		if (result.ier + ierro == 0) {
			//goto L110;
			if (ksgn == -1 && max(abs(result.result), abs(area)) <= deabs * .01) {
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
			if (.01 > result.result / area || result.result / area > 100. || errsum > abs(area)) {
				result.ier = 6;
			}
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}
		if (ierro == 3) {
			result.abserr += correc;
		}
		if (result.ier == 0) {
			result.ier = 3;
		}
		if (result.result == 0. || area == 0.) {
			if (result.abserr > errsum) {
				//goto L115;
				result.result = 0.;
				for (k = 1; k <= result.last; ++k)
					result.result += rlist[k];
				result.abserr = errsum;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}

			if (area == 0.) {
				//	goto L130;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
		}
		else { /* L105: */
			if (result.abserr / abs(result.result) > errsum / abs(area)) {
				//goto L115;
				result.result = 0.;
				for (k = 1; k <= result.last; ++k)
					result.result += rlist[k];
				result.abserr = errsum;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
		}

		/*           test on divergence */
		//L110:
		if (ksgn == -1 && max(abs(result.result), abs(area)) <= deabs * .01) {
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}
		if (.01 > result.result / area || result.result / area > 100. || errsum > abs(area)) {
			result.ier = 6;
		}
		result.neval = result.last * 30 - 15;
		if (inf == 2) result.neval <<= 1;
		if (result.ier > 2) --result.ier;
		return result;
	} /* rdqagie_ */


	/*
	 ***begin prologue  dqk15i
	 ***date written   800101   (yymmdd)
	 ***revision date  830518   (yymmdd)
	 ***category no.  h2a3a2,h2a4a2
	 ***keywords  15-point transformed gauss-kronrod rules
	 ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
      de doncker,elise,appl. math. & progr. div. - k.u.leuven
	 ***purpose  the original (infinite integration range is mapped
       onto the interval (0,1) and (a,b) is a part of (0,1).
       it is the purpose to compute
       i = integral of transformed integrand over (a,b),
       j = integral of abs(transformed integrand) over (a,b).
	 ***description

      integration rule
      standard fortran subroutine
      double precision version

      parameters
       on entry
         f      - double precision
                  fuction subprogram defining the integrand
                  function f(x). the actual name for f needs to be
                  declared e x t e r n a l in the calling program.

         boun   - double precision
                  finite bound of original integration
                  range (set to zero if inf = +2)

         inf    - int
                  if inf = -1, the original interval is
                              (-infinity,bound),
                  if inf = +1, the original interval is
                              (bound,+infinity),
                  if inf = +2, the original interval is
                              (-infinity,+infinity) and
                  the integral is computed as the sum of two
                  integrals, one over (-infinity,0) and one over
                  (0,+infinity).

         a      - double precision
                  lower limit for integration over subrange
                  of (0,1)

         b      - double precision
                  upper limit for integration over subrange
                  of (0,1)

       on return
         result - double precision
                  approximation to the integral i
                  result is computed by applying the 15-point
                  kronrod rule(resk) obtained by optimal addition
                  of abscissae to the 7-point gauss rule(resg).

         abserr - double precision
                  estimate of the modulus of the absolute error,
                  which should equal or exceed abs(i-result)

         resabs - double precision
                  approximation to the integral j

         resasc - double precision
                  approximation to the integral of
                  abs((transformed integrand)-i/(b-a)) over (a,b)

	 ***references  (none)
	 ***end prologue  dqk15i


      the abscissae and weights are supplied for the interval
      (-1,1).  because of symmetry only the positive abscissae and
      their corresponding weights are given.

      xgk    - abscissae of the 15-point kronrod rule
               xgk(2), xgk(4), ... abscissae of the 7-point
               gauss rule
               xgk(1), xgk(3), ...  abscissae which are optimally
               added to the 7-point gauss rule

      wgk    - weights of the 15-point kronrod rule

      wg     - weights of the 7-point gauss rule, corresponding
               to the abscissae xgk(2), xgk(4), ...
               wg(1), wg(3), ... are set to zero.





      list of major variables
      -----------------------

      centr  - mid point of the interval
      hlgth  - half-length of the interval
      absc*  - abscissa
      tabsc* - transformed abscissa
      fval*  - function value
      resg   - result of the 7-point gauss formula
      resk   - result of the 15-point kronrod formula
      reskh  - approximation to the mean value of the transformed
               integrand over (a,b), i.e. to i/(b-a)

      machine dependent constants
      ---------------------------

      epmach is the largest relative spacing.
      uflow is the smallest positive magnitude.
	 */
	static double dqk15i(UnivariateFunction f, double boun, int inf, double a, double b,
			double[] abserr, double[] resabs, double[] resasc)
	{
		/* Initialized data */

		final double wg[] = new double[] {
				0., .129484966168869693270611432679082,
				0., .27970539148927666790146777142378,
				0., .381830050505118944950369775488975,
				0., .417959183673469387755102040816327 };
		final double xgk[] = new double[] {
				.991455371120812639206854697526329,
				.949107912342758524526189684047851,
				.864864423359769072789712788640926,
				.741531185599394439863864773280788,
				.58608723546769113029414483825873,
				.405845151377397166906606412076961,
				.207784955007898467600689403773245, 0. };
		final double wgk[] = new double[] {
				.02293532201052922496373200805897,
				.063092092629978553290700663189204,
				.104790010322250183839876322541518,
				.140653259715525918745189590510238,
				.16900472663926790282658342659855,
				.190350578064785409913256402421014,
				.204432940075298892414161999234649,
				.209482141084727828012999174891714 };

		/* Local variables */
		double absc, dinf, resg, resk, fsum, absc1, absc2, fval1, fval2;
		int j;
		double hlgth, centr, reskh, uflow, result = 0;
		double tabsc1, tabsc2, fc, epmach;
		double[] fv1 = new double[7], fv2 = new double[7], vec = new double[15], vec2 = new double[15];


		/* ***first executable statement  dqk15i */
		epmach = DBL_EPSILON;
		uflow = DBL_MIN;
		dinf = (double) min(1, inf);

		centr = (a + b) * .5;
		hlgth = (b - a) * .5;
		tabsc1 = boun + dinf * (1. - centr) / centr;
		vec[0] = tabsc1;
		if (inf == 2) {
			vec2[0] = -tabsc1;
		}
		for (j = 1; j <= 7; ++j) {
			absc = hlgth * xgk[j - 1];
			absc1 = centr - absc;
			absc2 = centr + absc;
			tabsc1 = boun + dinf * (1. - absc1) / absc1;
			tabsc2 = boun + dinf * (1. - absc2) / absc2;
			vec[(j << 1) - 1] = tabsc1;
			vec[j * 2] = tabsc2;
			if (inf == 2) {
				vec2[(j << 1) - 1] = -tabsc1;
				vec2[j * 2] = -tabsc2;
			}
			/* L5: */
		}
		for (int i = 0; i < 15; i++)
			vec[i] = f.eval(vec[i]); /* -> new vec[] overwriting old vec[] */
		if (inf == 2) {
			for (int i = 0; i < 15; i++)
				vec2[i] = f.eval(vec2[i]); /* -> new vec[] overwriting old vec[] */
		}
		fval1 = vec[0];
		if (inf == 2) fval1 += vec2[0];
		fc = fval1 / centr / centr;

		/*           compute the 15-point kronrod approximation to
	     the integral, and estimate the error. */

		resg = wg[7] * fc;
		resk = wgk[7] * fc;
		resabs[0] = abs(resk);
		for (j = 1; j <= 7; ++j) {
			absc = hlgth * xgk[j - 1];
			absc1 = centr - absc;
			absc2 = centr + absc;
			tabsc1 = boun + dinf * (1. - absc1) / absc1;
			tabsc2 = boun + dinf * (1. - absc2) / absc2;
			fval1 = vec[(j << 1) - 1];
			fval2 = vec[j * 2];
			if (inf == 2) {
				fval1 += vec2[(j << 1) - 1];
			}
			if (inf == 2) {
				fval2 += vec2[j * 2];
			}
			fval1 = fval1 / absc1 / absc1;
			fval2 = fval2 / absc2 / absc2;
			fv1[j - 1] = fval1;
			fv2[j - 1] = fval2;
			fsum = fval1 + fval2;
			resg += wg[j - 1] * fsum;
			resk += wgk[j - 1] * fsum;
			resabs[0] += wgk[j - 1] * (abs(fval1) + abs(fval2));
			/* L10: */
		}
		reskh = resk * .5;
		resasc[0] = wgk[7] * abs(fc - reskh);
		for (j = 1; j <= 7; ++j) {
			resasc[0] += wgk[j - 1] * (abs(fv1[j - 1] - reskh) +
					abs(fv2[j - 1] - reskh));
			/* L20: */
		}
		result = resk * hlgth;
		resasc[0] *= hlgth;
		resabs[0] *= hlgth;
		abserr[0] = abs((resk - resg) * hlgth);
		if (resasc[0] != 0. && abserr[0] != 0.) {
			abserr[0] = resasc[0] * min(1., pow(abserr[0] * 200. / resasc[0], 1.5));
		}
		if (resabs[0] > uflow / (epmach * 50.)) {
			abserr[0] = max(epmach * 50. * resabs[0], abserr[0]);
		}
		return result;
	} /* dqk15i */

	/* ***begin prologue  dqpsrt
	 ***refer to  dqage,dqagie,dqagpe,dqawse
	 ***routines called  (none)
	 ***revision date  810101   (yymmdd)
	 ***keywords  sequential sorting
	 ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
       de doncker,elise,appl. math. & progr. div. - k.u.leuven
	 ***purpose  this routine maintains the descending ordering in the
        list of the local error estimated resulting from the
        interval subdivision process. at each call two error
        estimates are inserted using the sequential search
        method, top-down for the largest error estimate and
        bottom-up for the smallest error estimate.
	 ***description

       ordering routine
       standard fortran subroutine
       double precision version

       parameters (meaning at output)
          limit  - int
                   maximum number of error estimates the list
                   can contain

          last   - int
                   number of error estimates currently in the list

          maxerr - int
                   maxerr points to the nrmax-th largest error
                   estimate currently in the list

          ermax  - double precision
                   nrmax-th largest error estimate
                   ermax = elist(maxerr)

          elist  - double precision
                   vector of dimension last containing
                   the error estimates

          iord   - int
                   vector of dimension last, the first k elements
                   of which contain pointers to the error
                   estimates, such that
                   elist(iord(1)),...,  elist(iord(k))
                   form a decreasing sequence, with
                   k = last if last <= (limit/2+2), and
                   k = limit+1-last otherwise

          nrmax  - int
                   maxerr = iord(nrmax)

	 ***end prologue  dqpsrt
	 */
	static void dqpsrt(int limit, int last, int[] maxerr,
			double[] ermax, double[] elist, int[] iord, int[] nrmax)
	{
		/* Local variables */
		int i, j, k, ido, jbnd, isucc, jupbn;
		double errmin, errmax;

		/* Function Body */

		/*           check whether the list contains more than
	     two error estimates. */
		if (last <= 2) {
			iord[1] = 1;
			iord[2] = 2;
			//goto Last;
			maxerr[0] = iord[nrmax[0]];
			ermax[0] = elist[maxerr[0]];
			return;
		}
		/*           this part of the routine is only executed if, due to a
	     difficult integrand, subdivision increased the error
	     estimate. in the normal case the insert procedure should
	     start after the nrmax-th largest error estimate. */

		errmax = elist[maxerr[0]];
		if (nrmax[0] > 1) {
			ido = nrmax[0] - 1;
			for (i = 1; i <= ido; ++i) {
				isucc = iord[nrmax[0] - 1];
				if (errmax <= elist[isucc])
					break; /* out of for-loop */
				iord[nrmax[0]] = isucc;
				--(nrmax[0]);
				/* L20: */
			}
		}

		/*L30:       compute the number of elements in the list to be maintained
	     in descending order. this number depends on the number of
	     subdivisions still allowed. */
		if (last > limit / 2 + 2)
			jupbn = limit + 3 - last;
		else
			jupbn = last;

		errmin = elist[last];

		/*           insert errmax by traversing the list top-down,
	     starting comparison from the element elist(iord(nrmax+1)). */

		jbnd = jupbn - 1;
		for (i = nrmax[0] + 1; i <= jbnd; ++i) {
			isucc = iord[i];
			if (errmax >= elist[isucc]) {/* ***jump out of do-loop */
				/* L60: insert errmin by traversing the list bottom-up. */
				iord[i - 1] = maxerr[0];
				for (j = i, k = jbnd; j <= jbnd; j++, k--) {
					isucc = iord[k];
					if (errmin < elist[isucc]) {
						/* goto L80; ***jump out of do-loop */
						iord[k + 1] = last;
						// goto Last;
						maxerr[0] = iord[nrmax[0]];
						ermax[0] = elist[maxerr[0]];
						return;
					}
					iord[k + 1] = isucc;
				}
				iord[i] = last;
				//goto Last;
				maxerr[0] = iord[nrmax[0]];
				ermax[0] = elist[maxerr[0]];
				return;
			}
			iord[i - 1] = isucc;
		}

		iord[jbnd] = maxerr[0];
		iord[jupbn] = last;
		maxerr[0] = iord[nrmax[0]];
		ermax[0] = elist[maxerr[0]];
		return;
	} /* dqpsrt_ */

	/* ***begin prologue  dqelg
	***refer to  dqagie,dqagoe,dqagpe,dqagse
	***revision date  830518   (yymmdd)
	***keywords  epsilon algorithm, convergence acceleration,
	            extrapolation
	***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
	          de doncker,elise,appl. math & progr. div. - k.u.leuven
	***purpose  the routine determines the limit of a given sequence of
	           approximations, by means of the epsilon algorithm of
	           p.wynn. an estimate of the absolute error is also given.
	           the condensed epsilon table is computed. only those
	           elements needed for the computation of the next diagonal
	           are preserved.
	***description

	          epsilon algorithm
	          standard fortran subroutine
	          double precision version

	          parameters
	             n      - int
	                      epstab(n) contains the new element in the
	                      first column of the epsilon table.

	             epstab - double precision
	                      vector of dimension 52 containing the elements
	                      of the two lower diagonals of the triangular
	                      epsilon table. the elements are numbered
	                      starting at the right-hand corner of the
	                      triangle.

	             result - double precision
	                      resulting approximation to the integral

	             abserr - double precision
	                      estimate of the absolute error computed from
	                      result and the 3 previous results

	             res3la - double precision
	                      vector of dimension 3 containing the last 3
	                      results

	             nres   - int
	                      number of calls to the routine
	                      (should be zero at first call)

	***end prologue  dqelg


	          list of major variables
	          -----------------------

	          e0     - the 4 elements on which the computation of a new
	          e1       element in the epsilon table is based
	          e2
	          e3                 e0
	                       e3    e1    new
	                             e2

	          newelm - number of elements to be computed in the new diagonal
	          errA   - errA = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
	          result - the element in the new diagonal with least value of errA

	          machine dependent constants
	          ---------------------------

	          epmach is the largest relative spacing.
	          oflow is the largest positive magnitude.
	          limexp is the maximum number of elements the epsilon
	          table can contain. if this number is reached, the upper
	          diagonal of the epsilon table is deleted. */
	static final double dqelg(int[] n, double[] epstab, double[] abserr, double[] res3la, int[] nres)
	{
	    /* Local variables */
	    int i__, indx, ib, ib2, ie, k1, k2, k3, num, newelm, limexp;
	    double delta1, delta2, delta3, e0, e1, e1abs, e2, e3, epmach, epsinf;
	    double oflow, ss, res;
	    double errA, err1, err2, err3, tol1, tol2, tol3, result;

	    /* Function Body */
		/* ***first executable statement  dqelg */
	    epmach = DBL_EPSILON;
	    oflow = Double.MAX_VALUE;
	    ++(nres[0]);
	    abserr[0] = oflow;
	    result = epstab[n[0]];
	    if (n[0] < 3) {
	    	//goto L100;
		    abserr[0] = max(abserr[0], epmach * 5. * abs(result));
		    return result;
	    }
	    limexp = 50;
	    epstab[n[0] + 2] = epstab[n[0]];
	    newelm = (n[0] - 1) / 2;
	    epstab[n[0]] = oflow;
	    num = n[0];
	    k1 = n[0];
	    for (i__ = 1; i__ <= newelm; ++i__) {
	    	k2 = k1 - 1;
	    	k3 = k1 - 2;
	    	res = epstab[k1 + 2];
	    	e0 = epstab[k3];
	    	e1 = epstab[k2];
	    	e2 = res;
	    	e1abs = abs(e1);
	    	delta2 = e2 - e1;
	    	err2 = abs(delta2);
	    	tol2 = max(abs(e2), e1abs) * epmach;
	    	delta3 = e1 - e0;
	    	err3 = abs(delta3);
	    	tol3 = max(e1abs, abs(e0)) * epmach;
	    	if (err2 <= tol2 && err3 <= tol3) {
	    		/*           if e0, e1 and e2 are equal to within machine
				 accuracy, convergence is assumed. */
	    		result = res;/*		result = e2 */
	    		abserr[0] = err2 + err3;/*	abserr = abs(e1-e0)+abs(e2-e1) */

	    		// goto L100;	/* ***jump out of do-loop */
	    		abserr[0] = max(abserr[0], epmach * 5. * abs(result));
	    		return result;
	    	}

	    	e3 = epstab[k1];
	    	epstab[k1] = e1;
	    	delta1 = e1 - e3;
	    	err1 = abs(delta1);
	    	tol1 = max(e1abs, abs(e3)) * epmach;

	    	/*           if two elements are very close to each other, omit
		     a part of the table by adjusting the value of n */

	    	if (err1 > tol1 && err2 > tol2 && err3 > tol3) {
	    		ss = 1. / delta1 + 1. / delta2 - 1. / delta3;
	    		epsinf = abs(ss * e1);

	    		/*           test to detect irregular behaviour in the table, and
		     eventually omit a part of the table adjusting the value of n. */

	    		if (epsinf > 1e-4) {
	    			//goto L30;
	    			/* compute a new element and eventually adjust the value of result. */
		    		res = e1 + 1. / ss;
			    	epstab[k1] = res;
			    	k1 += -2;
			    	errA = err2 + abs(res - e2) + err3;
			    	if (errA <= abserr[0]) {
			    		abserr[0] = errA;
			    		result = res;
			    	}
			    	continue;
	    		}
	    	}

	    	n[0] = i__ + i__ - 1;
	    	//goto L50;/* ***jump out of do-loop */
	    	break;
	    }

	    /*           shift the table. */

	    //L50:
	    if (n[0] == limexp) {
	    	n[0] = (limexp / 2 << 1) - 1;
	    }

	    if (num / 2 << 1 == num) ib = 2; else ib = 1;
	    ie = newelm + 1;
	    for (i__ = 1; i__ <= ie; ++i__) {
	    	ib2 = ib + 2;
	    	epstab[ib] = epstab[ib2];
	    	ib = ib2;
	    }
	    if (num != n[0]) {
	    	indx = num - n[0] + 1;
	    	for (i__ = 1; i__ <= n[0]; ++i__) {
	    		epstab[i__] = epstab[indx];
	    		++indx;
	    	}
	    }
	    /*L80:*/
	    if (nres[0] >= 4) {
	    	/* L90: */
	    	abserr[0] = abs(result - res3la[3]) +
	    			abs(result - res3la[2]) +
	    			abs(result - res3la[1]);
	    	res3la[1] = res3la[2];
	    	res3la[2] = res3la[3];
	    	res3la[3] = result;
	    } else {
	    	res3la[nres[0]] = result;
	    	abserr[0] = oflow;
	    }

	    // L100:/* compute error estimate */
	    abserr[0] = max(abserr[0], epmach * 5. * abs(result));
	    return result;
	} /* dqelg */

	public static void main(String[] args) {
		UnivariateFunction f = new UnivariateFunction() {
			public void setParameters(double... params) {}
			public void setObjects(Object... obj) {}
			public double eval(double x) {
				return Normal.density(x, 0, 1, false);
			}
		};
		IntegrationResult result = dqagie(f, 0, 1, DBL_EPSILON*64, DBL_EPSILON*64, 100);
		System.out.println(result.result);
		System.out.println("Error = " + result.abserr); // Should be precisely 0.5

		f = new UnivariateFunction() {
			public void setParameters(double... params) {}
			public void setObjects(Object... obj) {}
			public double eval(double x) {
				return 1/((x+1) * sqrt(x));
			}
		};
		result = dqagie(f, 0, 1, DBL_EPSILON*64, DBL_EPSILON*64, 100);
		System.out.println(result.result);
		System.out.println("Error = " + result.abserr); // Should be precisely pi
	}
}
