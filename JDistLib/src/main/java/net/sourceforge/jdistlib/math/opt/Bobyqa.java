/*
 * Roby Joehanes
 *  
 * Copyright 2012 Roby Joehanes
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
package net.sourceforge.jdistlib.math.opt;

import static java.lang.Math.*;
import net.sourceforge.jdistlib.math.MultivariableFunction;

/**
 * Translation of the infamous Bobyqa algorithm by Michael J. D. Powell. Original Fortran code
 * can be found <a href="http://plato.asu.edu/ftp/other_software/bobyqa.zip">here</a>.
 * Paper is <a href="http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf">here</a>
 * 
 * @author Roby Joehanes
 *
 */
public class Bobyqa extends MultivariateOptimization {

	public OptimizationResult optimize(OptimizationConfig cfg)
	{
		BobyqaConfig bcfg = null;
		if (cfg instanceof BobyqaConfig) {
			bcfg = (BobyqaConfig) cfg;
		} else {
			bcfg = new BobyqaConfig(cfg);
		}
		return bobyqa(bcfg.initialGuess, bcfg.lowerBound, bcfg.upperBound,
			bcfg.objectiveFunction, bcfg.maxNumFunctionCall, bcfg.initialTrustRegionRadius,
			bcfg.stoppingTrustRegionRadius, bcfg.numInterpolationPoints, bcfg.isMinimize);
	}

	/*
	private static String toString(double[] value) {
		if (value == null) return "-";
		int n = value.length;
		StringBuilder buf = new StringBuilder();
		buf.append(value[0]);
		for (int i = 1; i < n; i++)
			buf.append(", " + value[i]);
		return buf.toString();
	}
	private static String toString(double[][] value) {
		if (value == null) return "-";
		StringBuilder buffer = new StringBuilder();
		int numValues = value.length;
		for (int i = 0; i < numValues; i++)
			buffer.append(toString(value[i]) + "\n");
		return buffer.toString();
	}

	static java.io.PrintWriter writer = null;
	static {
		try {
			writer = new java.io.PrintWriter("C:\\Users\\robbyjo\\Desktop\\mybobyqa.txt");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	static final void printState(String msg, int nf, double[] xbase, double[] xopt, double[] fval, double[] vlag, double[] d, double[][] bmat, double[][] zmat, double[][] xpt)
	{
		writer.println(msg);
		writer.println("nf: " + nf);
		writer.println("xbase: " + toString(xbase));
		writer.println("xopt: " + toString(xopt));
		writer.println("fval: " + toString(fval));
		writer.println("vlag: " + toString(vlag));
		writer.println("d: " + toString(d));
		writer.println("bmat: " + toString(bmat));
		writer.println("zmat: " + toString(zmat));
		writer.println("xpt: " + toString(xpt));
		writer.flush();
	}
	//*/

	/**
	 * <pre>
	 * This subroutine seeks the least value of a function of many variables,
	 * by applying a trust region method that forms quadratic models by
	 * interpolation. There is usually some freedom in the interpolation
	 * conditions, which is taken up by minimizing the Frobenius norm of
	 * the change to the second derivative of the model, beginning with the
	 * zero matrix. The values of the variables are constrained by upper and
	 * lower bounds. The arguments of the subroutine are as follows.
	 * 
	 * N must be set to the number of variables and must be at least two.
	 * NPT is the number of interpolation conditions. Its value must be in
	 *   the interval [N+2,(N+1)(N+2)/2]. Choices that exceed 2*N+1 are not
	 *   recommended.
	 * Initial values of the variables must be set in X(1),X(2),...,X(N). They
	 *   will be changed to the values that give the least calculated F.
	 * For I=1,2,...,N, XL(I) and XU(I) must provide the lower and upper
	 *   bounds, respectively, on X(I). The construction of quadratic models
	 *   requires XL(I) to be strictly less than XU(I) for each I. Further,
	 *   the contribution to a model from changes to the I-th variable is
	 *   damaged severely by rounding errors if XU(I)-XL(I) is too small.
	 * RHOBEG and RHOEND must be set to the initial and final values of a trust
	 *   region radius, so both must be positive with RHOEND no greater than
	 *   RHOBEG. Typically, RHOBEG should be about one tenth of the greatest
	 *   expected change to a variable, while RHOEND should indicate the
	 *   accuracy that is required in the final values of the variables. An
	 *   error return occurs if any of the differences XU(I)-XL(I), I=1,...,N,
	 *   is less than 2*RHOBEG.
	 * The value of IPRINT should be set to 0, 1, 2 or 3, which controls the
	 *   amount of printing. Specifically, there is no output if IPRINT=0 and
	 *   there is output only at the return if IPRINT=1. Otherwise, each new
	 *   value of RHO is printed, with the best vector of variables so far and
	 *   the corresponding value of the objective function. Further, each new
	 *   value of F with its variables are output if IPRINT=3.
	 * MAXFUN must be set to an upper bound on the number of calls of CALFUN.
	 * The array W will be used for working space. Its length must be at least
	 *   (NPT+5)*(NPT+N)+3*N*(N+5)/2.
	 *   
	 * DOUBLE PRECISION FUNCTION CALFUN (N,X,IP) has to be provided by
	 * the user. It returns the value of the objective function for
	 * the current values of the variables X(1),X(2),...,X(N), which are
	 * generated automatically in a way that satisfies the bounds given
	 * in XL and XU.
	 * 
	 * Return if the value of NPT is unacceptable.
	 * </pre>
	 * <P>RJ's note: Storage bound [3.5*n^2 + 23.5*n + 14, (n^4 + 35*n^2)/4 + 2*n^3 + 24*n + 6]
	 * @param x Initial estimate
	 * @param xl Lower estimate
	 * @param xu Upper estimate
	 * @param calfun Function to optimize
	 * @param npt Number of interpolation points
	 * @param rhobeg Initial trust region radius
	 * @param rhoend Final trust region radius
	 * @param maxfun Maximum number of calls to calfun
	 * @param isMinimize true if goal is to find the global minimum
	 */
	public static final OptimizationResult bobyqa(double[] x, double[] xl, double[] xu, MultivariableFunction calfun,
		int npt, double rhobeg, double rhoend, int maxfun, boolean isMinimize)
	{
		int
			n = x.length,
			np = n + 1,
			ndim = npt + n;
		final int toRescue = 190;
		if (xl.length != n || xu.length != n) // sanity check
			throw new RuntimeException();
		if (npt < n + 2 || npt > ((n + 2) * np) / 2)
			throw new RuntimeException("Return from BOBYQA because NPT is not in the required interval");
		double
			origx[] = x, // Store the original object since the object will be used for the storage of the result
			xbase[] = new double[n],
			xpt[][] = new double[npt][n],
			fval[] = new double[npt],
			xopt[] = new double[n],
			gopt[] = new double[n],
			hq[] = new double[n * np/2],
			pq[] = new double[npt],
			bmat[][] = new double[ndim][n],
			zmat[][] = new double[npt][npt-np],
			sl[] = new double[n],
			su[] = new double[n],
			xnew[] = new double[n],
			xalt[] = new double[n],
			d[] = new double[n],
			vlag[] = new double[ndim],
			work1[] = new double[n],
			work2[] = new double[npt],
			work3[] = new double[npt];
		x = new double[n]; // Make a new array and copy it
		System.arraycopy(origx, 0, x, 0, n);
		double two_rhobeg = 2 * rhobeg;
		// int isl = 0 + n + n * npt + npt + n + n + (n*np) / 2 + npt + ndim * n + npt * (npt - np);
		//int iw = (0 + n + n * npt + npt + n + n + (n*np) / 2 + npt + ndim * n + npt * (npt - np)) + 5*n + ndim;
		for (int j = 0; j < n; j++)
		{
			double temp = xu[j] - xl[j];
			if (temp < two_rhobeg)
				throw new RuntimeException("Return from BOBYQA because one of the differences xu[i]-xl[i] is less than 2*RHOBEG");
			sl[j] = xl[j] - x[j];
			su[j] = xu[j] - x[j];
			if (sl[j] >= -rhobeg)
			{
				if (sl[j] >= 0)
				{
					x[j] = xl[j];
					sl[j] = 0;
					su[j] = temp;
				} else {
					x[j] = xl[j] + rhobeg;
					sl[j] = -rhobeg;
					su[j] = max(xu[j] - x[j], rhobeg);
				}
			} else if (su[j] <= rhobeg) {
				if (su[j] <= 0)
				{
					x[j] = xu[j];
					sl[j] = -temp;
					su[j] = 0;
				} else {
					x[j] = xu[j] - rhobeg;
					sl[j] = min(xl[j] - x[j], -rhobeg);
					su[j] = rhobeg;
				}
			}
		}
		//bobyqb(x, xl, xu, calfun, rhobeg, rhoend, iprint, maxfun, xbase, xpt, fval,
		//	xopt, gopt, hq, pq, bmat, zmat, sl, su, xnew, xalt, d, vlag);
		
		/*
		 * The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
		 * BMAT and ZMAT for the first iteration, with the corresponding values of
		 * of NF and KOPT, which are the number of calls of CALFUN so far and the
		 * index of the interpolation point at the trust region centre. Then the
		 * initial XOPT is set too. The branch to label 720 occurs if MAXFUN is
		 * less than NPT. GOPT will be updated if KOPT is different from KBASE.
		 */
		//int n = x.length, npt = xpt.length, np = n + 1;
		//double[] w = new double[3 * ndim];
		double[] gnew = new double[n];
		int nf = 0, kopt = 0, nptm = npt - np, nh = n * np / 2;
		{
			int[] temp = prelim(x, xl, xu, calfun, rhobeg, maxfun, xbase, xpt, fval, gopt, hq, pq, bmat, zmat, sl, su, isMinimize);
			nf = temp[0]; kopt = temp[1];
		}
		double
			xoptsq = 0,
			fsave = fval[0],
			alpha = 0,
			cauchy = 0,
			adelt = 0,
			beta = 0,
			f = 0,
			denom = 0,
			den = 0,
			ratio = 0;
		boolean loop = true;
		for (int i = 0; i < n; i++)
		{
			double val = xopt[i] = xpt[kopt][i];
			xoptsq += val * val;
		}
		if (nf < npt)
			loop = false;
		int kbase = 0, knew = -1;
		// Complete the settings that are required for the iterative procedure.
		double
			rho = rhobeg,
			delta = rho,
			diffa = 0,
			diffb = 0,
			diffc = 0;
		int nresc = nf, ntrits = 0, nfsav = nf, itest = 0;
		int label = 20;
		double
			dsq = 0,
			crvmin = 0,
			dnorm = 0,
			distsq = 0;
		while(loop) switch(label)
		{
		case 20:
			/*
			 * Update GOPT if necessary before the first iteration and after each
			 * call of RESCUE that makes a call of CALFUN.
			 */
			//printState("bobyqb-20-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			if (kopt != kbase)
			{
				int ih = 0;
				for (int j = 0; j < n; j++)
					for (int i = 0; i <= j; i++)
					{
						if (i < j) gopt[j] = gopt[j] + hq[ih] * xopt[i];
						gopt[i] = gopt[i] + hq[ih] * xopt[j];
						ih++;
					}
				if (nf > npt)
				{
					for (int k = 0; k < npt; k++)
					{
						double temp = 0;
						for (int j = 0; j < n; j++)
							temp += xpt[k][j] * xopt[j];
						temp = pq[k] * temp;
						for (int i = 0; i < n; i++)
							gopt[i] = gopt[i] + temp * xpt[k][i];
					}
				}
			}
			label = 60;
			break;
		case 60:
			//printState("bobyqb-60-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			//printState("bobyqb-60-start-checkstate\n"+
			//	"delta="+delta+", dnorm="+dnorm+
			//	"distsq="+distsq, 0, null, null, null, null, null, null, null, null);
			{
				// trsbox(double[][] xpt, double[] xopt, double[] gopt, double[] hq, double[] pq,
				// double[] sl, double[] su, double delta, double[] xnew, double[] d, double[] gnew,
				// double dsq, double crvmin)
				double[] temp = trsbox(xpt, xopt, gopt, hq, pq, sl, su, delta, xnew, d, gnew, dsq, crvmin);
				dsq = temp[0]; crvmin = temp[1];
			}
			//printState("bobyqb-60-aftertrsbox-checkstate\n"+
			//	"delta="+delta+", dnorm="+dnorm+
			//	"distsq="+distsq, 0, null, null, null, null, null, null, null, null);
			dnorm = min(delta, sqrt(dsq));
			if (dnorm < 0.5 * rho)
			{
				ntrits = -1;
				distsq = 100*rho*rho;
				if (nf <= nfsav + 2)
				{
					label = 650;
					break;
				}
				/*
				 * The following choice between labels 650 and 680 depends on whether or
				 * not our work with the current RHO seems to be complete. Either RHO is
				 * decreased or termination occurs if the errors in the quadratic model at
				 * the last three interpolation points compare favourably with predictions
				 * of likely improvements to the model within distance HALF*RHO of XOPT.
				 */
				double errbig = max(max(diffa, diffb), diffc);
				if (crvmin > 0 && errbig > (0.125 * rho * rho) * crvmin)
				{
					label = 650; break; // goto 650
				}
				/*
				 * Due to the precision problem below (see my comments), I cannot verify
				 * this particular branch numerically with that of the FORTRAN code.
				 * However, test runs looks okay. -- RJ
				 */
				double bdtol = errbig / rho;
				for (int j = 0; j < n; j++)
				{
					double bdtest = bdtol;
					//if (xnew[j] == sl[j]) bdtest = gnew[j];
					//if (xnew[j] == su[j]) bdtest = -gnew[j];
					if (xnew[j] == sl[j]) bdtest = work1[j];
					if (xnew[j] == su[j]) bdtest = -work1[j];
					if (bdtest < bdtol)
					{
						int jp1 = j+1;
						double curv = hq[(jp1 + jp1*jp1) / 2 - 1];
						for (int k = 0; k < npt; k++)
						{
							double val = xpt[k][j];
							curv += pq[k] * val * val;
						}
						bdtest += 0.5 * curv * rho;
						if (bdtest < bdtol)
						{
							label = 650; break; // goto 650
						}
					}
				}
				label = 680; break; // goto 680
			} // end if (dnorm < 0.5 * rho)
			ntrits++;
			label = 90;
			break;
		case 90:
			/*
			 * Severe cancellation is likely to occur if XOPT is too far from XBASE.
			 * If the following test holds, then XBASE is shifted so that XOPT becomes
			 * zero. The appropriate changes are made to BMAT and to the second
			 * derivatives of the current model, beginning with the changes to BMAT
			 * that do not depend on ZMAT. VLAG is used temporarily for working space.
			 */
			//printState("bobyqb-90-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			if (dsq <= 1e-3 * xoptsq)
			{
				double
					fracsq = 0.25 * xoptsq,
					sumpq = 0;
				for (int k = 0; k < npt; k++)
				{
					sumpq += pq[k];
					double sum = -0.5 * xoptsq;
					for (int i = 0; i < n; i++)
						sum += xpt[k][i] * xopt[i];
					//w[npt+k] = sum;
					work2[k] = sum;
					double temp = fracsq - 0.5 * sum;
					for (int i = 0; i < n; i++)
					{
						//w[i] = bmat[k][i];
						work1[i] = bmat[k][i];
						vlag[i] = sum * xpt[k][i] + temp * xopt[i];
						int ip = npt + i;
						for (int j = 0; j <= i; j++)
							//bmat[ip][j] += w[i] * vlag[j] + vlag[i] * w[j];
							bmat[ip][j] = bmat[ip][j] + work1[i] * vlag[j] + vlag[i] * work1[j];
					}
				}
				// Then the revisions of BMAT that depend on ZMAT are calculated.
				for (int jj = 0; jj < nptm; jj++) // L110
				{
					double sumz = 0, sumw = 0;
					for (int k = 0; k < npt; k++)
					{
						sumz = sumz + zmat[k][jj];
						//vlag[k] = w[npt+k] * zmat[k][jj];
						vlag[k] = work2[k] * zmat[k][jj];
						sumw = sumw + vlag[k];
					}
					/*
					 * TODO This is the place where minor precision difference may start to take toll.
					 * Whether this makes any real difference in the long run or not is unknown.
					 * However, it looks like it's okay. -- RJ's note
					 * Fortran run:
					 *  JJ=1, SUMW= -5.10633202388532936E-014, SUMZ= -5.41788836017076392E-014
					 *  JJ=2, SUMW= -3.49720252756924310E-015, SUMZ= -5.40123501480138657E-014
					 *  JJ=3, SUMW= -1.11022302462515654E-016, SUMZ=  1.79856129989275360E-014
					 *  JJ=4, SUMW= -1.92068583260152081E-014, SUMZ= -4.70734562441066373E-014
					 *  JJ=5, SUMW= -2.10942374678779743E-015, SUMZ=  4.44089209850062616E-016
					 * 
					 * Java run:
					 *  jj=0, sumw= -4.445749324233361E-14, sumz= -3.1086244689504383E-15
					 *  jj=1, sumw=-9.547918011776346E-15, sumz=5.773159728050814E-15
					 *  jj=2, sumw=-7.105427357601002E-15, sumz=-5.773159728050814E-15
					 *  jj=3, sumw=8.215650382226158E-15, sumz=3.352873534367973E-14
					 *  jj=4, sumw=8.881784197001252E-15, sumz=8.215650382226158E-15
					 */
					//System.out.println("jj="+jj+", sumw="+sumw+", sumz="+sumz);
					for (int j = 0; j < n; j++)
					{
						double sum = (fracsq * sumz - 0.5 * sumw) * xopt[j];
						for (int k = 0; k < npt; k++)
							sum += vlag[k] * xpt[k][j];
						//w[j] = sum;
						work1[j] = sum;
						for (int k = 0; k < npt; k++)
							bmat[k][j] = bmat[k][j] + sum * zmat[k][jj];
					}
					for (int i = 0; i < n; i++)
					{
						int ip = i + npt;
						//double temp = w[i];
						double temp = work1[i];
						for (int j = 0; j <= i; j++)
							bmat[ip][j] = bmat[ip][j] + temp * work1[j];
							//bmat[ip][j] += temp * w[j];
					}
				}
				/*
				 * The following instructions complete the shift, including the changes
				 * to the second derivative parameters of the quadratic model.
				 */
				int ih = 0; // L150
				for (int j = 0; j < n; j++)
				{
					//w[j] = -0.5 * sumpq * xopt[j];
					work1[j] = -0.5 * sumpq * xopt[j];
					for (int k = 0; k < npt; k++)
					{
						//w[j] += pq[k] * xpt[k][j];
						work1[j] = work1[j] + pq[k] * xpt[k][j];
						xpt[k][j] = xpt[k][j] - xopt[j];
					}
					for (int i = 0; i <= j; i++)
					{
						//hq[ih] += w[i] * xopt[j] + xopt[i] * w[j];
						hq[ih] = hq[ih] + work1[i] * xopt[j] + xopt[i] * work1[j];
						bmat[npt+i][j] = bmat[npt+j][i];
						ih++;
					}
				}
				for (int i = 0; i < n; i++) // L180
				{
					double xopti = xopt[i];
					xbase[i] = xbase[i] + xopti;
					xnew[i] = xnew[i] - xopti;
					sl[i] = sl[i] - xopti;
					su[i] = su[i] - xopti;
					xopt[i] = 0;
				}
				xoptsq = 0;
			} // end if (dsq <= 1e-3 * xoptsq)
			label = ntrits == 0 ? 210 : 230; // if (ntrits == 0) goto 210 else goto 230;
			break;
		case toRescue:
			/*
			 * XBASE is also moved to XOPT by a call of RESCUE. This calculation is
			 * more expensive than the previous shift, because new matrices BMAT and
			 * ZMAT are generated from scratch, which may include the replacement of
			 * interpolation points whose positions seem to be causing near linear
			 * dependence in the interpolation conditions. Therefore RESCUE is called
			 * only if rounding errors have reduced by at least a factor of two the
			 * denominator of the formula for updating the H matrix. It provides a
			 * useful safeguard, but is not invoked in most applications of BOBYQA.
			 */
			// TODO Untested branch
			nfsav = nf;
			kbase = kopt;
			{
				int[] temp = rescue(xl, xu, calfun, maxfun, xbase, xpt, fval, xopt, gopt, hq, pq,
					bmat, zmat, sl, su, nf, delta, kopt, vlag, isMinimize);
				nf = temp[0]; kopt = temp[1];
			}
			/*
			 * XOPT is updated now in case the branch below to label 720 is taken.
			 * Any updating of GOPT occurs after the branch below to label 20, which
			 * leads to a trust region iteration as does the branch to label 60.
			 */
			xoptsq = 0;
			if (kopt != kbase)
			{
				for (int i = 0; i < n; i++)
				{
					double val = xopt[i] = xpt[kopt][i];
					xoptsq += val * val;
				}
			}
			if (nf < 0)
			{
				nf = maxfun; // Already called maxfun times
				loop = false; label = 720; break; // goto 720
			}
			nresc = nf;
			if (nfsav < nf)
			{
				nfsav = nf;
				label = 20; break; // goto 20
			}
			if (ntrits > 0)
			{
				label = 60; break; // goto 60
			}
			label = 210; // goto 210
			break;
		case 210:
			/*
			 * Pick two alternative vectors of variables, relative to XBASE, that
			 * are suitable as new positions of the KNEW-th interpolation point.
			 * Firstly, XNEW is set to the point on a line through XOPT and another
			 * interpolation point that minimizes the predicted value of the next
			 * denominator, subject to ||XNEW - XOPT|| .LEQ. ADELT and to the SL
			 * and SU bounds. Secondly, XALT is set to the best feasible point on
			 * a constrained version of the Cauchy step of the KNEW-th Lagrange
			 * function, the corresponding value of the square of this function
			 * being returned in CAUCHY. The choice between these alternatives is
			 * going to be made when the denominator is calculated.
			 */
			//printState("bobyqb-210-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			{
				double[] temp = altmov(xpt, xopt, bmat, zmat, sl, su, kopt, knew, adelt, xnew, xalt);
				alpha = temp[0]; cauchy = temp[1];
			}
			for (int i = 0; i < n; i++)
				d[i] = xnew[i] - xopt[i];
			label = 230;
			break;
		case 230:
			/*
			 * Calculate VLAG and BETA for the current choice of D. The scalar
			 * product of D with XPT(K,.) is going to be held in W(NPT+K) for
			 * use when VQUAD is calculated.
			 */
			//printState("bobyqb-230-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			for (int k = 0; k < npt; k++)
			{
				double suma = 0, sumb = 0, sum = 0;
				for (int j = 0; j < n; j++)
				{
					suma += xpt[k][j] * d[j];
					sumb += xpt[k][j] * xopt[j];
					sum += bmat[k][j] * d[j];
				}
				//w[k] = suma * (0.5 * suma + sumb);
				work3[k] = suma * (0.5 * suma + sumb);
				vlag[k] = sum;
				//w[npt+k] = suma;
				work2[k] = suma;
			}
			beta = 0;
			for (int jj = 0; jj < nptm; jj++)
			{
				double sum = 0;
				for (int k = 0; k < npt; k++)
					sum += zmat[k][jj] * work3[k];
					//sum += zmat[k][jj] * w[k];
				beta -= sum * sum;
				for (int k = 0; k < npt; k++)
					vlag[k] = vlag[k] + sum * zmat[k][jj];
			}
			dsq = 0;
			double bsum = 0, dx = 0;
			for (int j = 0; j < n; j++)
			{
				double val = d[j];
				dsq += val * val;
				double sum = 0;
				for (int k = 0; k < npt; k++)
					sum += work3[k] * bmat[k][j];
					//sum += w[k] * bmat[k][j];
				bsum += sum * d[j];
				int jp = npt + j;
				for (int i = 0; i < n; i++)
					sum += bmat[jp][i] * d[i];
				vlag[jp] = sum;
				bsum += sum * d[j];
				dx += d[j] * xopt[j];
			}
			beta = dx * dx + dsq * (xoptsq + dx + dx + 0.5*dsq) + beta - bsum;
			vlag[kopt] = vlag[kopt] + 1;
			/*
			 * If NTRITS is zero, the denominator may be increased by replacing
			 * the step D of ALTMOV by a Cauchy step. Then RESCUE may be called if
			 * rounding errors have damaged the chosen denominator.
			 */
			if (ntrits == 0)
			{
				double val = vlag[knew];
				denom = val * val + alpha * beta;
				if (denom < cauchy && cauchy > 0)
				{
					for (int i = 0; i < n; i++)
					{
						xnew[i] = xalt[i];
						d[i] = xnew[i] - xopt[i];
					}
					cauchy = 0;
					label = 230; break; // goto 230
				}
				if (denom <= 0.5 * val * val)
				{
					if (nf > nresc)
					{
						label = toRescue; break; // goto 190
					}
					loop = false; label = 720; break; // goto 720
				}
			} else {
				/*
				 * Alternatively, if NTRITS is positive, then set KNEW to the index of
				 * the next interpolation point to be deleted to make room for a trust
				 * region step. Again RESCUE may be called if rounding errors have damaged
				 * the chosen denominator, which is the reason for attempting to select
				 * KNEW before calculating the next value of the objective function.
				 */
				double
					delsq = delta * delta,
					scaden = 0,
					biglsq = 0;
				knew = -1;
				for (int k = 0; k < npt; k++)
				{
					if (k == kopt) continue;
					double hdiag = 0, val;
					for (int jj = 0; jj < nptm; jj++)
					{
						val = zmat[k][jj];
						hdiag += val * val;
					}
					val = vlag[k];
					den = beta * hdiag + val * val;
					distsq = 0;
					for (int j = 0; j < n; j++)
					{
						val = xpt[k][j] - xopt[j];
						distsq += val * val;
					}
					val = distsq / delsq;
					double temp = max(1, val * val);
					if (temp * den > scaden)
					{
						scaden = temp * den;
						knew = k;
						denom = den;
					}
					val = vlag[k];
					biglsq = max(biglsq, temp * val * val);
				}
				if (scaden <= 0.5 * biglsq)
				{
					if (nf > nresc)
					{
						label = toRescue; break; // goto 190
					}
					loop = false; label = 720; break; // goto 720
				}
			} // end if (ntrits == 0)
			label = 360;
			break;
		case 360:
			//printState("bobyqb-360-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			for (int i = 0; i < n; i++)
			{
				x[i] = min(max(xl[i], xbase[i]+xnew[i]), xu[i]);
				if (xnew[i] == sl[i]) x[i] = xl[i];
				if (xnew[i] == su[i]) x[i] = xu[i];
			}
			if (nf >= maxfun)
			{
				loop = false; label = 720; break; // goto 720
			}
			nf++;
			f = isMinimize ? calfun.eval(x) : -calfun.eval(x);
			if (ntrits == -1)
			{
				fsave = f;
				loop = false; label = 720; break; // goto 720
			}
			/*
			 * Use the quadratic model to predict the change in F due to the step D,
			 * and set DIFF to the error of this prediction.
			 */
			double
				fopt = fval[kopt],
				vquad = 0;
			int ih = 0;
			for (int j = 0; j < n; j++)
			{
				vquad += d[j] * gopt[j];
				for (int i = 0; i <= j; i++)
				{
					double temp = d[i] * d[j];
					if (i == j) temp = 0.5 * temp;
					vquad += hq[ih] * temp;
					ih++;
				}
			}
			for (int k = 0; k < npt; k++)
			{
				//double val = w[npt+k];
				double val = work2[k];
				vquad += 0.5 * pq[k] * val * val;
			}
			double  diff = f - fopt - vquad;
			diffc = diffb;
			diffb = diffa;
			diffa = abs(diff);
			if (dnorm > rho) nfsav = nf;
			// Pick the next value of DELTA after a trust region step.
			if (ntrits > 0)
			{
				if (vquad >= 0)
				{
					loop = false; label = 720; break; // goto 720
				}
				ratio = (f - fopt) / vquad;
				delta = ratio <= 0.1 ? min(0.5 * delta, dnorm)
					: max(0.5 * delta, (ratio <= 0.7 ? dnorm : 2 * dnorm));
				if (delta <= 1.5 * rho)
					delta = rho;
				//printState("bobyqb-360-ifntrits>0-checkstate\n"+
				//	"delta="+delta+", dnorm="+dnorm+
				//	"distsq="+distsq, 0, null, null, null, null, null, null, null, null);

				// Recalculate KNEW and DENOM if the new F is less than FOPT.
				if (f < fopt)
				{
					int ksav = knew;
					double
						densav = denom,
						delsq = delta * delta,
						scaden = 0,
						biglsq = 0,
						val;
					knew = -1;
					for (int k = 0; k < npt; k++)
					{
						double hdiag = 0;
						for (int jj = 0; jj < nptm; jj++)
						{
							val = zmat[k][jj];
							hdiag += val * val;
						}
						val = vlag[k];
						den = beta * hdiag + val * val;
						distsq = 0;
						for (int j = 0; j < n; j++)
						{
							val = xpt[k][j] - xnew[j];
							distsq += val * val;
						}
						val = distsq / delsq;
						double temp = max(1, val * val);
						if (temp * den > scaden)
						{
							scaden = temp * den;
							knew = k;
							denom = den;
						}
						val = vlag[k];
						biglsq = max(biglsq, temp * val * val);
					} // end for (int k = 0; k < npt; k++)
					if (scaden <= 0.5 * biglsq)
					{
						knew = ksav;
						denom = densav;
					}
				}
			} // end if (ntrits > 0)
			/*
			 * Update BMAT and ZMAT, so that the KNEW-th interpolation point can be
			 * moved. Also update the second derivative terms of the model.
			 */
			//printState("bobyqb-360-before-update", nf, xbase, xopt, fval, vlag, bmat, zmat);
			update(bmat, zmat, vlag, beta, denom, knew);
			//printState("bobyqb-360-after-update", nf, xbase, xopt, fval, vlag, bmat, zmat);
			ih = 0;
			double pqold = pq[knew];
			pq[knew] = 0;
			for (int i = 0; i < n; i++)
			{
				double temp = pqold * xpt[knew][i];
				for (int j = 0; j <= i; j++)
				{
					hq[ih] = hq[ih] + temp * xpt[knew][j];
					ih++;
				}
			}
			for (int jj = 0; jj < nptm; jj++)
			{
				double temp = diff * zmat[knew][jj];
				for (int k = 0; k < npt; k++)
					pq[k] = pq[k] + temp * zmat[k][jj];
			}
			/*
			 * Include the new interpolation point, and make the changes to GOPT at
			 * the old XOPT that are caused by the updating of the quadratic model.
			 */
			fval[knew] = f;
			for (int i = 0; i < n; i++)
			{
				xpt[knew][i] = xnew[i];
				//w[i] = bmat[knew][i];
				work1[i] = bmat[knew][i];
			}
			for (int k = 0; k < npt; k++)
			{
				double suma = 0, sumb = 0;
				for (int jj = 0; jj < nptm; jj++)
					suma += zmat[knew][jj] * zmat[k][jj];
				for (int j = 0; j < n; j++)
					sumb += xpt[k][j] * xopt[j];
				double temp = suma * sumb;
				for (int i = 0; i < n; i++)
					work1[i] = work1[i] + temp * xpt[k][i];
					//w[i] += temp * xpt[k][i];
			}
			for (int i = 0; i < n; i++)
				gopt[i] = gopt[i] + diff * work1[i];
				//gopt[i] += diff * w[i];
			// Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT.
			if (f < fopt)
			{
				kopt = knew;
				xoptsq = 0;
				ih = 0;
				for (int j = 0; j < n; j++)
				{
					xopt[j] = xnew[j];
					double val = xopt[j];
					xoptsq += val * val;
					for (int i = 0; i <= j; i++)
					{
						if (i < j)
							gopt[j] = gopt[j] + hq[ih] * d[i];
						gopt[i] = gopt[i] + hq[ih] * d[j];
						ih++;
					}
				}
				for (int k = 0; k < npt; k++)
				{
					double temp = 0;
					for (int j = 0; j < n; j++)
						temp += xpt[k][j] * d[j];
					temp *= pq[k];
					for (int i = 0; i < n; i++)
						gopt[i] = gopt[i] + temp * xpt[k][i];
				}
			} // end if (f < fopt)
			/*
			 * Calculate the parameters of the least Frobenius norm interpolant to
			 * the current data, the gradient of this interpolant at XOPT being put
			 * into VLAG(NPT+I), I=1,2,...,N.
			 */
			if (ntrits > 0)
			{
				for (int k = 0; k < npt; k++)
				{
					vlag[k] = fval[k] - fval[kopt];
					//w[k] = 0;
					work3[k] = 0;
				}
				for (int j = 0; j < nptm; j++)
				{
					double sum = 0;
					for (int k = 0; k < npt; k++)
						sum += zmat[k][j] * vlag[k];
					for (int k = 0; k < npt; k++)
						work3[k] = work3[k] + sum * zmat[k][j];
						//w[k] += sum * zmat[k][j];
				}
				for (int k = 0; k < npt; k++)
				{
					double sum = 0;
					for (int j = 0; j < n; j++)
						sum += xpt[k][j] * xopt[j];
					//w[k + npt] = w[k];
					//w[k] *= sum;
					work2[k] = work3[k];
					work3[k] = work3[k] * sum;
				}
				double gqsq = 0, gisq = 0;
				for (int i = 0; i < n; i++)
				{
					double sum = 0, val;
					for (int k = 0; k < npt; k++)
						sum += bmat[k][i] * vlag[k] + xpt[k][i] * work3[k];
						//sum += bmat[k][i] * vlag[k] + xpt[k][i] * w[k];
					if (xopt[i] == sl[i])
					{
						val = min(0, gopt[i]);
						gqsq += val * val;
						val = min(0, sum);
						gisq += val * val;
					} else if (xopt[i] == su[i])
					{
						val = max(0, gopt[i]);
						gqsq += val * val;
						val = max(0, sum);
						gisq += val * val;
					} else
					{
						val = gopt[i];
						gqsq += val * val;
						gisq += sum * sum;
					}
					vlag[npt+i] = sum;
				} // end for (int i = 0; i < n; i++)
				/*
				 * Test whether to replace the new quadratic model by the least Frobenius
				 * norm interpolant, making the replacement if the test is satisfied.
				 */
				itest++;
				if (gqsq < 10 * gisq)
					itest = 0;
				if (itest >= 3)
				{
					/*
					 * Due to the precision problem below (see my comments), I cannot verify
					 * this particular branch numerically with that of the FORTRAN code.
					 * However, test runs looks okay. -- RJ
					 */
					int val = max(npt, nh);
					for (int i = 0; i < val; i++)
					{
						if (i < n) gopt[i] = vlag[npt + i];
						//if (i < npt) pq[i] = w[npt + i];
						if (i < npt) pq[i] = work2[i];
						if (i < nh) hq[i] = 0;
					}
					itest = 0;
				}
			} // end if (ntrits > 0)
			/*
			 * If a trust region step has provided a sufficient decrease in F, then
			 * branch for another trust region calculation. The case NTRITS=0 occurs
			 * when the new interpolation point was reached by an alternative step.
			 */
			if (ntrits == 0 || f <= fopt + 0.1 * vquad)
			{
				label = 60; break; // goto 60
			}
			double v1 = 2 * delta, v2 = 10 * rho; 
			distsq = max(v1 * v1, v2 * v2);
			label = 650;
			break;
		case 650:
			/*
			 * Alternatively, find out if the interpolation points are close enough
			 * to the best point so far.
			 */
			//printState("bobyqb-650-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			knew = -1;
			for (int k = 0; k < npt; k++)
			{
				double sum = 0;
				for (int j = 0; j < n; j++)
				{
					double val = xpt[k][j] - xopt[j];
					sum += val * val;
				}
				if (sum > distsq)
				{
					knew = k;
					distsq = sum;
				}
			}
			/*
			 * If KNEW is positive, then ALTMOV finds alternative new positions for
			 * the KNEW-th interpolation point within distance ADELT of XOPT. It is
			 * reached via label 90. Otherwise, there is a branch to label 60 for
			 * another trust region iteration, unless the calculations with the
			 * current RHO are complete.
			 */
			if (knew >= 0)
			{
				double dist = sqrt(distsq);
				if (ntrits == -1)
				{
					delta = min(0.1 * delta, 0.5 * dist);
					if (delta <= 1.5 * rho) delta = rho;
					//printState("bobyqb-650-ifknew>=0,ntrits==-1-checkstate\n"+
					//	"delta="+delta+", dnorm="+dnorm+
					//	"distsq="+distsq, 0, null, null, null, null, null, null, null, null);
				}
				ntrits = 0;
				adelt = max(min(0.1 * dist, delta), rho);
				dsq = adelt * adelt;
				label = 90; break; // goto 90;
			}
			if (ntrits != -1)
			{
				if (ratio > 0 || max(delta, dnorm) > 0)
				{
					label = 60; break; // goto 60
				}
			}
			label = 680;
			break;
		case 680:
			/*
			 * The calculations with the current value of RHO are complete. Pick the
			 * next values of RHO and DELTA.
			 */
			//printState("bobyqb-680-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
			if (rho > rhoend)
			{
				delta = 0.5 * rho;
				ratio = rho / rhoend;
				rho = ratio <= 16 ? rhoend : ratio <= 250 ? sqrt(ratio) * rhoend : 0.1 * rho;
				delta = max(delta, rho);
				// Some debug printout, if you wish
				/*
				System.out.println("Rho = " + rho + ", Number of function calls = "+nf);
				System.out.println("Optimum F = " + fval[kopt] + ", on X = ");
				for (int i = 0; i < n; i++)
					System.out.print((xbase[i] + xopt[i]) + " ");
				System.out.println();
				//*/
				ntrits = 0;
				nfsav = nf;
				label = 60; break; // goto 60
			}
			if (ntrits == -1)
			{
				label = 360; break; // goto 360
			}
			loop = false; label = 720;
			break;
		}
		// Label 720
		//printState("bobyqb-720-start", nf, xbase, xopt, fval, vlag, d, bmat, zmat, xpt);
		if (fval[kopt] <= fsave)
		{
			for (int i = 0; i < n; i++)
			{
				x[i] = min(max(xl[i], xbase[i] + xopt[i]), xu[i]);
				if (xopt[i] == sl[i]) x[i] = xl[i];
				if (xopt[i] == su[i]) x[i] = xu[i];
			}
			f=fval[kopt];
		}
		//return f;
		return new OptimizationResult(x, f, nf, isMinimize);
	}

	/**
	 * <pre>
	 * The arguments N, NPT, X, XL, XU, RHOBEG, RHOEND, IPRINT and MAXFUN
	 *   are identical to the corresponding arguments in SUBROUTINE BOBYQA.
	 * XBASE holds a shift of origin that should reduce the contributions
	 *   from rounding errors to values of the model and Lagrange functions.
	 * XPT is a two-dimensional array that holds the coordinates of the
	 *   interpolation points relative to XBASE.
	 * FVAL holds the values of F at the interpolation points.
	 * XOPT is set to the displacement from XBASE of the trust region centre.
	 * GOPT holds the gradient of the quadratic model at XBASE+XOPT.
	 * HQ holds the explicit second derivatives of the quadratic model.
	 * PQ contains the parameters of the implicit second derivatives of the
	 *   quadratic model.
	 * BMAT holds the last N columns of H.
	 * ZMAT holds the factorization of the leading NPT by NPT submatrix of H,
	 *   this factorization being ZMAT times ZMAT^T, which provides both the
	 *   correct rank and positive semi-definiteness.
	 * NDIM is the first dimension of BMAT and has the value NPT+N.
	 * SL and SU hold the differences XL-XBASE and XU-XBASE, respectively.
	 *   All the components of every XOPT are going to satisfy the bounds
	 *   SL(I) .LEQ. XOPT(I) .LEQ. SU(I), with appropriate equalities when
	 *   XOPT is on a constraint boundary.
	 * XNEW is chosen by SUBROUTINE TRSBOX or ALTMOV. Usually XBASE+XNEW is the
	 *   vector of variables for the next call of CALFUN. XNEW also satisfies
	 *   the SL and SU constraints in the way that has just been mentioned.
	 * XALT is an alternative to XNEW, chosen by ALTMOV, that may replace XNEW
	 *   in order to increase the denominator in the updating of UPDATE.
	 * D is reserved for a trial step from XOPT, which is usually XNEW-XOPT.
	 * VLAG contains the values of the Lagrange functions at a new point X.
	 *   They are part of a product that requires VLAG to be of length NDIM.
	 * W is a one-dimensional array that is used for working space. Its length
	 *   must be at least 3*NDIM = 3*(NPT+N).
	 * </pre>
	 * @param x
	 * @param xl
	 * @param xu
	 * @param calfun
	 * @param rhobeg
	 * @param rhoend
	 * @param iprint
	 * @param maxfun
	 * @param xbase
	 * @param xpt
	 * @param fval
	 * @param xopt
	 * @param gopt
	 * @param hq
	 * @param pq
	 * @param bmat
	 * @param zmat
	 * @param sl
	 * @param su
	 * @param xnew
	 * @param xalt
	 * @param d
	 * @param vlag
	 */
	/*
	static final double bobyqb(double[] x, double[] xl, double[] xu, IMultivariableFunction calfun,
		double rhobeg, double rhoend, int iprint, int maxfun, double[] xbase, double[][] xpt, double[] fval,
		double[] xopt, double[] gopt, double[] hq, double[] pq, double[][] bmat, double[][] zmat,
		double[] sl, double[] su, double[] xnew, double[] xalt, double[] d, double[] vlag)
	{
	}
	//*/

	/**
	 * <pre>
	 * The arguments N, NPT, X, XL, XU, RHOBEG, IPRINT and MAXFUN are the
	 *   same as the corresponding arguments in SUBROUTINE BOBYQA.
	 * The arguments XBASE, XPT, FVAL, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU
	 *   are the same as the corresponding arguments in BOBYQB, the elements
	 *   of SL and SU being set in BOBYQA.
	 * GOPT is usually the gradient of the quadratic model at XOPT+XBASE, but
	 *   it is set by PRELIM to the gradient of the quadratic model at XBASE.
	 *   If XOPT is nonzero, BOBYQB will change it to its usual value later.
	 * NF is maintined as the number of calls of CALFUN so far.
	 * KOPT will be such that the least calculated value of F so far is at
	 *   the point XPT(KOPT,.)+XBASE in the space of the variables.
	 * 
	 * SUBROUTINE PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
	 * BMAT and ZMAT for the first iteration, and it maintains the values of
	 * NF and KOPT. The vector X is also changed by PRELIM.
	 * </pre>
	 * @param x
	 * @param xl
	 * @param xu
	 * @param calfun
	 * @param rhobeg
	 * @param maxfun
	 * @param xbase
	 * @param xpt
	 * @param fval
	 * @param gopt
	 * @param hq
	 * @param pq
	 * @param bmat
	 * @param zmat
	 * @param sl
	 * @param su
	 * @param isMinimize true if goal is to find the global minimum
	 * @return integer array of two elements: nf and kopt
	 */
	private static final int[] prelim(double[] x, double[] xl, double[] xu, MultivariableFunction calfun, double rhobeg, int maxfun,
		double[] xbase, double[][] xpt, double[] fval, double[] gopt, double[] hq, double[] pq, double[][] bmat,
		double[][] zmat, double[] sl, double[] su, boolean isMinimize)
	{
		int nf, kopt = 0;
		int
			n = x.length,
			ndim = bmat.length,
			npt = zmat.length,
			mult = isMinimize ? 1 : -1;
		double
			rhosq = rhobeg * rhobeg,
			recip = 1/rhosq,
			fbeg = 0;
		int np = n + 1, nnpdiv2 = n*np/2, nptminnp = npt - np;
		/*
		 * Set XBASE to the initial vector of variables, and set the initial
		 * elements of XPT, BMAT, HQ, PQ and ZMAT to zero.
		 */
		for (int j = 0; j < n; j++)
		{
			xbase[j] = x[j];
			for (int k = 0; k < npt; k++)
				xpt[k][j] = 0;
			for (int i = 0; i < ndim; i++)
				bmat[i][j] = 0;
		}
		for (int ih = 0; ih < nnpdiv2; ih++)
			hq[ih] = 0;
		for (int k = 0; k < npt; k++)
		{
			pq[k] = 0;
			for (int j = 0; j < nptminnp; j++)
				zmat[k][j] = 0;
		}
		/*
		 * Begin the initialization procedure. NF becomes one more than the number
		 * of function values so far. The coordinates of the displacement of the
		 * next initial interpolation point from XBASE are set in XPT(NF+1,.).
		 */
		nf = 0;
		int ipt = 0, jpt = 0;
		double stepa = 0, stepb = 0;
		do // L50
		{
			int nfm = nf, nfx = nf - n; nf++;
			if (nfm <= 2 * n)
			{
				if (nfm >= 1 && nfm <= n)
				{
					stepa = su[nfm-1] == 0 ? -rhobeg : rhobeg;
					xpt[nf-1][nfm-1] = stepa;
				} else if (nfm > n)
				{
					stepa = xpt[nf-n-1][nfx-1];
					stepb = -rhobeg;
					if (sl[nfx-1] == 0) stepb = min(2*rhobeg, su[nfx-1]);
					if (su[nfx-1] == 0) stepb = max(-2*rhobeg, sl[nfx-1]);
					xpt[nf-1][nfx-1] = stepb;
				}
			} else
			{
				int itemp = (nfm - np) / n;
				jpt = nfm - itemp * n - n;
				ipt = jpt + itemp;
				if (ipt > n)
				{
					itemp = jpt;
					jpt = ipt - n;
					ipt = itemp;
				}
				xpt[nf-1][ipt-1] = xpt[ipt][ipt-1];
				xpt[nf-1][jpt-1] = xpt[jpt][jpt-1];
			}
			/*
			 * Calculate the next value of F. The least function value so far and
			 * its index are required.
			 */
			for (int j = 0; j < n; j++)
			{
				x[j] = min(max(xl[j], xbase[j] + xpt[nf-1][j]), xu[j]);
				if (xpt[nf-1][j] == sl[j]) x[j] = xl[j];
				if (xpt[nf-1][j] == su[j]) x[j] = xu[j];
			}
			double f = calfun.eval(x) * mult;
			fval[nf-1] = f;
			if (nf == 1)
			{
				fbeg = f;
				kopt = 1;
			} else if (f < fval[kopt-1])
				kopt = nf;
			/*
			 * Set the nonzero initial elements of BMAT and the quadratic model in the
			 * cases when NF is at most 2*N+1. If NF exceeds N+1, then the positions
			 * of the NF-th and (NF-N)-th interpolation points may be switched, in
			 * order that the function value at the first of them contributes to the
			 * off-diagonal second derivative terms of the initial quadratic model.
			 */
			int ih;
			if (nf <= 2*n+1)
			{
				if (nf >= 2 && nf <= n+1)
				{
					gopt[nfm-1] = (f - fbeg) / stepa;
					if (npt < nf + n)
					{
						bmat[0][nfm-1] = -1 / stepa;
						bmat[nf-1][nfm-1] = 1/stepa;
						bmat[npt+nfm-1][nfm-1] = -0.5 * rhosq;
					}
				} else if (nf >= n+2)
				{
					ih = (nfx * (nfx + 1)) / 2;
					double temp = (f - fbeg) / stepb;
					double diff = stepb - stepa;
					hq[ih - 1] = 2 * (temp - gopt[nfx - 1]) / diff;
					gopt[nfx - 1] = (gopt[nfx - 1] * stepb - temp * stepa) / diff;
					if (stepa * stepb < 0 && f < fval[nf - n - 1])
					{
						fval[nf - 1] = fval[nf - n - 1];
						fval[nf - n - 1] = f;
						if (kopt == nf) kopt = nf - n;
						xpt[nf-n-1][nfx - 1] = stepb;
						xpt[nf - 1][nfx - 1] = stepa;
					}
					bmat[0][nfx - 1] = -(stepa + stepb) / (stepa * stepb);
					bmat[nf - 1][nfx - 1] = -0.5/xpt[nf - n - 1][nfx - 1];
					bmat[nf - n - 1][nfx - 1] = -bmat[0][nfx - 1] - bmat[nf - 1][nfx - 1];
					zmat[0][nfx - 1] = sqrt(2) / (stepa * stepb);
					zmat[nf - 1][nfx - 1] = sqrt(0.5) / rhosq;
					zmat[nf - n - 1][nfx - 1] = -zmat[0][nfx - 1] - zmat[nf - 1][nfx - 1];
				} // end else if (nf >= n+2)
			} else {
				/*
				 * Set the off-diagonal second derivatives of the Lagrange functions and
				 * the initial quadratic model.
				 */
				ih = (ipt * (ipt - 1)) / 2 + jpt;
				zmat[0][nfx - 1] = zmat[nf - 1][nfx - 1] = recip;
				zmat[ipt][nfx - 1] = zmat[jpt][nfx - 1] = -recip;
				double temp = xpt[nf - 1][ipt - 1] * xpt[nf - 1][jpt - 1];
				hq[ih - 1] = (fbeg-fval[ipt]-fval[jpt] + f) / temp;
			}
		} while (nf < npt && nf < maxfun); // end do-while loop L50
		return new int[] {nf, kopt-1};
	}

	/**
	 * <pre>
	 * The arrays BMAT and ZMAT are updated, as required by the new position
	 * of the interpolation point that has the index KNEW. The vector VLAG has
	 * N+NPT components, set on entry to the first NPT and last N components
	 * of the product Hw in equation (4.11) of the Powell (2006) paper on
	 * NEWUOA. Further, BETA is set on entry to the value of the parameter
	 * with that name, and DENOM is set to the denominator of the updating
	 * formula. Elements of ZMAT may be treated as zero if their moduli are
	 * at most ZTEST. The first NDIM elements of W are used for working space.
	 * </pre>
	 * @param bmat
	 * @param zmat
	 * @param vlag
	 * @param beta
	 * @param denom
	 * @param knew
	 */
	private static final void update(double[][] bmat, double[][] zmat, double[] vlag, double beta, double denom, int knew)
	{
		int
			ndim = bmat.length,
			n = bmat[0].length,
			npt = zmat.length;
		int nptm = npt - n - 1;
		double ztest = 0, w[] = new double[ndim];
		for (int k = 0; k < npt; k++)
			for (int j = 0; j < nptm; j++)
				ztest = max(ztest, abs(zmat[k][j]));
		ztest *= 1e-20;
		// Apply the rotations that put zeros in the KNEW-th row of ZMAT.
		// int jl = 1
		for (int j = 1; j < nptm; j++)
		{
			if (abs(zmat[knew][j]) > ztest)
			{
				double
					tempa = zmat[knew][0],
					tempb = zmat[knew][j],
					temp = sqrt(tempa * tempa + tempb * tempb);
				tempa = tempa / temp;
				tempb = tempb / temp;
				for (int i = 0; i < npt; i++)
				{
					temp = tempa * zmat[i][0] + tempb * zmat[i][j];
					zmat[i][j] = tempa * zmat[i][j] - tempb * zmat[i][0];
					zmat[i][0] = temp;
				}
			}
			zmat[knew][j] = 0;
		}
		// Put the first NPT components of the KNEW-th column of HLAG into W,
		// and calculate the parameters of the updating formula.
		for (int i = 0; i < npt; i++)
			w[i] = zmat[knew][0] * zmat[i][0];
		double
			alpha = w[knew],
			tau = vlag[knew],
			// Complete the updating of ZMAT.
			temp = sqrt(denom),
			tempb = zmat[knew][0] / temp,
			tempa = tau / temp;
		vlag[knew] = vlag[knew] - 1;
		for (int i = 0; i < npt; i++)
			zmat[i][0] = tempa * zmat[i][0] - tempb * vlag[i];

		// Finally, update the matrix BMAT.
        //System.out.println("alpha=" + alpha + ", beta=" + beta + ", tau=" + tau + ", denom=" + denom);
        //System.out.println(toString(w));
		for (int j = 0; j < n; j++)
		{
			int jp = npt + j;
			w[jp] = bmat[knew][j];
			tempa = (alpha * vlag[jp] - tau * w[jp]) / denom;
			tempb = (-beta * w[jp] - tau * vlag[jp]) / denom;
			//System.out.println("tempa=" + tempa + ", tempb=" + tempb);
			//System.out.println("j=" + j + ": " + toString(w));
			for (int i = 0; i <= jp; i++)
			{
				// NOTE: DO NOT change the expression to bmat[i][j] += ...
				// This will cause difference in 18th digit under zero but will be carried out in thousands iteration!
				bmat[i][j] = bmat[i][j] + tempa * vlag[i] + tempb * w[i];
				if (i >= npt)
					bmat[jp][i-npt]=bmat[i][j];
			}
		}
	}

	/**
	 * <pre>
	 * The arguments N, NPT, XL, XU, MAXFUN, XBASE, XPT, FVAL, XOPT,
	 *   GOPT, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU have the same meanings as
	 *   the corresponding arguments of BOBYQB on the entry to RESCUE.
	 * NF is maintained as the number of calls of CALFUN so far, except that
	 *   NF is set to -1 if the value of MAXFUN prevents further progress.
	 * KOPT is maintained so that FVAL(KOPT) is the least calculated function
	 *   value. Its correct value must be given on entry. It is updated if a
	 *   new least function value is found, but the corresponding changes to
	 * XOPT and GOPT have to be made later by the calling program.
	 * DELTA is the current trust region radius.
	 * VLAG is a working space vector that will be used for the values of the
	 *   provisional Lagrange functions at each of the interpolation points.
	 *   They are part of a product that requires VLAG to be of length NDIM.
	 * PTSAUX is also a working space array. For J=1,2,...,N, PTSAUX(1,J) and
	 *   PTSAUX(2,J) specify the two positions of provisional interpolation
	 *   points when a nonzero step is taken along e_J (the J-th coordinate
	 *   direction) through XBASE+XOPT, as specified below. Usually these
	 *   steps have length DELTA, but other lengths are chosen if necessary
	 *   in order to satisfy the given bounds on the variables.
	 * PTSID is also a working space array. It has NPT components that denote
	 *   provisional new positions of the original interpolation points, in
	 *   case changes are needed to restore the linear independence of the
	 *   interpolation conditions. The K-th point is a candidate for change
	 *   if and only if PTSID(K) is nonzero. In this case let p and q be the
	 *   int parts of PTSID(K) and (PTSID(K)-p) multiplied by N+1. If p
	 *   and q are both positive, the step from XBASE+XOPT to the new K-th
	 *   interpolation point is PTSAUX(1,p)*e_p + PTSAUX(1,q)*e_q. Otherwise
	 *   the step is PTSAUX(1,p)*e_p or PTSAUX(2,q)*e_q in the cases q=0 or
	 *   p=0, respectively.
	 * The first NDIM+NPT elements of the array W are used for working space.
	 * The final elements of BMAT and ZMAT are set in a well-conditioned way
	 *   to the values that are appropriate for the new interpolation points.
	 * The elements of GOPT, HQ and PQ are also revised to the values that are
	 *   appropriate to the final quadratic model.
	 * </pre>
	 * @param xl
	 * @param xu
	 * @param calfun
	 * @param maxfun
	 * @param xbase
	 * @param xpt
	 * @param fval
	 * @param xopt
	 * @param gopt
	 * @param hq
	 * @param pq
	 * @param bmat
	 * @param zmat
	 * @param sl
	 * @param su
	 * @param nf
	 * @param delta
	 * @param kopt
	 * @param vlag
	 * @param isMinimize true if the goal is to find the global minimum
	 * @return integer array of two elements: nf and kopt
	 */
	private static final int[] rescue(double[] xl, double[] xu, MultivariableFunction calfun, int maxfun,
		double[] xbase, double[][] xpt, double[] fval, double[] xopt, double[] gopt, double[] hq, double[] pq,
		double[][] bmat, double[][] zmat, double[] sl, double[] su, int nf,
		double delta, int kopt, double[] vlag, boolean isMinimize)
	{
		// TODO This entire subroutine is untested
		int
			n = xl.length,
			ndim = bmat.length,
			npt = xpt.length,
			np = n + 1,
			mult = isMinimize ? 1 : -1,
			nptm = npt - np;
		assert(npt == zmat.length);
		double
			sfrac = 0.5 / np,
			beta = 0,
			denom = 0,
			ptsaux[][] = new double[2][n],
			ptsid[] = new double[npt],
			w[] = new double[ndim + npt];
		/*
		 * Shift the interpolation points so that XOPT becomes the origin, and set
		 * the elements of ZMAT to zero. The value of SUMPQ is required in the
		 * updating of HQ below. The squares of the distances from XOPT to the
		 * other interpolation points are set at the end of W. Increments of WINC
		 * may be added later to these squares to balance the consideration of
		 * the choice of point that is going to become current.
		 */
		double
			sumpq = 0,
			winc = 0;
		for (int k = 0; k < npt; k++)
		{
			double distsq = 0;
			for (int j = 0; j < n; j++)
			{
				xpt[k][j] = xpt[k][j] - xopt[j];
				double temp = xpt[k][j];
				distsq += temp * temp;
			}
			sumpq += pq[k];
			w[ndim + k] = distsq;
			winc = max(winc, distsq);
			for (int j = 0; j < nptm; j++)
				zmat[k][j] = 0;
		}
		/*
		 * Update HQ so that HQ and PQ define the second derivatives of the model
		 * after XBASE has been shifted to the trust region centre.
		 */
		int ih = 0;
		for (int j = 0; j < n; j++)
		{
			w[j] = 0.5 * sumpq * xopt[j];
			for(int k = 0; k < npt; k++)
				w[j] = w[j] + pq[k]*xpt[k][j];
			for (int i = 0; i <= j; i++)
			{
				hq[ih] = hq[ih] + w[i] * xopt[j] + w[j] * xopt[i];
				ih++;
			}
		}
		/*
		 * Shift XBASE, SL, SU and XOPT. Set the elements of BMAT to zero, and
		 * also set the elements of PTSAUX.
		 */
		for (int j = 0; j < n; j++)
		{
			xbase[j] = xbase[j] + xopt[j];
			sl[j] = sl[j] - xopt[j];
			su[j] = su[j] - xopt[j];
			xopt[j] = 0;
			ptsaux[0][j] = min(delta, su[j]);
			ptsaux[1][j] = max(-delta, sl[j]);
			if (ptsaux[0][j] + ptsaux[1][j] < 0)
			{
				double temp = ptsaux[0][j];
				ptsaux[0][j] = ptsaux[1][j];
				ptsaux[1][j] = temp;
			}
			if (abs(ptsaux[1][j]) < 0.5 * abs(ptsaux[0][j]))
				ptsaux[1][j] = 0.5 * ptsaux[0][j];
			for (int i = 0; i < ndim; i++)
				bmat[i][j] = 0;
		}
		double fbase = fval[kopt];
		/*
		 * Set the identifiers of the artificial interpolation points that are
		 * along a coordinate direction from XOPT, and set the corresponding
		 * nonzero elements of BMAT and ZMAT.
		 */
		ptsid[0] = sfrac;
		for (int j = 0; j < n; j++)
		{
			int jp = j+1, jpn = jp+n;
			double ptsaux1j = ptsaux[0][j], ptsaux2j = ptsaux[1][j];
			ptsid[jp] = j + sfrac;
			if (jpn < npt) // if (jpn + 1 <= npt)
			{
				ptsid[jpn] = j * 1.0 / np + sfrac;
				double temp = 1.0 / (ptsaux1j - ptsaux2j);
				bmat[jp][j] = -temp + 1.0 / ptsaux1j;
				bmat[jpn][j] = temp + 1.0 / ptsaux2j;
				bmat[0][j] = -bmat[jp][j] - bmat[jpn][j];
				zmat[0][j] = sqrt(2.0) / abs(ptsaux1j * ptsaux2j);
				zmat[jp][j] = zmat[0][j] * ptsaux2j * temp;
				zmat[jpn][j] = -zmat[0][j] * ptsaux1j * temp;
			} else {
				bmat[0][j] = -1.0 / ptsaux1j;
				bmat[jp][j] = 1.0 / ptsaux1j;
				bmat[j+npt][j] = -0.5 * ptsaux1j * ptsaux1j;
			}
		}
		// Set any remaining identifiers with their nonzero elements of ZMAT.
		if (npt >= n + np)
		{
			for (int k = 2 * np - 1; k < npt; k++) // TODO beware of ptsid
			{
				int
					iw = (int) ((k - np - 0.5) / n),
					ip = k - np - iw * n,
					iq = ip + iw,
					knpm1 = k - np;
				if (iq >= n) iq -= n;
				ptsid[k] = ip + iq * 1.0 / np + sfrac;
				double temp = 1.0 / (ptsaux[0][ip] * ptsaux[1][iq]);
				zmat[0][knpm1] = temp;
				zmat[ip][knpm1] = -temp;
				zmat[iq][knpm1] = -temp;
				zmat[k][knpm1] = temp;
 			}
		}
		int
			nrem = npt,
			kold = 0,
			knew = kopt;
		boolean gotoL120 = false;
		// Reorder the provisional points in the way that exchanges PTSID(KOLD) with PTSID(KNEW).

		for(;;) {
			// L80:
			if (!gotoL120)
			{
				for (int j = 0; j < n; j++)
				{
					double temp = bmat[kold][j];
					bmat[kold][j] = bmat[knew][j];
					bmat[knew][j] = temp;
				}
				for (int j = 0; j < nptm; j++)
				{
					double temp = zmat[kold][j];
					zmat[kold][j] = zmat[knew][j];
					zmat[knew][j] = temp;
				}
				ptsid[kold] = ptsid[knew];
				ptsid[knew] = 0;
				w[ndim + knew] = 0;
				nrem--;
				if (knew != kopt)
				{
					double temp = vlag[kold];
					vlag[kold] = vlag[knew];
					vlag[knew] = temp;
					/*
					 *  Update the BMAT and ZMAT matrices so that the status of the KNEW-th
					 *  interpolation point can be changed from provisional to original. The
					 *  branch to label 350 occurs if all the original points are reinstated.
					 *  The nonnegative values of W(NDIM+K) are required in the search below.
					 */
					// update(double[][] bmat, double[][] zmat, double[] vlag, double beta, double denom, int knew)
					update(bmat, zmat, vlag, beta, denom, knew);
					if (nrem == 0)
						// goto L350 // TODO
						return new int[] {nf, kopt};
					for (int k = 0; k < npt; k++)
						w[ndim+k] = abs(w[ndim+k]);
				}
			}
			/*
			 * Pick the index KNEW of an original interpolation point that has not
			 * yet replaced one of the provisional interpolation points, giving
			 * attention to the closeness to XOPT and to previous tries with KNEW.
			 */
			// L120
			gotoL120 = false;
			double dsqmin = 0;
			for (int k = 0; k < npt; k++)
			{
				double wndimk = w[ndim + k];
				if ((wndimk > 0) && (dsqmin == 0 || wndimk < dsqmin))
				{
					knew = k;
					dsqmin = wndimk;
				}
			}
			if (dsqmin == 0)
				break; // goto L260
			// Form the W-vector of the chosen original interpolation point.
			for (int j = 0; j < n; j++)
				w[npt + j] = xpt[knew][j];
			for (int k = 0; k < npt; k++)
			{
				double sum = 0;
				if (k != kopt) {
					if (ptsid[k] == 0) {
						for (int j = 0; j < n; j++)
							sum += w[npt+j] * xpt[k][j];
					} else {
						int ip = (int) ptsid[k];
						if (ip > 0)
							sum = w[npt + ip] * ptsaux[0][ip];
						int iq = (int) (np * ptsid[k] - (ip * np));
						if (iq > 0)
						{
							int iw = 0;
							if (ip == 0) iw = 1;
							sum += w[npt+iq] * ptsaux[iw][iq];
						}
					}
				}
				w[k] = 0.5 * sum * sum;
			}
			// Calculate VLAG and BETA for the required updating of the H matrix if
			// XPT(KNEW,.) is reinstated in the set of interpolation points.
			for (int k = 0; k < npt; k++)
			{
				double sum = 0;
				for (int j = 0; j < n; j++)
					sum+= bmat[k][j] * w[npt + j];
				vlag[k] = sum;
			}
			beta = 0;
			for (int j = 0; j < nptm; j++)
			{
				double sum = 0;
				for (int k = 0; k < npt; k++)
					sum += zmat[k][j] * w[k];
				beta -= sum * sum;
				for (int k = 0; k < npt; k++)
					vlag[k] += sum * zmat[k][j];
			}
			double bsum = 0, distsq =0;
			for (int j = 0; j < n; j++)
			{
				double sum = 0;
				for (int k = 0; k < npt; k++)
					sum += bmat[k][j] * w[k];
				int jp = j + npt;
				bsum += sum * w[jp];
				for (int ip = npt; ip < ndim; ip++)
					sum += bmat[ip][j] * w[ip];
				bsum += sum * w[jp];
				vlag[jp] = sum;
				sum = xpt[knew][j];
				distsq += sum * sum;
			}
			beta = 0.5 * distsq * distsq + beta - bsum;
			vlag[kopt] = vlag[kopt] + 1;
			/*
			 * KOLD is set to the index of the provisional interpolation point that is
			 * going to be deleted to make way for the KNEW-th original interpolation
			 * point. The choice of KOLD is governed by the avoidance of a small value
			 * of the denominator in the updating calculation of UPDATE.
			 */
			denom = 0;
			double vlmxsq = 0;
			for (int k = 0; k < npt; k++)
			{
				double vlagksq = vlag[k];
				vlagksq *= vlagksq;
				if (ptsid[k] != 0)
				{
					double hdiag = 0;
					for (int j = 0; j < nptm; j++)
						hdiag += zmat[k][j] * zmat[k][j];
					double den = beta * hdiag + vlagksq;
					if (den > denom)
					{
						kold = k;
						denom = den;
					}
				}
				vlmxsq = max(vlmxsq, vlagksq);
			}
			if (denom <= 0.01 * vlmxsq)
			{
				w[ndim + knew] = -w[ndim+knew] - winc;
				gotoL120 = true;
				// goto L120;
			}
			// goto L80
		}
		/*
		 * When label 260 is reached, all the final positions of the interpolation
		 * points have been chosen although any changes have not been included yet
		 * in XPT. Also the final BMAT and ZMAT matrices are complete, but, apart
		 * from the shift of XBASE, the updating of the quadratic model remains to
		 * be done. The following cycle through the new interpolation points begins
		 * by putting the new point in XPT(KPT,.) and by setting PQ(KPT) to zero,
		 * except that a RETURN occurs if MAXFUN prohibits another value of F.
		 */
		// L260
		for (int kpt = 0; kpt < npt; kpt++)
		{
			double xp = 0, xq = 0;
			if (ptsid[kpt] < 0) continue; // goto 340
			if (nf >= maxfun)
			{
				nf = -1;
				break; // goto 350
			}
			ih = 0;
			for (int j = 0; j < n; j++)
			{
				w[j] = xpt[kpt][j];
				xpt[kpt][j] = 0;
				double temp = pq[kpt] * w[j];
				for (int i = 0; i <= j; i++)
				{
					hq[ih] = hq[ih] + temp * w[i];
					ih++;
				}
			}
			pq[kpt] = 0;
			int ip = (int) ptsid[kpt];
			int iq = (int) (np * ptsid[kpt] - (ip * np));
			if (ip >= 0)
			{
				xp = ptsaux[0][ip];
				xpt[kpt][ip] = xp;
			}
			if (iq >= 0)
			{
				xq = ptsaux[ip == 0 ? 1 : 0][iq];
				xpt[kpt][iq] = xq;
			}
			// Set VQUAD to the value of the current model at the new point.
			double vquad = fbase;
			int ihp = 0, ihq = 0;
			if (ip >= 0)
			{
				ihp = (ip + ip * ip) / 2;
				vquad += xp * (gopt[ip] + 0.5 * xp * hq[ihp-1]);
			}
			if (iq >= 0)
			{
				ihq = (iq + iq * iq) / 2;
				vquad += xq * (gopt[iq] + 0.5 * xq * hq[ihq-1]);
				if (ip >= 0)
				{
					int iw = max(ihp, ihq) - abs(ip - iq);
					vquad += xp * xq * hq[iw-1];
				}
			}
			for (int k = 0; k < npt; k++)
			{
				double temp = 0;
				if (ip >= 0) temp += xp * xpt[k][ip];
				if (iq >= 0) temp += xq * xpt[k][iq];
				vquad += 0.5 * pq[k] * temp * temp;
			}
			/*
			 * Calculate F at the new interpolation point, and set DIFF to the factor
			 * that is going to multiply the KPT-th Lagrange function when the model
			 * is updated to provide interpolation to the new function value.
			 */
			for (int i = 0; i < n; i++)
			{
				w[i] = min(max(xl[i], xbase[i] + xpt[kpt][i]), xu[i]);
				if (xpt[kpt][i] == sl[i]) w[i] = xl[i];
				if (xpt[kpt][i] == su[i]) w[i] = xu[i];
			}
			nf++;
			double f = calfun.eval(w) * mult;
			fval[kpt] = f;
			if (f < fval[kopt]) kopt = kpt;
			double diff = f - vquad;

			/*
			 * Update the quadratic model. The RETURN from the subroutine occurs when
			 * all the new interpolation points are included in the model.
			 */
			for (int i = 0; i < n; i++)
				gopt[i] = gopt[i] + diff * bmat[kpt][i];
			for (int k = 0; k < npt; k++)
			{
				double sum = 0;
				for (int j = 0; j < nptm; j++)
					sum += zmat[k][j] * zmat[kpt][j];
				double temp = diff * sum;
				if (ptsid[k] < 0)
					pq[k] = pq[k] + temp;
				else {
					ip = (int) ptsid[k];
					iq = (int) (np * ptsid[k] - (ip * np));
					ihq = (iq * iq + iq) / 2;
					if (ip == 0)
					{
						double val = ptsaux[1][iq];
						hq[ihq] = hq[ihq] + temp * val * val;
					} else {
						ihp = (ip * ip + ip) / 2;
						double val = ptsaux[0][ip];
						hq[ihp] = hq[ihp] + temp * val * val;
						if (iq >= 0)
						{
							double val2 = ptsaux[0][iq];
							hq[ihq] = hq[ihq] + temp * val2 * val2;
							int iw = max(ihp, ihq) - abs(iq - ip);
							hq[iw] = hq[iw] + temp * val * val2;
						}
					}
				}
			}
			ptsid[kpt] = -1;
		} // For L260
		return new int[] {nf, kopt};
	}

	/**
	 * <pre>
	 * The arguments N, NPT, XPT, XOPT, BMAT, ZMAT, NDIM, SL and SU all have
	 *   the same meanings as the corresponding arguments of BOBYQB.
	 * KOPT is the index of the optimal interpolation point.
	 * KNEW is the index of the interpolation point that is going to be moved.
	 * ADELT is the current trust region bound.
	 * XNEW will be set to a suitable new position for the interpolation point
	 *   XPT(KNEW,.). Specifically, it satisfies the SL, SU and trust region
	 *   bounds and it should provide a large denominator in the next call of
	 *   UPDATE. The step XNEW-XOPT from XOPT is restricted to moves along the
	 *   straight lines through XOPT and another interpolation point.
	 * XALT also provides a large value of the modulus of the KNEW-th Lagrange
	 *   function subject to the constraints that have been mentioned, its main
	 *   difference from XNEW being that XALT-XOPT is a constrained version of
	 *   the Cauchy step within the trust region. An exception is that XALT is
	 *   not calculated if all components of GLAG (see below) are zero.
	 * ALPHA will be set to the KNEW-th diagonal element of the H matrix.
	 * CAUCHY will be set to the square of the KNEW-th Lagrange function at
	 *   the step XALT-XOPT from XOPT for the vector XALT that is returned,
	 *   except that CAUCHY is set to zero if XALT is not calculated.
	 * GLAG is a working space vector of length N for the gradient of the
	 *   KNEW-th Lagrange function at XOPT.
	 * HCOL is a working space vector of length NPT for the second derivative
	 *   coefficients of the KNEW-th Lagrange function.
	 * W is a working space vector of length 2N that is going to hold the
	 *   constrained Cauchy step from XOPT of the Lagrange function, followed
	 *   by the downhill version of XALT when the uphill step is calculated.
	 * </pre>
	 * @param xpt
	 * @param xopt
	 * @param bmat
	 * @param zmat
	 * @param sl
	 * @param su
	 * @param kopt
	 * @param knew
	 * @param adelt
	 * @param xnew
	 * @param xalt
	 * @return an array of two elements: alpha, cauchy
	 */
	private static final double[] altmov(double[][] xpt, double[] xopt, double[][] bmat, double[][] zmat, double[] sl, double[] su,
		int kopt, int knew, double adelt, double[] xnew, double[] xalt)
	{
		int
			n = xpt[0].length,
			// ndim = bmat.length,
			npt = xpt.length;
		final double __const = 1 + sqrt(2.0);
		double alpha, cauchy;
		double[]
			glag = new double[n],
			hcol = new double[npt],
			w = new double[2*n];
		/*
		 * Set the first NPT components of W to the leading elements of the
		 * KNEW-th column of the H matrix.
		 */
		for (int k = 0; k < npt; k++)
			hcol[k] = 0;
		for (int j = 0; j < npt - n - 1; j++)
		{
			double temp = zmat[knew][j];
			for (int k = 0; k < npt; k++)
				hcol[k] = hcol[k] + temp * zmat[k][j];
		}
		alpha = hcol[knew];
		double ha = 0.5 * alpha;
		// Calculate the gradient of the KNEW-th Lagrange function at XOPT.
		for (int i = 0; i < n; i++)
			glag[i] = bmat[knew][i];
		for (int k = 0; k < npt; k++)
		{
			double temp = 0;
			for (int j = 0; j < n; j++)
				temp += xpt[k][j] * xopt[j];
			temp *= hcol[k];
			for (int i = 0; i < n; i++)
				glag[i] = glag[i] + temp * xpt[k][i];
		}
		/*
		 * Search for a large denominator along the straight lines through XOPT
		 * and another interpolation point. SLBD and SUBD will be lower and upper
		 * bounds on the step along each of these lines in turn. PREDSQ will be
		 * set to the square of the predicted denominator for each line. PRESAV
		 * will be set to the largest admissible value of PREDSQ that occurs.
		 */
		double presav = 0, stpsav = 0;
		int ksav = 0, ibdsav = 0;
		for (int k = 0; k < npt; k++)
		{
			if (k == kopt) continue;
			double dderiv = 0, distsq = 0;
			for (int i = 0; i < n; i++)
			{
				double temp = xpt[k][i] - xopt[i];
				dderiv += glag[i] * temp;
				distsq += temp * temp;
			}
			double subd = adelt / sqrt(distsq), slbd = -subd, sumin = min(1, subd);
			int ilbd = 0, iubd = 0;
			// Revise SLBD and SUBD if necessary because of the bounds in SL and SU.
			for (int i = 0; i < n; i++)
			{
				double temp = xpt[k][i] - xopt[i];
				if (temp > 0)
				{
					if (slbd * temp < sl[i] - xopt[i])
					{
						slbd = (sl[i] - xopt[i]) / temp;
						ilbd = -i-1;
					}
					if (subd * temp > su[i] - xopt[i])
					{
						subd = max(sumin, (su[i] - xopt[i])/temp);
						iubd = i+1;
					}
				} else if (temp < 0) {
					if (slbd * temp > su[i] - xopt[i])
					{
						slbd = (su[i] - xopt[i])/temp;
						ilbd = i+1;
					}
					if (subd * temp < sl[i] - xopt[i])
					{
						subd = max(sumin, (sl[i] - xopt[i])/temp);
						iubd = -i-1;
					}
				}
			}
			/*
			 * Seek a large modulus of the KNEW-th Lagrange function when the index
			 * of the other interpolation point on the line through XOPT is KNEW.
			 */
			double diff, step = slbd, vlag;
			int isbd = ilbd;
			if (k == knew)
			{
				diff = dderiv - 1;
				vlag = slbd * (dderiv - slbd * diff);
				double temp = subd * (dderiv - subd * diff);
				if (abs(temp) > abs(vlag))
				{
					step = subd;
					vlag = temp;
					isbd = iubd;
				}
				double
				tempd = 0.5 * dderiv,
				tempa = tempd - diff * slbd,
				tempb = tempd - diff * subd;
				if (tempa * tempb < 0)
				{
					temp = tempd * tempd / diff;
					if (abs(temp) > abs(vlag))
					{
						step = tempd/diff;
						vlag = temp;
						isbd = 0;
					}
				}
			} else {
				// Search along each of the other lines through XOPT and another point.
				vlag = slbd * (1.0 - slbd);
				double temp = subd * (1.0 - subd);
				if (abs(temp) > abs(vlag))
				{
					step = subd;
					vlag = temp;
					isbd = iubd;
				}
				if (subd > 0.5)
				{
					if (abs(vlag) < 0.25)
					{
						step = 0.5;
						vlag = 0.25;
						isbd = 0;
					}
				}
				vlag *= dderiv;
			}
			// Calculate PREDSQ for the current line search and maintain PRESAV.
			double temp = step * (1.0 - step) * distsq;
			double predsq = vlag * vlag * (vlag * vlag + ha * temp * temp);
			if (predsq > presav)
			{
				presav = predsq;
				ksav = k;
				stpsav = step;
				ibdsav = isbd;
			}
		}
		// Construct XNEW in a way that satisfies the bound constraints exactly.
		for (int i = 0; i < n; i++)
		{
			double temp = xopt[i] + stpsav * (xpt[ksav][i] - xopt[i]);
			xnew[i] = max(sl[i], min(su[i], temp));
		}
		if (ibdsav < 0) xnew[-ibdsav-1] = sl[-ibdsav-1];
		else if (ibdsav > 0) xnew[ibdsav-1] = su[ibdsav-1];
		/*
		 * Prepare for the iterative method that assembles the constrained Cauchy
		 * step in W. The sum of squares of the fixed components of W is formed in
		 * WFIXSQ, and the free components of W are set to BIGSTP.
		 */
		double bigstp = adelt + adelt;
		int iflag = 0;
		double csave = 0, temp;
		// L100
		while(true) {
			double wfixsq = 0, ggfree = 0;
			for (int i = 0; i < n; i++)
			{
				w[i] = 0;
				double glagi = glag[i];
				if (min(xopt[i] - sl[i], glagi) > 0 || max(xopt[i] - su[i], glagi) < 0)
				{
					w[i] = bigstp;
					ggfree += glagi * glagi;
				}
			}
			if (ggfree == 0)
				return new double[] {alpha, 0}; // goto 200; 
			double step = 0;
			// Investigate whether more components of W can be fixed.
			while ((temp = adelt * adelt - wfixsq) > 0) // L120
			{
				double wsqsav = wfixsq;
				step = sqrt(temp / ggfree);
				ggfree = 0;
				for (int i = 0; i < n; i++)
				{
					double wi;
					if (w[i] == bigstp)
					{
						temp = xopt[i] - step * glag[i];
						if (temp <= sl[i])
						{
							wi = w[i] = sl[i] - xopt[i];
							wfixsq += wi * wi;
						} else if (temp >= su[i])
						{
							wi = w[i] = su[i] - xopt[i];
							wfixsq += wi * wi;
						} else
						{
							wi = glag[i];
							ggfree += wi * wi;
						}
					}
				}
				if (!(wfixsq > wsqsav && ggfree > 0))
					break;
			}
			/*
			 * Set the remaining free components of W and all components of XALT,
			 * except that W may be scaled later.
			 */
			double gw = 0;
			for (int i = 0; i < n; i++)
			{
				double glagi = glag[i];
				if (w[i] == bigstp)
				{
					w[i] = -step * glagi;
					xalt[i] = max(sl[i], min(su[i], xopt[i] + w[i]));
				} else if (w[i] == 0)
					xalt[i] = xopt[i];
				else if (glagi > 0)
					xalt[i] = sl[i];
				else
					xalt[i] = su[i];
				gw += glagi * w[i];
			}
			/*
			 * Set CURV to the curvature of the KNEW-th Lagrange function along W.
			 * Scale W by a factor less than one if that can reduce the modulus of
			 * the Lagrange function at XOPT+W. Set CAUCHY to the final value of
			 * the square of this function.
			 */
			double curv = 0;
			for (int k = 0; k < npt; k++)
			{
				temp = 0;
				for (int j = 0; j < n; j++)
					temp += xpt[k][j] * w[j];
				curv += hcol[k] * temp * temp;
			}
			if (iflag == 1) curv = -curv;
			if (curv > -gw && curv < -gw * __const)
			{
				double scale = -gw / curv;
				for (int i = 0; i < n; i++)
				{
					temp = xopt[i] + scale * w[i];
					xalt[i] = max(sl[i], min(su[i], temp));
				}
				cauchy = 0.5 * gw * scale;
				cauchy *= cauchy;
			} else
			{
				cauchy = gw + 0.5 * curv;
				cauchy *= cauchy;
			}
			/*
			 * If IFLAG is zero, then XALT is calculated as before after reversing
			 * the sign of GLAG. Thus two XALT vectors become available. The one that
			 * is chosen is the one that gives the larger value of CAUCHY.
			 */
			if (iflag == 0)
			{
				for (int i = 0; i < n; i++)
				{
					glag[i] = -glag[i];
					w[n+i] = xalt[i];
				}
				csave = cauchy;
				iflag = 1;
				// goto 100;
				continue;
			}
			break;
		} // end for (;;) L100
		if (csave > cauchy)
		{
			for (int i = 0; i < n; i++)
				xalt[i] = w[n+i];
			cauchy = csave;
		}
		return new double[] {alpha, cauchy};
	}

	/**
	 * <pre>
	 * The arguments N, NPT, XPT, XOPT, GOPT, HQ, PQ, SL and SU have the same
	 *   meanings as the corresponding arguments of BOBYQB.
	 * DELTA is the trust region radius for the present calculation, which
	 *   seeks a small value of the quadratic model within distance DELTA of
	 *   XOPT subject to the bounds on the variables.
	 * XNEW will be set to a new vector of variables that is approximately
	 *   the one that minimizes the quadratic model within the trust region
	 *   subject to the SL and SU constraints on the variables. It satisfies
	 *   as equations the bounds that become active during the calculation.
	 * D is the calculated trial step from XOPT, generated iteratively from an
	 *   initial value of zero. Thus XNEW is XOPT+D after the final iteration.
	 * GNEW holds the gradient of the quadratic model at XOPT+D. It is updated
	 *   when D is updated.
	 * XBDI is a working space vector. For I=1,2,...,N, the element XBDI(I) is
	 *   set to -1.0, 0.0, or 1.0, the value being nonzero if and only if the
	 *   I-th variable has become fixed at a bound, the bound being SL(I) or
	 *   SU(I) in the case XBDI(I)=-1.0 or XBDI(I)=1.0, respectively. This
	 *   information is accumulated during the construction of XNEW.
	 * The arrays S, HS and HRED are also used for working space. They hold the
	 *   current search direction, and the changes in the gradient of Q along S
	 *   and the reduced D, respectively, where the reduced D is the same as D,
	 *   except that the components of the fixed variables are zero.
	 * DSQ will be set to the square of the length of XNEW-XOPT.
	 * CRVMIN is set to zero if D reaches the trust region boundary. Otherwise
	 *   it is set to the least curvature of H that occurs in the conjugate
	 *   gradient searches that are not restricted by any constraints. The
	 *   value CRVMIN=-1.0D0 is set, however, if all of these searches are
	 *   constrained.
	 *   
	 * A version of the truncated conjugate gradient is applied. If a line
	 * search is restricted by a constraint, then the procedure is restarted,
	 * the values of the variables that are at their bounds being fixed. If
	 * the trust region boundary is reached, then further changes may be made
	 * to D, each one being in the two dimensional space that is spanned
	 * by the current D and the gradient of Q at XOPT+D, staying on the trust
	 * region boundary. Termination occurs when the reduction in Q seems to
	 * be close to the greatest reduction that can be achieved.
	 * </pre>
	 * @param xpt
	 * @param xopt
	 * @param gopt
	 * @param hq
	 * @param pq
	 * @param sl
	 * @param su
	 * @param delta
	 * @param xnew
	 * @param d
	 * @param gnew
	 * @param dsq
	 * @param crvmin
	 * @return an array of two elements: dsq, crvmin
	 */
	private static final double[] trsbox(double[][] xpt, double[] xopt, double[] gopt, double[] hq, double[] pq,
		double[] sl, double[] su, double delta, double[] xnew, double[] d, double[] gnew, double dsq, double crvmin)
	{
		/*
		 * The sign of GOPT(I) gives the sign of the change to the I-th variable
		 * that will reduce Q from its value at XOPT. Thus XBDI(I) shows whether
		 * or not to fix the I-th variable at one of its bounds initially, with
		 * NACT being set to the number of fixed variables. D and GNEW are also
		 * set for the first iteration. DELSQ is the upper bound on the sum of
		 * squares of the free variables. QRED is the reduction in Q so far.
		 */
		//printState("trsbox-00-start-hq="+toString(hq), 0, null, xopt, null, null, d, null, null, xpt);
		int n = xpt[0].length, npt = xpt.length, iterc = 0, nact = 0;
		int[] xbdi = new int[n];
		double[] s = new double[n], hs = new double[n], hred = new double[n];
		//double sqstp = 0; // seems unused
		for (int i = 0; i < n; i++)
		{
			xbdi[i] = 0;
			if (xopt[i] <= sl[i])
			{
				if (gopt[i] >= 0) xbdi[i] = -1;
			} else if (xopt[i] >= su[i])
			{
				if (gopt[i] <= 0) xbdi[i] = 1;
			}
			if (xbdi[i] != 0) nact++;
			d[i] = 0;
			gnew[i] = gopt[i];
		}
		//printState("trsbox-00-after-firstloop", 0, null, xopt, null, null, d, null, null, xpt);
		double
			delsq = delta * delta,
			qred = 0,
			beta = 0,
			gredsq = 0,
			dredsq = 0,
			dredg = 0,
			ggsav = 0,
			stepsq = 0,
			sredg = 0,
			redsav = 0,
			angbd = 1;
		int itermax = 0, itcsav = 0, label = 30, xsav = 0, iact = 0;
		crvmin = -1;
		boolean loop = true;

		// L30
		for(;loop;)
		{
			switch (label)
			{
			case 30:
				//printState("trsbox-30-start", 0, null, xopt, null, null, d, null, null, xpt);
				stepsq = 0;
				for (int i = 0; i < n; i++)
				{
					if (xbdi[i] != 0)
						s[i] = 0;
					else if (beta == 0)
						s[i] = -gnew[i];
					else
						s[i] = beta * s[i] - gnew[i];
					double val = s[i];
					stepsq += val * val;
				}
				//printState("trsbox-30-checkstate:\n" +
				//		"stepsq="+stepsq+", beta="+beta+
				//		", gredsq="+gredsq, 0, null, null, null, null, null, null, null, null);
				if (stepsq == 0)
				{
					//label = 190; // goto 190;
					loop = false;
					break;
				}
				if (beta == 0)
				{
					gredsq = stepsq;
					itermax = iterc + n - nact;
				}
				if (gredsq * delsq <= qred * 1e-4 * qred)
				{
					//label = 190; // goto 190;
					loop = false;
					break;
				}
				/*
				 * Multiply the search direction by the second derivative matrix of Q and
				 * calculate some scalars for the choice of steplength. Then set BLEN to
				 * the length of the the step to the trust region boundary and STPLEN to
				 * the steplength, ignoring the simple bounds.
				 */
				label = 210;
				break; // goto 210;
			case 50:
				// L50
				//printState("trsbox-50-start", 0, null, xopt, null, null, d, null, null, xpt);
				double resid = delsq, ds = 0, shs = 0;
				for (int i = 0; i < n; i ++)
					if (xbdi[i] == 0)
					{
						double di = d[i], si = s[i];
						resid -= di * di;
						ds += si * di;
						shs += si * hs[i];
					}
				if (resid <= 0)
				{
					// goto 90
					crvmin = 0;
					label = 100;
					break;
				}
				double
					temp = sqrt(stepsq * resid + ds * ds),
					blen = ds < 0 ? (temp - ds) / stepsq : resid / (temp + ds),
					stplen = shs > 0 ? min(blen, gredsq / shs) : blen;
				//printState("trsbox-50-checkstate:\n" +
				//	"stepsq="+stepsq+", resid="+resid+", " +
				//	"ds="+ds+", gredsq="+gredsq+ ", blen="+blen, 0, null, null, null, null, null, null, null, null);
				/*
				 * Reduce STPLEN if necessary in order to preserve the simple bounds,
				 * letting IACT be the index of the new constrained variable.
				 */
				iact = -1;
				for (int i = 0; i < n; i++)
					if (s[i] != 0)
					{
						double xsum = xopt[i] + d[i];
						temp = s[i] > 0 ? (su[i] - xsum) / s[i] : (sl[i] - xsum) / s[i];
						if (temp < stplen)
						{
							stplen = temp;
							iact = i;
						}
					}
				// Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q.
				double sdec = 0;
				if (stplen > 0)
				{
					//printState("trsbox-50-ifstplen>0", 0, null, xopt, null, null, d, null, null, xpt);
					iterc++;
					temp = shs/stepsq;
					if (iact == -1 && temp > 0)
					{
						crvmin = min(crvmin, temp);
						if (crvmin == -1) crvmin = temp;
					}
					ggsav = gredsq;
					gredsq = 0;
					//printState("trsbox-50-ifstplen>0-beforeloop-stplen="+stplen, 0, null, xopt, null, null, d, null, null, xpt);
					//printState("trsbox-50-ifstplen>0-beforeloop-gnew="+toString(gnew), 0, null, null, null, null, null, null, null, null);
					//printState("trsbox-50-ifstplen>0-beforeloop-hs="+toString(hs), 0, null, null, null, null, null, null, null, null);
					for (int i = 0; i < n; i++)
					{
						gnew[i] = gnew[i] + stplen * hs[i];
						if (xbdi[i] == 0)
						{
							double val = gnew[i];
							gredsq += val * val;
						}
						d[i] = d[i] + stplen * s[i];
					}
					//printState("trsbox-50-ifstplen>0-afterloop-gredsq="+gredsq, 0, null, xopt, null, null, d, null, null, xpt);
					sdec = max(stplen*(ggsav - 0.5 * stplen * shs), 0);
					qred += sdec;
				}
				// Restart the conjugate gradient method if it has hit a new bound.
				if (iact >= 0)
				{
					nact++;
					xbdi[iact] = 1;
					if (s[iact] < 0) xbdi[iact] = -1;
					double val = d[iact];
					delsq -= val * val;
					if (delsq <= 0) 
					{
						crvmin = 0;
						label = 100; // goto 90
						break;
					}
					// goto 20;
					beta = 0;
					label = 30;
					break;
				}
				/*
				 * If STPLEN is less than BLEN, then either apply another conjugate
				 * gradient iteration or RETURN.
				 */
				if (stplen < blen)
				{
					if (iterc == itermax || sdec <= 0.01 * qred)
					{
						// label = 190; // goto 190
						loop = false;
						break;
					}
					beta = gredsq / ggsav;
					label = 30; // goto 30;
					break;
				}
				crvmin = 0;
				label = 100;
				break;
			case 100: // L100
				//printState("trsbox-100-start", 0, null, xopt, null, null, d, null, null, xpt);
				/*
				 * Prepare for the alternative iteration by calculating some scalars
				 * and by multiplying the reduced D by the second derivative matrix of
				 * Q, where S holds the reduced D in the call of GGMULT.
				 */
				if (nact >= n - 1)
				{
					// label = 190; // goto 190
					loop = false;
					break;
				}
				dredsq = dredg = gredsq = 0;
				for (int i = 0; i < n; i++)
				{
					if (xbdi[i] == 0)
					{
						double di = d[i], gi = gnew[i];
						dredsq += di * di;
						dredg += di * gi;
						gredsq += gi * gi;
						s[i] = di;
					} else s[i] = 0;
				}
				//printState("trsbox-100-checkstate:\n" +
				//		"stepsq="+stepsq+", gredsq="+gredsq, 0, null, null, null, null, null, null, null, null);
				itcsav = iterc;
				label = 210; // goto 210;
				break;
			case 120:
				//printState("trsbox-120-start", 0, null, xopt, null, null, d, null, null, xpt);
				/*
				 * Let the search direction S be a linear combination of the reduced D
				 * and the reduced G that is orthogonal to the reduced D.
				 */
				iterc++;
				temp = gredsq * dredsq - dredg * dredg;
				if (temp <= qred * 1e-4 * qred)
				{
					// label = 190; // goto 190
					loop = false;
					break;
				}
				temp = sqrt(temp);
				for (int i = 0; i < n; i++)
					s[i] = xbdi[i] == 0 ? (dredg * d[i] - dredsq * gnew[i])/temp : 0;
				sredg = -temp;
				/*
				 * By considering the simple bounds on the variables, calculate an upper
				 * bound on the tangent of half the angle of the alternative iteration,
				 * namely ANGBD, except that, if already a free variable has reached a
				 * bound, there is a branch back to label 100 after fixing that variable.
				 */
				angbd = 1;
				iact = -1;
				label = 0;
				for (int i = 0; i < n; i++)
				{
					if (xbdi[i] == 0)
					{
						double
							tempa = xopt[i] + d[i] - sl[i],
							tempb = su[i] - xopt[i] - d[i];
						if (tempa <= 0)
						{
							nact++;
							xbdi[i] = -1;
							label = 100;
							break;
						} else if (tempb <= 0)
						{
							nact++;
							xbdi[i] = 1;
							label = 100;
							break;
						}
						double
							// ratio = 1, // seems unused
							di = d[i],
							si = s[i],
							ssq = di * di + si * si,
							val = xopt[i] - sl[i];
						temp = ssq - val * val;
						if (temp > 0)
						{
							temp = sqrt(temp) - s[i];
							if (angbd * temp > tempa)
							{
								angbd = tempa / temp;
								iact = i;
								xsav = -1;
							}
						}
						val = su[i] - xopt[i];
						temp = ssq - val * val;
						if (temp > 0)
						{
							temp = sqrt(temp) + s[i];
							if (angbd * temp > tempb)
							{
								angbd = tempb / temp;
								iact = i;
								xsav = 1;
							}
						}
					} // end if (xbdi[i] == 0)
				} // end for (int i = 0; i < n; i++)
				// Calculate HHD and some curvatures for the alternative iteration.
				if (label == 0)
					label = 210;
				break;
			case 150:
				//printState("trsbox-150-start", 0, null, xopt, null, null, d, null, null, xpt);
				double dhs = 0, dhd = 0; shs = 0;
				for (int i = 0; i < n; i++)
				{
					if (xbdi[i] == 0)
					{
						shs += s[i] * hs[i];
						dhs += d[i] * hs[i];
						dhd += d[i] * hred[i];
					}
				}
				/*
				 * Seek the greatest reduction in Q for a range of equally spaced values
				 * of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle of
				 * the alternative iteration.
				 */
				double redmax = redsav = 0, rdprev = 0, rdnext = 0, angt = 0, sth;
				int isav = 0, iu = (int) (17. * angbd + 3.1);
				for (int i = 1; i <= iu; i++)
				{
					angt = (angbd * i) / iu;
					sth = (angt + angt) / (1 + angt * angt);
					temp = shs + angt * (angt * dhd - dhs - dhs);
					double rednew = sth * (angt * dredg - sredg - 0.5 * sth * temp);
					if (rednew > redmax)
					{
						redmax = rednew;
						isav = i;
						rdprev = redsav;
					} else if (i == isav + 1)
						rdnext = rednew;
					redsav = rednew;
				}
				/*
				 * Return if the reduction is zero. Otherwise, set the sine and cosine
				 * of the angle of the alternative iteration, and calculate SDEC.
				 */
				if (isav == 0)
				{
					//label = 190; break; // goto 190;
					loop = false;
					break;
				}
				if (isav < iu)
				{
					temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
					angt = angbd * (isav + 0.5 * temp) / iu;
				}
				double cth = (1 - angt * angt) / (1 + angt * angt);
				sth = (angt + angt) / (1 + angt * angt);
				temp = shs + angt * (angt * dhd - dhs - dhs);
				sdec = sth * (angt * dredg - sredg - 0.5 * sth * temp);
				if (sdec <= 0)
				{
					// label = 190; break; // goto 190;
					loop = false;
					break;
				}
				/*
				 * Update GNEW, D and HRED. If the angle of the alternative iteration
				 * is restricted by a bound on a free variable, that variable is fixed
				 * at the bound.
				 */
				dredg = gredsq = 0;
				for (int i = 0; i < n; i++)
				{
					gnew[i] = gnew[i] + (cth - 1) * hred[i] + sth * hs[i];
					if (xbdi[i] == 0)
					{
						d[i] = cth * d[i] + sth * s[i];
						double val = gnew[i];
						dredg += d[i] * val;
						gredsq += val * val;
					}
					hred[i] = cth * hred[i] + sth * hs[i];
				}
				//printState("trsbox-150-checkstate:\n" +
				//		"stepsq="+stepsq+", gredsq="+gredsq, 0, null, null, null, null, null, null, null, null);
				qred += sdec;
				if (iact >= 0 && isav == iu)
				{
					nact++;
					xbdi[iact] = xsav;
					label = 100; break; // goto 100
				}
				/*
				 * If SDEC is sufficiently small, then RETURN after setting XNEW to
				 * XOPT+D, giving careful attention to the bounds.
				 */
				if (sdec > 0.01 * qred)
				{
					label = 120; break; // goto 120
				}
				loop = false; // goto 190
				// End untested branch
				break;
			case 210:
				//printState("trsbox-210-start", 0, null, xopt, null, null, d, null, null, xpt);
				/*
				 * The following instructions multiply the current S-vector by the second
				 * derivative matrix of the quadratic model, putting the product in HS.
				 * They are reached from three different parts of the software above and
				 * they can be regarded as an external subroutine.
				 */
				int ih = 0;
				for (int j = 0; j < n; j++)
				{
					hs[j] = 0;
					for (int i = 0; i <= j; i++)
					{
						if (i < j)
							hs[j] = hs[j] + hq[ih] * s[i];
						hs[i] = hs[i] + hq[ih] * s[j];
						ih++;
					}
				}
				for (int k = 0; k < npt; k++)
				{
					if (pq[k] != 0)
					{
						temp = 0;
						for (int j = 0; j < n; j++)
							temp += xpt[k][j] * s[j];
						temp *= pq[k];
						for (int i = 0; i < n; i++)
							hs[i] = hs[i] + temp * xpt[k][i];
					}
				}
				if (crvmin != 0)
				{
					label = 50; break; // goto 50;
				}
				if (iterc > itcsav)
				{
					label = 150; break; // goto 150;
				}
				for (int i = 0; i < n; i++)
					hred[i] = hs[i];
				label = 120; // goto 120;
				break;
			} // end humongous switch
		} // end for (;;) L30	
		// Label 190
		//printState("trsbox-190-start", 0, null, xopt, null, null, d, null, null, xpt);
		dsq = 0;
		for (int i = 0; i < n; i++)
		{
			xnew[i] = max(min(xopt[i] + d[i], su[i]), sl[i]);
			if (xbdi[i] == -1) xnew[i] = sl[i];
			if (xbdi[i] == 1) xnew[i] = su[i];
			double val = d[i] = xnew[i] - xopt[i];
			dsq += val * val;
		}
		return new double[] {dsq, crvmin};
	}

	public static void main(String[] args)
	{
		MultivariableFunction ifun = new MultivariableFunction() {
			public double eval(double... x) {
				int n = x.length;
				double f = 0;
				for (int i = 3; i < n; i += 2)
					for (int j = 1; j < i - 1; j += 2)
					{
						double v1 = x[i-1] - x[j-1], v2 = x[i] - x[j];
						f += 1 / sqrt(max(v1 * v1 + v2 * v2, 1e-6));
					}
				return f;
			}
		};
		double rhobeg = 1e-1, rhoend = 1e-6;
		int maxfun = 500000, max_n = 80, n = 10;
		System.out.println("rhobeg = " + rhobeg + ", rhoend = " + rhoend);

		/*
		 * TODO The program is less optimal with the following input:
		 * Java output:
		 * n = 80, npt = 86
		 * Number of function calls = 19302
		 * Minimum value of F = 769.8202594453368
		 * 
		 * n = 80, npt = 161
		 * Number of function calls = 18331
		 * Minimum value of F = 769.820259440993
		 * 
		 * Fortran output:
		 * n = 80, npt = 86
		 * Number of function values = 13732
		 * Least value of F =  7.692677518765548D+02
		 * 
		 * n = 80, npt = 161
		 * Number of function values =  4907
		 * Least value of F =  7.698202594229921D+02
		 */
		do {
			int npt = n+6;
			double[]
				xl = new double[n],
				xu = new double[n],
				x = new double[n];
			for (int i = 0; i < n; i++)
			{
				xl[i] = -1;
				xu[i] = 1;
			}
			for (int i = 0; i < n / 2; i++)
			{
				double temp = (i + 1) * 4 * PI / n;
				x[2 * i] = cos(temp);
				x[2 * i + 1] = sin(temp);
			}
			System.out.println("n = " + n + ", npt = " + npt);
			OptimizationResult result = bobyqa(x, xl, xu, ifun, npt, rhobeg, rhoend, maxfun, true);
			System.out.println(result);
			//writer.close();
			//System.exit(0);
			npt = 2*n + 1;
			System.out.println("n = " + n + ", npt = " + npt);
			result = bobyqa(x, xl, xu, ifun, npt, rhobeg, rhoend, maxfun, true);
			System.out.println(result);
			n *= 2;
		} while (n <= max_n);
	}
}
