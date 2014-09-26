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

/**
 * 
 * @author Roby Joehanes
 *
 */
public class SmoothSplineResult {
	static final String sLn = System.getProperty("line.separator"); //$NON-NLS-1$
	public double[]
		mKnots,
		mCoefficients,
		mLeverage,
		mSmoothedValues;

	public SmoothSplineCriterion mCriterion;

	public double
		mFitCVScore, // Cross Validation (CV) score from the original method
		mCVScore, // Actual CV score. Should be the same as above if there's no zero weights
		mSmoothingParameter, // Only meaningful if initial smoothing param is not supplied
		mLambda,
		mEstimatedDF,
		mPenalizedCriterion,
		mXMin, // For estimation
		mXMax; // For estimation

	public int mIterNo;
	public boolean mHasFactorizationProblems;

	/**
	 * Debugging
	 */
	@Override
	public String toString()
	{
		StringBuilder buf = new StringBuilder();
		buf.append("Minimizing criterion: " + mCriterion + sLn); //$NON-NLS-1$
		buf.append("Smoothing parameter: " + mSmoothingParameter + sLn); //$NON-NLS-1$
		buf.append("Lambda: " + mLambda + sLn); //$NON-NLS-1$
		buf.append("Num. iterations: " + mIterNo+ sLn); //$NON-NLS-1$
		buf.append("Equivalent degrees of freedom (DF): " + mEstimatedDF + sLn); //$NON-NLS-1$
		buf.append("Penalized criterion: " + mPenalizedCriterion + sLn); //$NON-NLS-1$
		if (mCriterion == SmoothSplineCriterion.CV)
			buf.append("PRESS: " + mCVScore + sLn); //$NON-NLS-1$
		else
			buf.append("GCV: " + mCVScore + sLn); //$NON-NLS-1$
		return buf.toString();
	}
}
