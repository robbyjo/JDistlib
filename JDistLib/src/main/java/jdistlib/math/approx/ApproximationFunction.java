/*  This program is free software; you can redistribute it and/or modify
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
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package jdistlib.math.approx;

import jdistlib.math.UnivariateFunction;

/**
 * Create an approximation function. This is pretty much like R's
 * <tt>approxfun</tt>. I'm emulating it here. I don't deal with
 * NaN and Infinity. I don't do sanity check and no uniqueness
 * check here. So, watch out.
 * 
 * @author Roby Joehanes
 *
 */
public class ApproximationFunction implements UnivariateFunction
{
	protected double
		mX[],
		mY[],
		mLo,
		mHi,
		mLeftCompromise,
		mRightCompromise;
	protected ApproximationType mType;

	public ApproximationFunction(ApproximationType t, double[] x, double[] y, double lo, double hi, double compromise)
	{
		assert (compromise >= 0 && compromise <= 1 && x.length == y.length &&
			(t == ApproximationType.CONSTANT || t == ApproximationType.LINEAR));
		mX = x;
		mY = y;
		mLo = lo;
		mHi = hi;
		mLeftCompromise = compromise;
		mRightCompromise = 1 - compromise;
		mType = t;
	}

	/* (non-Javadoc)
	 * @see qmath.IFunction#eval(double)
	 */
	public double eval(double x)
	{
		int
			left = 0,
			right = mX.length - 1;
		if (x < mX[left])
			return mLo;
		if (x > mX[right])
			return mHi;
		while(left < right - 1)
		{
			int mid = (left + right)/2;
			if(x < mX[mid])
				right = mid;
			else
				left = mid;
		}
		if(x == mX[right])
			return mY[right];
		if(x == mX[left])
			return mY[left];
		switch (mType)
		{
			case CONSTANT:
				x = mY[left] * mLeftCompromise + mY[right] * mRightCompromise;
				break;
			case LINEAR:
				x = mY[left] + (mY[right] - mY[left]) * ((x - mX[left])/(mX[right] - mX[left]));
				break;
			default:
				throw new RuntimeException();
		}
		return x;
	}

	public void setParameters(double... params) {
	}

	public void setObjects(Object... obj) {
	}
}
