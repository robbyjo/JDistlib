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
		mCompromise;
	protected ApproximationType mType;

	public ApproximationFunction(ApproximationType t, double[] x, double[] y, double lo, double hi, double compromise)
	{
		assert (compromise >= 0 && compromise <= 1 && x.length == y.length &&
			(t == ApproximationType.CONSTANT || t == ApproximationType.LINEAR));
		mX = x;
		mY = y;
		mLo = lo;
		mHi = hi;
		mCompromise = compromise;
		mType = t;
	}

	/* (non-Javadoc)
	 * @see qmath.IFunction#eval(double)
	 */
	public double eval(double x) {
		switch (mType) {
			case CONSTANT:
				return constant(x, mX, mY, mLo, mHi, mCompromise);
			case LINEAR:
				return linear(x, mX, mY, mLo, mHi);
			default:
				throw new RuntimeException();
		}
	}

	public void setParameters(double... params) {
	}

	public void setObjects(Object... obj) {
	}

	/**
	 * Linear approximation
	 * @param v
	 * @param x
	 * @param y
	 * @param lo
	 * @param hi
	 * @return Approximated value
	 */
	public static final double linear(double v, double[] x, double[] y, double lo, double hi) {
		int
			left = 0,
			right = x.length - 1;
		if (v < x[left])
			return lo;
		if (v > x[right])
			return hi;
		while(left < right - 1) {
			int mid = (left + right)/2;
			if(v < x[mid]) right = mid; else left = mid;
		}
		if(v == x[right])
			return y[right];
		if(v == x[left])
			return y[left];
		return v = y[left] + (y[right] - y[left]) * ((v - x[left])/(x[right] - x[left]));
	}

	/**
	 * Constant approximation
	 * @param v
	 * @param x
	 * @param y
	 * @param lo
	 * @param hi
	 * @param compromise
	 * @return Approximated value
	 */
	public static final double constant(double v, double[] x, double[] y, double lo, double hi, double compromise) {
		int
			left = 0,
			right = x.length - 1;
		if (v < x[left])
			return lo;
		if (v > x[right])
			return hi;
		while(left < right - 1) {
			int mid = (left + right)/2;
			if(v < x[mid]) right = mid; else left = mid;
		}
		if(v == x[right])
			return y[right];
		if(v == x[left])
			return y[left];
		return y[left] * compromise + y[right] * (1-compromise);
	}
}
