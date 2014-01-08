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
package jdistlib.math;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.pow;

/**
 * 
 * @author Roby Joehanes
 *
 */
public class VectorMath {
	public static final double[] vpow(double x, double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	public static final double[] vpow(double[] x, double e) {
		double[] v = new double[x.length];
		for (int i = 0; i < x.length; i++)
			v[i] = pow(x[i], e);
		return v;
	}

	public static final double[] vpow(double[] x, double[] e) {
		if (x.length != e.length)
			throw new RuntimeException();
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x[i], e[i]);
		return v;
	}

	public static final double[] vpow(double x, int[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = pow(x, e[i]);
		return v;
	}

	public static final double[] vplus(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]+b[i];
		return v;
	}

	public static final double[] vplus(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a+b[i];
		return v;
	}

	public static final double[] vplus(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]+b;
		return v;
	}

	public static final double[] vmin(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = -e[i];
		return v;
	}

	public static final double[] vmin(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]-b[i];
		return v;
	}

	public static final double[] vmin(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]-b;
		return v;
	}

	public static final double[] vmin(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a-b[i];
		return v;
	}

	public static final double[] vtimes(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a*b[i];
		return v;
	}

	public static final double[] vtimes(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]*b;
		return v;
	}

	public static final double[] vtimes(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]*b[i];
		return v;
	}

	public static final double[] vdiv(double[] a, double b) {
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]/b;
		return v;
	}

	public static final double[] vdiv(double a, double[] b) {
		double[] v = new double[b.length];
		for (int i = 0; i < b.length; i++)
			v[i] = a/b[i];
		return v;
	}

	public static final double[] vdiv(double[] a, double[] b) {
		if (a.length != b.length)
			throw new RuntimeException();
		double[] v = new double[a.length];
		for (int i = 0; i < a.length; i++)
			v[i] = a[i]/b[i];
		return v;
	}

	public static final double[] vcomp(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = 1-e[i];
		return v;
	}

	public static final double[] vabs(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = abs(e[i]);
		return v;
	}

	public static final double[] vexp(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = exp(e[i]);
		return v;
	}

	public static final double[] vlog(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = log(e[i]);
		return v;
	}

	public static final double[] vlog1pComps(double[] e) {
		double[] v = new double[e.length];
		for (int i = 0; i < e.length; i++)
			v[i] = log1p(-e[i]);
		return v;
	}

	public static final double[] diff(double[] e, int lag, int order) {
		double[] v = new double[e.length];
		System.arraycopy(e, 0, v, 0, e.length);
		int vlen = v.length;
		for (int i = 0; i < order; i++, vlen -= lag)
			for (int j = lag; j < vlen; j++)
				v[j - lag] = v[j] - v[j - lag];
		e = new double[e.length - lag * order];
		System.arraycopy(v, 0, e, 0, e.length);
		return e;
	}

	public static final boolean allFinite(double[] e) {
		for (double _e : e)
			if (MathFunctions.isInfinite(_e))
				return false;
		return true;
	}

	public static final boolean allLt(double[] e, double v) {
		for (double _e : e)
			if (_e >= v)
				return false;
		return true;
	}

	public static final boolean allGt(double[] e, double v) {
		for (double _e : e)
			if (_e <= v)
				return false;
		return true;
	}
}
