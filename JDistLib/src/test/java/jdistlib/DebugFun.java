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
 *  http://www.gnu.org/licenses/gpl-2.0.html
 */
package jdistlib;

import static java.lang.Math.abs;
import jdistlib.math.VectorMath;

public class DebugFun {

	public static final double rErr(double approx, double truval) {
		return rErr(approx, truval, 1e-30);
	}

	public static final double rErr(double approx, double truval, double eps) {
		return abs(truval) >= eps ? 1 - approx / truval : approx - truval;
	}

	public static final boolean isEqual(double a, double b) {
		return VectorMath.isEqual(a, b, TestDPQR.defaultNumericalError);
	}

	public static final boolean isEqualScaled(double a, double b) {
		return VectorMath.isEqualScaled(a, b, TestDPQR.defaultNumericalError);
	}

	public static final boolean allEqual(double[] a, double[] b) {
		return VectorMath.allEqual(a, b, TestDPQR.defaultNumericalError);
	}

	public static final boolean allEqualScaled(double[] a, double[] b) {
		return VectorMath.allEqualScaled(a, b, TestDPQR.defaultNumericalError);
	}

	public static final boolean printBool(boolean b) {
		System.out.println(b ? "[1] TRUE" : "[1] FALSE");
		return b;
	}

	public static final boolean printBool(boolean... b) {
		if (b == null || b.length == 0) return false;
		System.out.print("[1]");
		boolean bb = true;
		for (int i = 0; i < b.length; i++) {
			System.out.print(b[i] ? " TRUE" : " FALSE");
			bb = bb & b[i];
		}
		System.out.println();
		return bb;
	}

	public static final boolean printAllEqual(double[] a, double[] b, double tol) {
		boolean v = VectorMath.allEqual(a, b, tol);
		printBool(v);
		if (v) return true;
		int n = a.length;
		boolean[] vv = new boolean[n];
		for (int i = 0; i < a.length; i++)
			vv[i] = VectorMath.isEqual(a[i], b[i], tol);
		System.out.print("True values: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(a[i]+ " ");
		System.out.println();
	
		System.out.print("Results: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(b[i]+ " ");
		System.out.println();
	
		System.out.print("|Diff|: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(abs(a[i]-b[i])+ " ");
		System.out.println();
		return false;
	}

	public static final boolean printAllEqual(double[] a, double[] b) {
		return printAllEqual(a, b, TestDPQR.defaultNumericalError);
	}

	public static final boolean printAllEqualScaled(double[] a, double[] b, double tol) {
		boolean v = VectorMath.allEqualScaled(a, b, tol);
		printBool(v);
		if (v) return true;
		int n = a.length;
		boolean[] vv = new boolean[n];
		for (int i = 0; i < a.length; i++)
			vv[i] = VectorMath.isEqualScaled(a[i], b[i], tol);
		System.out.print("True values: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(a[i]+ " ");
		System.out.println();
	
		System.out.print("Results: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(b[i]+ " ");
		System.out.println();
	
		System.out.print("Relative Diff: ");
		for (int i = 0; i < n; i++)
			if (!vv[i])
				System.out.print(abs(a[i]-b[i])/a[i]+ " ");
		System.out.println();
		return false;
	}

	public static final boolean printAllEqualScaled(double[] a, double[] b) {
		return printAllEqualScaled(a, b, TestDPQR.defaultNumericalError);
	}

}
