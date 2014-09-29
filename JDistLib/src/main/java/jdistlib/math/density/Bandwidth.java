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
package jdistlib.math.density;

import jdistlib.math.Optimization;
import jdistlib.math.UnivariateFunction;

import static java.lang.Math.*;
import static jdistlib.math.VectorMath.*;

public class Bandwidth {
	public static final Bandwidth NRD0 = new Bandwidth() {
		public double calculate(double[] x) { return NRD0(x); }
	},
	NRD = new Bandwidth() {
		public double calculate(double[] x) { return NRD(x); }
	},
	UCV = new Bandwidth() {
		public double calculate(double[] x) { return CV(x, nb, lower, upper, tol, true); }
	},
	BCV = new Bandwidth() {
		public double calculate(double[] x) { return CV(x, nb, lower, upper, tol, false); }
	},
	SJ_STE = new Bandwidth() {
		public double calculate(double[] x) { return SJ(x, nb, lower, upper, tol, false); }
	},
	SJ_DPI = new Bandwidth() {
		public double calculate(double[] x) { return SJ(x, nb, lower, upper, tol, true); }
	};
	// This structure allows manual specification

	double lower = Double.NaN, upper = Double.NaN, tol = Double.NaN;
	int nb = 1000;

	public double calculate(double[] x) {
		throw new RuntimeException("Unknown bandwidth rule");
	}

	public static final double NRD0(double[] x) {
		double
		hi = sd(x),
		lo = min(hi, iqr(x) / 1.34);
		if (lo == 0) {
			if (hi == 0) {
				lo = abs(x[0]);
				if (lo == 0) lo = 1;
			} else lo = hi;
		}
		return 0.9 * lo * pow(x.length, -0.2);
	}

	public static final double NRD(double[] x) {
		double h = iqr(x) / 1.34;
		return 1.06 * min(sqrt(var(x)), h) * pow(x.length, -0.2);
	}
	public static final double CV(double[] x, int nb, double lower, double upper, double tol, boolean isUnbiased) {
		if (nb <= 0) throw new RuntimeException();
		int n = x.length;
		double hmax = 1.144 * sd(x) * pow(n, -0.2);
		if (Double.isNaN(lower)) lower = 0.1 * hmax;
		if (Double.isNaN(upper)) upper = hmax;
		if (Double.isNaN(tol)) tol = 0.1 * lower;
		int[] cnt = new int[nb];
		double xmin, xmax, dd;
		xmin = xmax = x[0];
		for (int i = 1; i < n; i++) {
			if (xmin > x[i]) xmin = x[i];
			if (xmax < x[i]) xmax = x[i];
		}
		dd = ((xmax - xmin) * 1.01) / nb;
		for (int i = 1; i < n; i++) {
			int ii = (int)(x[i] / dd);
			for (int j = 0; j < i; j++) {
				int jj = (int)(x[j] / dd);
				cnt[abs(ii - jj)]++;
			}
		}
		UnivariateFunction fun = null;
		if (isUnbiased) {
			fun = new UnivariateFunction() {
				double d; int n;
				int[] cnt;
				public void setParameters(double... params) {
					n = (int) params[0];
					d = params[1];
				}

				public void setObjects(Object... obj) {
					cnt = (int[]) obj[0];
				}

				public double eval(double h) {
					int nbin = cnt.length;
					final int DELMAX = 1000;
					double sum = 0.0, term, u;
					for (int i = 0; i < nbin; i++) {
						double delta = i * d / h;
						delta *= delta;
						if (delta >= DELMAX) break;
						term = exp(-delta / 4) - sqrt(8.0) * exp(-delta / 2);
						sum += term * cnt[i];
					}
					u = 1 / (2 * n * h * sqrt(PI)) + sum / (n * n * h * sqrt(PI));
					return u;
				}
			};
		} else {
			fun = new UnivariateFunction() {
				double d; int n;
				int[] cnt;
				public void setParameters(double... params) {
					n = (int) params[0];
					d = params[1];
				}
				
				public void setObjects(Object... obj) {
					cnt = (int[]) obj[0];
				}
				
				public double eval(double h) {
					int nbin = cnt.length;
					final int DELMAX = 1000;
					double sum = 0.0, term, u;
					for (int i = 0; i < nbin; i++) {
						double delta = i * d / h;
						delta *= delta;
						if (delta >= DELMAX) break;
						term = exp(-delta / 4) * (delta * delta - 12 * delta + 12);
						sum += term * cnt[i];
					}
					u = 1 / (2 * n * h * sqrt(PI)) + sum / (64 * n * n * h * sqrt(PI));
					return u;
				}
			};
		}
		fun.setParameters(n, dd);
		fun.setObjects(cnt);
		double h = Optimization.optimize(fun, lower, upper, tol, 1000);
		if(h < lower+tol | h > upper-tol)
			System.err.println("Warning: minimum occurred at one end of the range");
		return h;
	}

	public static final double SJ(double[] x, int nb, double lower, double upper, double tol, boolean isDPI) {
		if (nb <= 0) throw new RuntimeException();
		int n = x.length;
		int[] cnt = new int[nb];
		double xmin, xmax, dd;
		xmin = xmax = x[0];
		for (int i = 1; i < n; i++) {
			if (xmin > x[i]) xmin = x[i];
			if (xmax < x[i]) xmax = x[i];
		}
		dd = ((xmax - xmin) * 1.01) / nb;
		for (int i = 1; i < n; i++) {
			int ii = (int)(x[i] / dd);
			for (int j = 0; j < i; j++) {
				int jj = (int)(x[j] / dd);
				cnt[abs(ii - jj)]++;
			}
		}
		double
			scale = min(sd(x), iqr(x)/1.349),
			a = 1.24 * scale * pow(n, (-1.0/7)),
			b = 1.23 * scale * pow(n, (-1.0/9)),
			c1 = 1/(2*sqrt(PI)*n),
			TD = -bw_phi6(n, dd, cnt, b);
		if (Double.isInfinite(TD) || TD <= 0)
			throw new RuntimeException("sample is too sparse to find TD");
		if (isDPI)
			return pow(c1/bw_phi4(n, dd, cnt, pow(2.394/(n * TD), 1.0/7)), 1.0/5);
		// SJ_STE
		double hmax = 1.144 * scale * pow(n, -0.2);
		boolean isLimitUnspecified = Double.isNaN(lower) || Double.isNaN(upper);
		if (Double.isNaN(lower)) lower = 0.1 * hmax;
		if (Double.isNaN(upper)) upper = hmax;
		if (Double.isNaN(tol)) tol = 0.1 * lower;
		double alph2 = 1.357*pow(bw_phi4(n, dd, cnt, a)/TD, 1.0/7);
		if (Double.isInfinite(alph2))
			throw new RuntimeException("sample is too sparse to find alph2");
		UnivariateFunction fSD = new UnivariateFunction() {
			int n;
			double d, c1, alph2;
			int[] cnt;
			public double eval(double h) {
				double v = pow(c1 / bw_phi4(n, d, cnt, alph2 * pow(h, 5.0/7)), 0.2) - h;
				return v;
			}

			public void setParameters(double... params) {
				n = (int) params[0]; d = params[1];
				c1 = params[2]; alph2 = params[3];
			}

			public void setObjects(Object... obj) {
				cnt = (int[]) obj[0];
			}
		};
		fSD.setParameters(n, dd, c1, alph2);
		fSD.setObjects(cnt);
		int itry = 1;
		while (fSD.eval(lower) * fSD.eval(upper) > 0) {
			if (itry > 99 || !isLimitUnspecified)
				throw new RuntimeException("no solution in the specified range of bandwidths");
			if((itry & 1) == 0) upper *= 1.2; else lower /= 1.2;
			itry++;
		}
		double res = Optimization.zeroin(fSD, lower, upper, tol, 1000);
		return res;
	}

	private static final double bw_phi4(int n, double d, int[] cnt, double h) {
		int nbin = cnt.length;
		final int DELMAX = 1000;
		double sum = 0.0, term, u;
		for (int i = 0; i < nbin; i++) {
			double delta = i * d / h; delta *= delta;
			if (delta >= DELMAX) break;
			term = exp(-delta / 2) * (delta * delta - 6 * delta + 3);
			sum += term * cnt[i];
		}
		sum = 2 * sum + n * 3;	/* add in diagonal */
		u = sum / (n * (n - 1) * pow(h, 5.0) * sqrt(2 * PI));
		return u;
	}

	private static final double bw_phi6(int n, double d, int[] cnt, double h) {
		int nbin = cnt.length;
		final int DELMAX = 1000;
		double sum = 0.0, term, u;
		for (int i = 0; i < nbin; i++) {
			double delta = i * d / h; delta *= delta;
			if (delta >= DELMAX) break;
			term = exp(-delta / 2) *
					(delta * delta * delta - 15 * delta * delta + 45 * delta - 15);
			sum += term * cnt[i];
		}
		sum = 2 * sum - 15 * n;	/* add in diagonal */
		u = sum / (n * (n - 1) * pow(h, 7.0) * sqrt(2 * PI));
		return u;
	}

	public int getNumBins() {
		return nb;
	}

	public void setNumBins(int nb) {
		this.nb = nb;
	}

	public double getLower() {
		return lower;
	}

	public void setLower(double lower) {
		this.lower = lower;
	}

	public double getUpper() {
		return upper;
	}

	public void setUpper(double upper) {
		this.upper = upper;
	}

	public double getTolerance() {
		return tol;
	}

	public void setTolerance(double tol) {
		this.tol = tol;
	}

	/*
	public static final void main(String[] args) {
		// Taken from precip data
		double[] v = new double[] {67.0, 54.7, 7.0, 48.5, 14.0, 17.2, 20.7, 13.0, 43.4, 40.2, 38.9, 54.5, 59.8, 48.3, 22.9,
		11.5, 34.4, 35.1, 38.7, 30.8, 30.6, 43.1, 56.8, 40.8, 41.8, 42.5, 31.0, 31.7, 30.2, 25.9,
		49.2, 37.0, 35.9, 15.0, 30.2,  7.2, 36.2, 45.5,  7.8, 33.4, 36.1, 40.2, 42.7, 42.5, 16.2,
		39.0, 35.0, 37.0, 31.4, 37.6, 39.9, 36.2, 42.8, 46.4, 24.7, 49.1, 46.0, 35.9,  7.8, 48.2,
		15.2, 32.5, 44.7, 42.6, 38.8, 17.4, 40.8, 29.1, 14.6, 59.2};
		System.out.println(Bandwidth.NRD0.calculate(v));
		System.out.println(Bandwidth.NRD.calculate(v));
		System.out.println(Bandwidth.UCV.calculate(v));
		System.out.println(Bandwidth.BCV.calculate(v));
		System.out.println(Bandwidth.SJ_DPI.calculate(v));
		System.out.println(Bandwidth.SJ_STE.calculate(v));
	}
	//*/
}
