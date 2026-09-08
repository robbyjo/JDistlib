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

import java.util.Arrays;

import org.jtransforms.fft.DoubleFFT_1D;

import jdistlib.math.MathFunctions;
import jdistlib.math.VectorMath;
import jdistlib.math.approx.ApproximationFunction;
import jdistlib.math.approx.ApproximationType;
import jdistlib.util.Utilities;

import static java.lang.Math.*;
import static jdistlib.math.Constants.M_LN2;

/**
 * Corresponds to R's density function. Using JTransform's FFT routine
 * @author Roby Joehanes
 *
 */
public class Density {
	private Density() {}
	public double[] x, y;

	public static final Density density(double[] x, Bandwidth bandwidth, double adjust, Kernel kernel,
		double[] weights, double width, int n, double from, double to, double cut) {
		if (x == null || x.length == 0 || kernel == null || n < 2 || n > (1 << 26))
			throw new IllegalArgumentException("Nonempty data, a kernel and 2 <= n <= 2^26 are required");
		int N = x.length, nx = 0;
		if (weights != null && weights.length != N)
			throw new IllegalArgumentException("Data and weights must have equal lengths");
		double[] newx = new double[N], new_wt = new double[N];
		for (int i = 0; i < N; i++) {
			if (weights != null && (!Double.isFinite(weights[i]) || weights[i] < 0))
				throw new IllegalArgumentException("Weights must be finite and nonnegative");
			double wt = weights == null ? 1. / N : weights[i];
			if (MathFunctions.isFinite(x[i])) {
				newx[nx] = x[i];
				new_wt[nx] = wt;
				nx++;
			}
		}
		if (nx == 0 || (bandwidth != null && nx < 2))
			throw new IllegalArgumentException("Insufficient finite observations for the requested bandwidth");
		// Nonfinite observations contribute no kernel; retain each finite
		// observation's original mass exactly once, including explicit weights.
		x = Arrays.copyOf(newx, nx);
		weights = Arrays.copyOf(new_wt, nx);

		int n_user = n;
		n = max(n, 512);
		if (n > 512) n = 1 << ((int) ceil(log(n) / M_LN2));
		double bw = bandwidth != null ? bandwidth.calculate(x) : width / kernel.getFactor();
		bw *= adjust;
		if (!Double.isFinite(bw) || bw <= 0) throw new IllegalArgumentException("Bandwidth must be positive and finite");
		if (Double.isNaN(from)) from = VectorMath.min(x) - cut * bw;
		if (Double.isNaN(to)) to = VectorMath.max(x) + cut * bw;
		if (!Double.isFinite(from) || !Double.isFinite(to) || from >= to)
			throw new IllegalArgumentException("Density bounds must be finite and increasing");
		double lo = from - 4 * bw, up = to + 4 * bw;
		if (!Double.isFinite(up - lo)) throw new IllegalArgumentException("Density grid span overflows");
		double[] y = bindist(x, weights, lo, up, n, 1);
		// Match the data bin width, as in R >= 4.4 with old.coords=FALSE.
		double[] kords = Utilities.seq_int(0, ((2. * n - 1) / (n - 1)) * (up-lo), 2 * n);
		int two_n = 2*n;
		for (int i = n+1; i < two_n; i++)
			kords[i] = -kords[two_n-i];
		kords = kernel.process(bw, kords);

		double[] new_kords = new double[kords.length * 2];
		System.arraycopy(kords, 0, new_kords, 0, kords.length);
		DoubleFFT_1D fft = new DoubleFFT_1D(kords.length);
		fft.realForwardFull(new_kords);
		for (int i = 1; i < new_kords.length; i += 2)
			new_kords[i] = -new_kords[i];
		kords = new_kords;
		new_kords = null;
		double[] new_y = new double[y.length * 2];
		System.arraycopy(y, 0, new_y, 0, y.length);
		fft.realForwardFull(new_y);
		y = new_y;
		for (int i = 0; i < kords.length; i += 2) {
			double
				a = kords[i],
				b = kords[i+1],
				c = y[i],
				d = y[i+1];
			kords[i] = a*c - b*d;
			kords[i+1] = a*d + b*c;
		}
		fft.complexInverse(kords, false);
		new_kords = new double[n];
		for (int i = 0; i < n; i++)
			new_kords[i] = max(0, kords[2*i] / two_n);
		kords = new_kords;
		new_kords = null;
		double[] xords = Utilities.seq_int(lo, up, n);
		x = Utilities.seq_int(from, to, n_user);
		ApproximationFunction fun = new ApproximationFunction(ApproximationType.LINEAR, xords, kords, Double.NaN, Double.NaN, 0);
		y = new double[n_user];
		for (int i = 0; i < n_user; i++)
			y[i] = fun.eval(x[i]);
		Density dd = new Density();
		dd.x = x;
		dd.y = y;
		return dd;
	}

	public static final Density density(double[] x) {
		return density(x, Bandwidth.NRD0, 1, Kernel.GAUSSIAN, null, 0, 512, Double.NaN, Double.NaN, 3);
	}

	public static final Density density(double[] x, double adjust) {
		return density(x, Bandwidth.NRD0, adjust, Kernel.GAUSSIAN, null, 0, 512, Double.NaN, Double.NaN, 3);
	}

	private static final double[] bindist(double[] x, double[] w, double xlo, double xhi, int n, double totMass) {
		int ylen = 2*n;
		double[] y = new double[ylen];
		int ixmin = 0, ixmax = n - 2, xlen = x.length;
		double xdelta = (xhi - xlo) / (n - 1);

		for(int i = 0; i < ylen; i++) y[i] = 0;
		for(int i = 0; i < xlen; i++) {
			if(MathFunctions.isFinite(x[i])) {
				double xpos = (x[i] - xlo) / xdelta;
				int ix = (int) floor(xpos);
				double fx = xpos - ix;
				double wi = w[i];
				if(ixmin <= ix && ix <= ixmax) {
					y[ix] += (1 - fx) * wi;
					y[ix + 1] += fx * wi;
				}
				else if(ix == -1) y[0] += fx * wi;
				else if(ix == ixmax + 1) y[ix] += (1 - fx) * wi;
			}
		}
		for(int i = 0; i < ylen; i++) y[i] *= totMass;
		return y;
	}

	public static final void main(String[] args) {
//        double[] input = new double[]{ 0.0176, -0.0620, 0.2467, 0.4599, -0.0582, 0.4694, 0.0001, -0.2873};
//        DoubleFFT_1D fftDo = new DoubleFFT_1D(input.length);
//        double[] fft = new double[input.length * 2];
//        System.arraycopy(input, 0, fft, 0, input.length);
//        fftDo.realForwardFull(fft);
// 
//        for(double d: fft) {
//            System.out.println(d);
//        }

		double[] v = new double[] {67.0, 54.7, 7.0, 48.5, 14.0, 17.2, 20.7, 13.0, 43.4, 40.2, 38.9, 54.5, 59.8, 48.3, 22.9,
			11.5, 34.4, 35.1, 38.7, 30.8, 30.6, 43.1, 56.8, 40.8, 41.8, 42.5, 31.0, 31.7, 30.2, 25.9,
			49.2, 37.0, 35.9, 15.0, 30.2,  7.2, 36.2, 45.5,  7.8, 33.4, 36.1, 40.2, 42.7, 42.5, 16.2,
			39.0, 35.0, 37.0, 31.4, 37.6, 39.9, 36.2, 42.8, 46.4, 24.7, 49.1, 46.0, 35.9,  7.8, 48.2,
			15.2, 32.5, 44.7, 42.6, 38.8, 17.4, 40.8, 29.1, 14.6, 59.2};
		density(v, Bandwidth.NRD0, 1, Kernel.GAUSSIAN, null, 0, 512, Double.NaN, Double.NaN, 3);
	}
}
