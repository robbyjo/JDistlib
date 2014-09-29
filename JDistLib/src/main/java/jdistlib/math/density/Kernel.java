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

import jdistlib.Normal;
import static java.lang.Math.*;

public enum Kernel {
	GAUSSIAN,
	RECTANGULAR,
	TRIANGULAR,
	EPANECHNIKOV,
	BIWEIGHT,
	COSINE,
	OPTCOSINE;

	public double getKernelValue() {
		double val = Double.NaN;
		switch(this) {
			case GAUSSIAN: val = 0.28209479177387814347403972578038629292202531466449942842204; break; // 1 / (2 *sqrt(pi))
			case RECTANGULAR: val = 0.28867513459481288225457439025097872782380087563506343800930; break; // sqrt(3)/6
			case TRIANGULAR: val = 0.27216552697590867757747600830065459910732749785074112538141; break; // sqrt(6)/9
			case EPANECHNIKOV: val = 0.26832815729997476356910084024775314825287420315338308691251; break; // 3/(5*sqrt(5))
			case BIWEIGHT: val = 0.26997462357801944801036895445298575772553665133494389595595; break; // 5*sqrt(7)/49
			case COSINE: val = 0.27113404139349600880768139132607390883621162441806725187945; break; // 3/4*sqrt(1/3 - 2/pi^2)
			case OPTCOSINE: val = 0.26847555627566835086519788444052383420282057036984861080853; break; // sqrt(1-8/pi^2)*pi^2/16
		}
		return val;
	}

	public double getFactor() {
		double val = Double.NaN;
		switch(this) {
			case GAUSSIAN: val = 4; break;
			case RECTANGULAR: val = 3.46410161513775458705489268301174473388561050762076125611161; break; // 2*sqrt(3)
			case TRIANGULAR: val = 4.89897948556635619639456814941178278393189496131334025686539; break; // 2 * sqrt(6)
			case EPANECHNIKOV: val = 4.47213595499957939281834733746255247088123671922305144854179; break; // 2 * sqrt(5)
			case BIWEIGHT: val = 5.29150262212918118100323150727852085142051836616490036073667; break; // 2 * sqrt(7)
			case COSINE: val = 5.53231896773542561830175406819223588624710552821404457600186; break; // 2/sqrt(1/3 - 2/pi^2)
			case OPTCOSINE: val = 4.59520623497439335845325655100236130574085692247056662314846; break; // 2/sqrt(1-8/pi^2)
		}
		return val;
	}

	public double[] process(double bw, double[] kords) {
		int n = kords.length;
		double result[] = new double[n], a;
		switch (this) {
			case GAUSSIAN:
				for (int i = 0; i < n; i++)
					result[i] = Normal.density(kords[i], 0, bw, false);
				break;
			case RECTANGULAR:
				a = bw*sqrt(3);
				for (int i = 0; i < n; i++)
					result[i] = abs(kords[i]) < a ? 0.5/a : 0;
				break;
			case TRIANGULAR:
				a = bw*sqrt(6);
				for (int i = 0; i < n; i++) {
					double ax = abs(kords[i]);
					result[i] = ax < a ? (1 - ax/a)/a : 0;
				}
				break;
			case EPANECHNIKOV:
				a = bw*sqrt(5);
				for (int i = 0; i < n; i++) {
					double ax = abs(kords[i]);
					if (ax < a) {
						double axa = ax/a;
						result[i] = 3/4*(1 - axa * axa)/a;
					} else result[i] = 0;
				}
				break;
			case BIWEIGHT:
				a = bw*sqrt(7);
				for (int i = 0; i < n; i++) {
					double ax = abs(kords[i]);
					if (ax < a) {
						double axa = ax/a;
						axa = (1 - axa * axa);
						result[i] = 15/16*axa*axa/a;
					} else result[i] = 0;
				}
				break;
			case COSINE:
				a = bw/sqrt(1.0/3 - 2.0/(PI * PI));
				for (int i = 0; i < n; i++)
					result[i] = abs(kords[i]) < a ? (1 + cos(PI * kords[i]/a))/(2*a) : 0;
				break;
			case OPTCOSINE:
				a = bw/sqrt(1.0 - 8.0/(PI * PI));
				for (int i = 0; i < n; i++)
					result[i] = abs(kords[i]) < a ? PI/4.0 * cos(PI * kords[i]/(2*a))/a : 0;
				break;
		}
		return result;
	}
}
