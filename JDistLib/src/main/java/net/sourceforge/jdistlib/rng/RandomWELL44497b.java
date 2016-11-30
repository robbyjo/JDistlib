/*
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
package net.sourceforge.jdistlib.rng;

/**
 * <P>Implementation of WELL 44497b (Well Equidistributed Long-period Linear) random number generator
 * by Francois Panneton, et al. The period is approximately 2^44497.
 * 
 * <P><a href="http://www.iro.umontreal.ca/~panneton/WELLRNG.html">Webpage</a>
 * <P>Paper: Panneton, F. and L'Ecuyer, P. Improved (2006) Long-Period Generators Based on Linear Recurrences Modulo 2,
 * ACM Transaction on Mathematical Software 32:1, pp 1--16(<a href="http://dl.acm.org/citation.cfm?id=1132974">link</a>)
 * 
 */
public class RandomWELL44497b extends RandomEngine
{
	private static final long serialVersionUID = 1L;
	private static final int W = 32, R = 1391, P = 15,
		MASKU = (0xffffffff>>>(W-P)), MASKL = ~MASKU, // M1 = 23, M2 = 481, M3 = 229,
		TEMPERB = 0x93dd1400, TEMPERC = 0xfa118000;
	//private static final double FACT = 2.32830643653869628906e-10;

	private boolean mHaveNextGaussian = false;
	private double mNextGaussian;
	private int state_i = 0, WELLRNG44497a = 1, STATE[] = new int[R];

	public RandomWELL44497b() {
		setSeed(System.currentTimeMillis());
	}

	public RandomWELL44497b(int[] init) {
		System.arraycopy(init, 0, STATE, 0, R);
	}

	public void setSeed(long seed) {
		mSeed = seed;
		STATE[0]= (int)(seed & 0xffffffff);
		for (int mti=1; mti<R; mti++) {
			STATE[mti] = (1812433253 * (STATE[mti-1] ^ (STATE[mti-1] >>> 30)) + mti); 
		}

	}
	@Override
	public final double nextGaussian() {
		if (mHaveNextGaussian) {
			mHaveNextGaussian = false;
			return mNextGaussian;
		}
		double v1, v2, s;

		do {
			v1 = nextDouble();
			v2 = nextDouble();
			s = v1 * v1 + v2 * v2;
		} while (s >= 1 || s==0);
		double multiplier = Math.sqrt(-2 * Math.log(s)/s);
		mHaveNextGaussian = true;
		mNextGaussian = v2 * multiplier;
		return v1 * multiplier;
	}

	// Shamelessly taken from Colt
	public double nextDouble() {
		return ((((long)(nextInt() >>> 6)) << 27) + (nextInt() >>> 5)) / (double)(1L << 53);
	}

	public float nextFloat() {
		return (float) nextDouble();
	}

	public RandomWELL44497b clone() {
		return new RandomWELL44497b();
	}

	@Override
	public int nextInt() {
		int z0, z1, z2, y;
		switch (WELLRNG44497a) {
		case 1:
			z0 = (STATE[state_i+R-1] & MASKL) | (STATE[state_i+R-2] & MASKU);
			z1 = (STATE[state_i]^(STATE[state_i]<<(-(-24)))) ^ (STATE[state_i+23]^(STATE[state_i+23]>>30));
			z2 = (STATE[state_i+481]^(STATE[state_i+481]<<(-(-10)))) ^ (STATE[state_i+229]<<(-(-26)));
			STATE[state_i] = z1 ^ z2;
			STATE[state_i-1+R] = z0 ^ (z1^(z1>>20)) ^ ((z2 & 0x00020000) != 0?((((z2<<9)^(z2>>(W-9)))&0xfbffffff)^0xb729fcec):(((z2<<9)^(z2>>(W-9)))&0xfbffffff)) ^ STATE[state_i];
			state_i = R-1;
			WELLRNG44497a = 3;
			y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
			y = y ^ (( y << 15) & TEMPERC);
			return y;
		case 2:
			z0 = (STATE[state_i-1] & MASKL) | (STATE[state_i+R-2] & MASKU);
			z1 = (STATE[state_i]^(STATE[state_i]<<(-(-24)))) ^ (STATE[state_i+23]^(STATE[state_i+23]>>30));
			z2 = (STATE[state_i+481]^(STATE[state_i+481]<<(-(-10)))) ^ (STATE[state_i+229]<<(-(-26)));
			STATE[state_i] = z1 ^ z2;
			STATE[state_i-1] = z0 ^ (z1^(z1>>20)) ^ ((z2 & 0x00020000) != 0?((((z2<<9)^(z2>>(W-9)))&0xfbffffff)^0xb729fcec):(((z2<<9)^(z2>>(W-9)))&0xfbffffff)) ^ STATE[state_i];
			state_i=0;
			WELLRNG44497a = 1;
			y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
			y = y ^ (( y << 15) & TEMPERC);
			return y;
		case 3:
			z0 = (STATE[state_i-1] & MASKL) | (STATE[state_i-2] & MASKU);
			z1 = (STATE[state_i]^(STATE[state_i]<<(-(-24)))) ^ (STATE[state_i+23 -R]^(STATE[state_i+23 -R]>>30));
			z2 = (STATE[state_i+481 -R]^(STATE[state_i+481 -R]<<(-(-10)))) ^ (STATE[state_i+229 -R]<<(-(-26)));
			STATE[state_i] = z1 ^ z2;
			STATE[state_i-1] = z0 ^ (z1^(z1>>20)) ^ ((z2 & 0x00020000) != 0?((((z2<<9)^(z2>>(W-9)))&0xfbffffff)^0xb729fcec):(((z2<<9)^(z2>>(W-9)))&0xfbffffff)) ^ STATE[state_i];
			state_i--;
			if(state_i+23<R)
				WELLRNG44497a = 4;
			y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
			y = y ^ (( y << 15) & TEMPERC);
			return y;
		case 4:
			z0 = (STATE[state_i-1] & MASKL) | (STATE[state_i-2] & MASKU);
			z1 = (STATE[state_i]^(STATE[state_i]<<(-(-24)))) ^ (STATE[state_i+23]^(STATE[state_i+23]>>30));
			z2 = (STATE[state_i+481 -R]^(STATE[state_i+481 -R]<<(-(-10)))) ^ (STATE[state_i+229 -R]<<(-(-26)));
			STATE[state_i] = z1 ^ z2;
			STATE[state_i-1] = z0 ^ (z1^(z1>>20)) ^ ((z2 & 0x00020000) != 0?((((z2<<9)^(z2>>(W-9)))&0xfbffffff)^0xb729fcec):(((z2<<9)^(z2>>(W-9)))&0xfbffffff)) ^ STATE[state_i];
			state_i--;
			if (state_i+229 < R)
				WELLRNG44497a = 5;
			y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
			y = y ^ (( y << 15) & TEMPERC);
			return y;
		case 5:
			z0 = (STATE[state_i-1] & MASKL) | (STATE[state_i-2] & MASKU);
			z1 = (STATE[state_i]^(STATE[state_i]<<(-(-24)))) ^ (STATE[state_i+23]^(STATE[state_i+23]>>30));
			z2 = (STATE[state_i+481 -R]^(STATE[state_i+481 -R]<<(-(-10)))) ^ (STATE[state_i+229]<<(-(-26)));
			STATE[state_i] = z1 ^ z2;
			STATE[state_i-1] = z0 ^ (z1^(z1>>20)) ^ ((z2 & 0x00020000) != 0?((((z2<<9)^(z2>>(W-9)))&0xfbffffff)^0xb729fcec):(((z2<<9)^(z2>>(W-9)))&0xfbffffff)) ^ STATE[state_i];
			state_i--;
			if(state_i+481 < R)
				WELLRNG44497a = 6;
			y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
			y = y ^ (( y << 15) & TEMPERC);
			return y;
		case 6:
			z0 = (STATE[state_i-1] & MASKL) | (STATE[state_i-2] & MASKU);
			z1 = (STATE[state_i]^(STATE[state_i]<<(-(-24)))) ^ (STATE[state_i+23]^(STATE[state_i+23]>>30));
			z2 = (STATE[state_i+481]^(STATE[state_i+481]<<(-(-10)))) ^ (STATE[state_i+229]<<(-(-26)));
			STATE[state_i] = z1 ^ z2;
			STATE[state_i-1] = z0 ^ (z1^(z1>>20)) ^ ((z2 & 0x00020000) != 0?((((z2<<9)^(z2>>(W-9)))&0xfbffffff)^0xb729fcec):(((z2<<9)^(z2>>(W-9)))&0xfbffffff)) ^ STATE[state_i];
			state_i--;
			if(state_i == 1 )
				WELLRNG44497a = 2;
			y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
			y = y ^ (( y << 15) & TEMPERC);
			return y;
		}
		throw new RuntimeException();
	}

	@Override
	public int nextInt(int n) {
		return nextInt() * n;
	}

	@Override
	public long nextLong() {
		return (((long) nextInt()) << 32) + (long) nextInt();
	}

	@Override
	public long nextLong(long l) {
		return nextLong() * l;
	}
}
