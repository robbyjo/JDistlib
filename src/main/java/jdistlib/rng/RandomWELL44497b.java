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
package jdistlib.rng;

/**
 * <P>Implementation of WELL 44497b (Well Equidistributed Long-period Linear) random number generator
 * by Francois Panneton, et al. The period is approximately 2^44497.
 * 
 * <P><a href="http://www.iro.umontreal.ca/~panneton/WELLRNG.html">Webpage</a>
 * <P>Paper: Panneton, F. and L'Ecuyer, P. Improved (2006) Long-Period Generators Based on Linear Recurrences Modulo 2,
 * ACM Transaction on Mathematical Software 32:1, pp 1--16(<a href="http://dl.acm.org/citation.cfm?id=1132974">link</a>)
 * 
 */
public class RandomWELL44497b extends RandomEngine {
	// The corrected recurrence cannot reproduce serialized pre-audit streams.
	private static final long serialVersionUID = 2L;
	private static final int R = 1391;
	private int[] STATE = new int[R];
	private int state_i;
	private boolean mHaveNextGaussian;
	private double mNextGaussian;

	public RandomWELL44497b() { this(System.currentTimeMillis()); }
	public RandomWELL44497b(long seed) { setSeed(seed); }
	public RandomWELL44497b(int[] init) {
		if (init == null || init.length != R) throw new IllegalArgumentException("WELL44497 requires 1391 state words");
		int nonzero = 0; for (int value : init) nonzero |= value;
		if (nonzero == 0) throw new IllegalArgumentException("WELL state must not be all zero");
		System.arraycopy(init, 0, STATE, 0, R);
	}
	@Override public void setSeed(long seed) {
		mSeed = seed;
		// java.util.Random calls the override before subclass fields exist.
		if (STATE == null) return;
		STATE[0] = (int) seed;
		for (int i = 1; i < R; i++) STATE[i] = 1812433253 * (STATE[i-1] ^ (STATE[i-1] >>> 30)) + i;
		state_i = 0; mHaveNextGaussian = false;
	}
	@Override public int nextInt() {
		int previous = state_i == 0 ? R - 1 : state_i - 1;
		int previous2 = previous == 0 ? R - 1 : previous - 1;
		int v0 = STATE[state_i], v1 = STATE[(state_i + 23) % R];
		int v2 = STATE[(state_i + 481) % R], v3 = STATE[(state_i + 229) % R];
		int z0 = (STATE[previous] & 0xffff8000) | (STATE[previous2] & 0x00007fff);
		int z1 = (v0 ^ (v0 << 24)) ^ (v1 ^ (v1 >>> 30));
		int z2 = (v2 ^ (v2 << 10)) ^ (v3 << 26);
		int rotated = ((z2 << 9) ^ (z2 >>> 23)) & 0xfbffffff;
		if ((z2 & 0x00020000) != 0) rotated ^= 0xb729fcec;
		STATE[state_i] = z1 ^ z2;
		STATE[previous] = z0 ^ z1 ^ (z1 >>> 20) ^ rotated ^ STATE[state_i];
		STATE[previous2] &= 0xffff8000;
		state_i = previous;
		int y = STATE[state_i];
		y ^= (y << 7) & 0x93dd1400;
		return y ^ ((y << 15) & 0xfa118000);
	}
	@Override public long nextLong() { return ((long) nextInt() << 32) | (nextInt() & 0xffffffffL); }
	@Override public double nextDouble() { return (((long) (nextInt() >>> 6) << 27) + (nextInt() >>> 5)) * 0x1.0p-53; }
	@Override public float nextFloat() { return (nextInt() >>> 8) * 0x1.0p-24f; }
	@Override public double nextGaussian() {
		if (mHaveNextGaussian) { mHaveNextGaussian = false; return mNextGaussian; }
		double x, y, radius;
		do { x = 2 * nextDouble() - 1; y = 2 * nextDouble() - 1; radius = x*x + y*y; } while (radius >= 1 || radius == 0);
		double multiplier = Math.sqrt(-2 * Math.log(radius) / radius);
		mNextGaussian = y * multiplier; mHaveNextGaussian = true; return x * multiplier;
	}
	@Override public RandomWELL44497b clone() {
		RandomWELL44497b copy = new RandomWELL44497b(STATE);
		copy.state_i = state_i; copy.mSeed = mSeed;
		copy.mHaveNextGaussian = mHaveNextGaussian; copy.mNextGaussian = mNextGaussian;
		return copy;
	}
}
