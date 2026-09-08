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
package jdistlib.rng;

/**
 * <P>Implementation of CMWC4096 (Complementary-multiply-with-carry) random number generator
 * by George Marsaglia. The period is approximately 2^131086.
 * 
 * <P><a href="http://groups.google.com/group/comp.lang.c/browse_thread/thread/a9915080a4424068/">Original post</a>
 * <P>Paper: Marsaglia, George. 2003. Random number generators. Journal of Modern Applied
 * Statistical Methods, 2(1): 2-13. (<a href="http://tbf.coe.wayne.edu/jmasm/vol2_no1.pdf">link</a>)
 * 
 * <P>The 64-bit seed is expanded into the full 4096-word state using SplitMix64.
 * The initial carry is 362436. Corrected unsigned arithmetic changes streams
 * from older releases; serialized states from the broken implementation are rejected.
 * 
 * @author Roby Joehanes
 *
 */
public class RandomCMWC extends RandomEngine {
	private static final long serialVersionUID = 2L;
	private static final long MASK = 0xffffffffL;
	private long[] mBuffer = new long[4096];
	private int mIndex;
	private long carry;
	private boolean mHaveNextGaussian;
	private double mNextGaussian;

	public RandomCMWC() { this(362436L); }
	public RandomCMWC(long seed) { setSeed(seed); }
	private RandomCMWC(boolean copyingState) { /* state is supplied by clone() */ }
	@Override public void setSeed(long seed) {
		mSeed = seed;
		if (mBuffer == null) return; // superclass construction
		// SplitMix64 expands all seed bits into nondegenerate 32-bit words.
		long state = seed;
		for (int i = 0; i < mBuffer.length; i++) {
			state += 0x9e3779b97f4a7c15L;
			long value = state;
			value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
			value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
			mBuffer[i] = (value ^ (value >>> 31)) & MASK;
		}
		carry = 362436; mIndex = 4095; mHaveNextGaussian = false;
	}
	@Override public int nextInt() {
		mIndex = (mIndex + 1) & 4095;
		long product = 18782L * mBuffer[mIndex] + carry;
		carry = product >>> 32;
		long value = (product + carry) & MASK;
		if (value < carry) { value = (value + 1) & MASK; carry++; }
		mBuffer[mIndex] = (0xfffffffeL - value) & MASK;
		return (int) mBuffer[mIndex];
	}
	@Override public long nextLong() { return ((long) nextInt() << 32) | (nextInt() & MASK); }
	@Override public double nextDouble() { return (((long) (nextInt() >>> 6) << 27) + (nextInt() >>> 5)) * 0x1.0p-53; }
	@Override public float nextFloat() { return (nextInt() >>> 8) * 0x1.0p-24f; }
	@Override public double nextGaussian() {
		if (mHaveNextGaussian) { mHaveNextGaussian = false; return mNextGaussian; }
		double x, y, radius;
		do { x = 2 * nextDouble() - 1; y = 2 * nextDouble() - 1; radius = x*x + y*y; } while (radius >= 1 || radius == 0);
		double multiplier = Math.sqrt(-2 * Math.log(radius) / radius);
		mNextGaussian = y * multiplier; mHaveNextGaussian = true; return x * multiplier;
	}
	@Override public RandomCMWC clone() {
		RandomCMWC copy = new RandomCMWC(true);
		copy.mSeed = mSeed;
		copy.mBuffer = mBuffer.clone(); copy.mIndex = mIndex; copy.carry = carry;
		copy.mHaveNextGaussian = mHaveNextGaussian; copy.mNextGaussian = mNextGaussian;
		return copy;
	}
}
