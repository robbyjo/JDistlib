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
 * @author Roby Joehanes
 *
 */
public abstract class RandomEngine extends java.util.Random
{
	private static final long serialVersionUID = 1L;
	protected long mSeed;

	public void setSeed(long seed)
	{	mSeed = seed; }

	public long getSeed()
	{	return mSeed; }

	public abstract double nextGaussian();
	public abstract double nextDouble();
	public abstract float nextFloat();
	public abstract int nextInt();
	/** Uniform bounded draw using rejection to avoid modulo bias. */
	public int nextInt(int n) {
		if (n <= 0) throw new IllegalArgumentException("bound must be positive");
		int bits = nextInt() >>> 1;
		if ((n & -n) == n) return (int) ((n * (long) bits) >> 31);
		int value = bits % n;
		while (bits - value + (n - 1) < 0) { bits = nextInt() >>> 1; value = bits % n; }
		return value;
	}
	public abstract long nextLong();
	public long nextLong(long n) {
		if (n <= 0) throw new IllegalArgumentException("bound must be positive");
		long bits, value;
		do { bits = nextLong() >>> 1; value = bits % n; } while (bits - value + (n - 1) < 0);
		return value;
	}

	/** Route inherited Random methods through this engine's state. */
	@Override protected int next(int bits) { return bits == 0 ? 0 : nextInt() >>> (32 - bits); }

	public double random()
	{	return nextDouble(); }

	public abstract RandomEngine clone();
}
