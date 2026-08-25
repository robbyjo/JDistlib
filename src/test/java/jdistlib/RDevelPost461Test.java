package jdistlib;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import jdistlib.Binomial.BinomialKind;
import jdistlib.disttest.DistributionTest;
import jdistlib.disttest.TestKind;
import jdistlib.rng.RandomEngine;

/** Regression cases from R-devel revisions 89909, 90068, 90223, and 90299. */
public class RDevelPost461Test {

	@Test
	public void correctedBtpeIsDefaultAndLegacyStreamsRemainAvailable() {
		assertEquals(BinomialKind.BTPE, Binomial.create_random_state().getKind());
		assertEquals(BinomialKind.BUGGY_BTPE,
				Binomial.create_random_state(BinomialKind.BUGGY_BTPE).getKind());

		double[] corrected = Binomial.random(25, 320, .25,
				new RCompatibleMersenneTwister(428),
				Binomial.create_random_state(BinomialKind.BTPE));
		double[] legacy = Binomial.random(25, 320, .25,
				new RCompatibleMersenneTwister(428),
				Binomial.create_random_state(BinomialKind.BUGGY_BTPE));

		assertArrayEquals(Arrays.copyOf(corrected, 24), Arrays.copyOf(legacy, 24), 0.0);
		assertEquals(96.0, corrected[24], 0.0);
		assertEquals(103.0, legacy[24], 0.0);
		Arrays.sort(corrected);
		assertArrayEquals(new double[] {
			72, 75, 75, 76, 77, 77, 77, 78, 80, 81, 81, 81, 82,
			83, 83, 83, 84, 84, 87, 87, 93, 94, 94, 96, 97
		}, corrected, 0.0);
	}

	@Test
	public void largeHypergeometricPathUsesTheCombinedPopulationBoundary() {
		double t30 = 1_073_741_824.0;
		RCompatibleMersenneTwister random = new RCompatibleMersenneTwister(19);
		double[] draws = HyperGeometric.random(3, t30, t30, 18, random);
		assertArrayEquals(new double[] {12, 9, 8}, draws, 0.0);
		assertTrue(Double.isNaN(HyperGeometric.random(Double.NaN, 2, 1, random)));
	}

	@Test
	public void multinomialUsesSequentialBinomialsWithKahanArithmetic() {
		double[] probabilities = {.5, .1, .1, .1, .1, .1};
		int[][] draws = Multinomial.random(3, 4, probabilities,
				new RCompatibleMersenneTwister(1));
		assertArrayEquals(new int[] {1, 0, 1, 2, 0, 0}, draws[0]);
		assertArrayEquals(new int[] {1, 2, 1, 0, 0, 0}, draws[1]);
		assertArrayEquals(new int[] {2, 0, 0, 0, 0, 2}, draws[2]);

		int[] compensated = Multinomial.random(7,
				new double[] {1e16, 1, 1}, new RCompatibleMersenneTwister(1));
		assertEquals(7, compensated[0] + compensated[1] + compensated[2]);
	}

	@Test
	public void wilcoxonDefaultsRoundRanksAndZapAlmostZeroDifferences() {
		double[] x = {1.1, 2, 1.15, 1e-100};
		double[] defaults = DistributionTest.wilcoxon_test(x, 0, true,
				TestKind.TWO_SIDED);
		double[] unrounded = DistributionTest.wilcoxon_test(x, 0, true,
				TestKind.TWO_SIDED, Double.POSITIVE_INFINITY,
				Double.POSITIVE_INFINITY);
		double[] explicitZap = DistributionTest.wilcoxon_test(x, 0, true,
				TestKind.TWO_SIDED, Double.POSITIVE_INFINITY, 12);
		assertEquals(9.0, defaults[0], 0.0);
		assertEquals(10.0, unrounded[0], 0.0);
		assertEquals(.125, unrounded[1], 0.0);
		assertEquals(9.0, explicitZap[0], 0.0);

		double[] signed = DistributionTest.wilcoxon_test(
				new double[] {1, -2, -3}, 0, false, TestKind.TWO_SIDED);
		assertArrayEquals(new double[] {1, .5}, signed, 2e-16);

		double[] a = {1.00000001, 2};
		double[] b = {1.00000002, 3};
		double defaultRank = DistributionTest.mann_whitney_u_test(a, b, 0,
				false, false, TestKind.TWO_SIDED)[0];
		double fullPrecisionRank = DistributionTest.mann_whitney_u_test(a, b, 0,
				false, false, TestKind.TWO_SIDED, Double.POSITIVE_INFINITY,
				Double.POSITIVE_INFINITY)[0];
		assertEquals(1.5, defaultRank, 0.0);
		assertEquals(1.0, fullPrecisionRank, 0.0);
	}

	/** R's 32-bit Mersenne-Twister stream, used only for upstream vectors. */
	private static final class RCompatibleMersenneTwister extends RandomEngine {
		private static final long serialVersionUID = 1L;
		private static final int N = 624;
		private static final int M = 397;
		private static final int MATRIX_A = 0x9908b0df;
		private final int[] state;
		private int index;

		RCompatibleMersenneTwister(int initialSeed) {
			state = new int[N];
			int seed = initialSeed;
			for (int i = 0; i < 50; i++) seed = 69069 * seed + 1;
			seed = 69069 * seed + 1; // R stores and then replaces the MT index.
			for (int i = 0; i < N; i++) {
				seed = 69069 * seed + 1;
				state[i] = seed;
			}
			index = N;
		}

		private RCompatibleMersenneTwister(RCompatibleMersenneTwister source) {
			state = source.state.clone();
			index = source.index;
		}

		@Override
		public double nextDouble() {
			if (index >= N) twist();
			int value = state[index++];
			value ^= value >>> 11;
			value ^= (value << 7) & 0x9d2c5680;
			value ^= (value << 15) & 0xefc60000;
			value ^= value >>> 18;
			double result = Integer.toUnsignedLong(value) * 0x1.0p-32;
			if (result <= 0.0) return 0.5 / 0xffffffffL;
			return result;
		}

		private void twist() {
			for (int i = 0; i < N; i++) {
				int value = (state[i] & 0x80000000)
						| (state[(i + 1) % N] & 0x7fffffff);
				state[i] = state[(i + M) % N] ^ (value >>> 1)
						^ ((value & 1) == 0 ? 0 : MATRIX_A);
			}
			index = 0;
		}

		@Override public double nextGaussian() { throw new UnsupportedOperationException(); }
		@Override public float nextFloat() { return (float) nextDouble(); }
		@Override public int nextInt() { return (int) (nextDouble() * 0x1.0p32); }
		@Override public int nextInt(int n) { return (int) (nextDouble() * n); }
		@Override public long nextLong() { return ((long) nextInt() << 32) ^ Integer.toUnsignedLong(nextInt()); }
		@Override public long nextLong(long n) { return (long) (nextDouble() * n); }
		@Override public RandomEngine clone() { return new RCompatibleMersenneTwister(this); }
	}
}
