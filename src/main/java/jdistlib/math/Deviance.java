/*
 * Copyright (C) 2000-2025 The R Core Team
 * Copyright (C) 2026 Roby Joehanes
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
package jdistlib.math;

import static java.lang.Math.*;
import static jdistlib.math.Constants.M_LN2;

/**
 * Extended-precision deviance calculation ported from R's {@code ebd0}.
 */
final class Deviance {
	private static final int SCALE_BITS = 10;
	private static final int SCALE_SIZE = 128;

	private static final float[][] SCALE = {
	{ +0x1.62e430p-1f, -0x1.05c610p-29f, -0x1.950d88p-54f, +0x1.d9cc02p-79f },
	{ +0x1.5ee02cp-1f, -0x1.6dbe98p-25f, -0x1.51e540p-50f, +0x1.2bfa48p-74f },
	{ +0x1.5ad404p-1f, +0x1.86b3e4p-26f, +0x1.9f6534p-50f, +0x1.54be04p-74f },
	{ +0x1.570124p-1f, -0x1.9ed750p-25f, -0x1.f37dd0p-51f, +0x1.10b770p-77f },
	{ +0x1.5326e4p-1f, -0x1.9b9874p-25f, -0x1.378194p-49f, +0x1.56feb2p-74f },
	{ +0x1.4f4528p-1f, +0x1.aca70cp-28f, +0x1.103e74p-53f, +0x1.9c410ap-81f },
	{ +0x1.4b5bd8p-1f, -0x1.6a91d8p-25f, -0x1.8e43d0p-50f, -0x1.afba9ep-77f },
	{ +0x1.47ae54p-1f, -0x1.abb51cp-25f, +0x1.19b798p-51f, +0x1.45e09cp-76f },
	{ +0x1.43fa00p-1f, -0x1.d06318p-25f, -0x1.8858d8p-49f, -0x1.1927c4p-75f },
	{ +0x1.3ffa40p-1f, +0x1.1a427cp-25f, +0x1.151640p-53f, -0x1.4f5606p-77f },
	{ +0x1.3c7c80p-1f, -0x1.19bf48p-34f, +0x1.05fc94p-58f, -0x1.c096fcp-82f },
	{ +0x1.38b320p-1f, +0x1.6b5778p-25f, +0x1.be38d0p-50f, -0x1.075e96p-74f },
	{ +0x1.34e288p-1f, +0x1.d9ce1cp-25f, +0x1.316eb8p-49f, +0x1.2d885cp-73f },
	{ +0x1.315124p-1f, +0x1.c2fc60p-29f, -0x1.4396fcp-53f, +0x1.acf376p-78f },
	{ +0x1.2db954p-1f, +0x1.720de4p-25f, -0x1.d39b04p-49f, -0x1.f11176p-76f },
	{ +0x1.2a1b08p-1f, -0x1.562494p-25f, +0x1.a7863cp-49f, +0x1.85dd64p-73f },
	{ +0x1.267620p-1f, +0x1.3430e0p-29f, -0x1.96a958p-56f, +0x1.f8e636p-82f },
	{ +0x1.23130cp-1f, +0x1.7bebf4p-25f, +0x1.416f1cp-52f, -0x1.78dd36p-77f },
	{ +0x1.1faa34p-1f, +0x1.70e128p-26f, +0x1.81817cp-50f, -0x1.c2179cp-76f },
	{ +0x1.1bf204p-1f, +0x1.3a9620p-28f, +0x1.2f94c0p-52f, +0x1.9096c0p-76f },
	{ +0x1.187ce4p-1f, -0x1.077870p-27f, +0x1.655a80p-51f, +0x1.eaafd6p-78f },
	{ +0x1.1501c0p-1f, -0x1.406cacp-25f, -0x1.e72290p-49f, +0x1.5dd800p-73f },
	{ +0x1.11cb80p-1f, +0x1.787cd0p-25f, -0x1.efdc78p-51f, -0x1.5380cep-77f },
	{ +0x1.0e4498p-1f, +0x1.747324p-27f, -0x1.024548p-51f, +0x1.77a5a6p-75f },
	{ +0x1.0b036cp-1f, +0x1.690c74p-25f, +0x1.5d0cc4p-50f, -0x1.c0e23cp-76f },
	{ +0x1.077070p-1f, -0x1.a769bcp-27f, +0x1.452234p-52f, +0x1.6ba668p-76f },
	{ +0x1.04240cp-1f, -0x1.a686acp-27f, -0x1.ef46b0p-52f, -0x1.5ce10cp-76f },
	{ +0x1.00d22cp-1f, +0x1.fc0e10p-25f, +0x1.6ee034p-50f, -0x1.19a2ccp-74f },
	{ +0x1.faf588p-2f, +0x1.ef1e64p-27f, -0x1.26504cp-54f, -0x1.b15792p-82f },
	{ +0x1.f4d87cp-2f, +0x1.d7b980p-26f, -0x1.a114d8p-50f, +0x1.9758c6p-75f },
	{ +0x1.ee1414p-2f, +0x1.2ec060p-26f, +0x1.dc00fcp-52f, +0x1.f8833cp-76f },
	{ +0x1.e7e32cp-2f, -0x1.ac796cp-27f, -0x1.a68818p-54f, +0x1.235d02p-78f },
	{ +0x1.e108a0p-2f, -0x1.768ba4p-28f, -0x1.f050a8p-52f, +0x1.00d632p-82f },
	{ +0x1.dac354p-2f, -0x1.d3a6acp-30f, +0x1.18734cp-57f, -0x1.f97902p-83f },
	{ +0x1.d47424p-2f, +0x1.7dbbacp-31f, -0x1.d5ada4p-56f, +0x1.56fcaap-81f },
	{ +0x1.ce1af0p-2f, +0x1.70be7cp-27f, +0x1.6f6fa4p-51f, +0x1.7955a2p-75f },
	{ +0x1.c7b798p-2f, +0x1.ec36ecp-26f, -0x1.07e294p-50f, -0x1.ca183cp-75f },
	{ +0x1.c1ef04p-2f, +0x1.c1dfd4p-26f, +0x1.888eecp-50f, -0x1.fd6b86p-75f },
	{ +0x1.bb7810p-2f, +0x1.478bfcp-26f, +0x1.245b8cp-50f, +0x1.ea9d52p-74f },
	{ +0x1.b59da0p-2f, -0x1.882b08p-27f, +0x1.31573cp-53f, -0x1.8c249ap-77f },
	{ +0x1.af1294p-2f, -0x1.b710f4p-27f, +0x1.622670p-51f, +0x1.128578p-76f },
	{ +0x1.a925d4p-2f, -0x1.0ae750p-27f, +0x1.574ed4p-51f, +0x1.084996p-75f },
	{ +0x1.a33040p-2f, +0x1.027d30p-29f, +0x1.b9a550p-53f, -0x1.b2e38ap-78f },
	{ +0x1.9d31c0p-2f, -0x1.5ec12cp-26f, -0x1.5245e0p-52f, +0x1.2522d0p-79f },
	{ +0x1.972a34p-2f, +0x1.135158p-30f, +0x1.a5c09cp-56f, +0x1.24b70ep-80f },
	{ +0x1.911984p-2f, +0x1.0995d4p-26f, +0x1.3bfb5cp-50f, +0x1.2c9dd6p-75f },
	{ +0x1.8bad98p-2f, -0x1.1d6144p-29f, +0x1.5b9208p-53f, +0x1.1ec158p-77f },
	{ +0x1.858b58p-2f, -0x1.1b4678p-27f, +0x1.56cab4p-53f, -0x1.2fdc0cp-78f },
	{ +0x1.7f5fa0p-2f, +0x1.3aaf48p-27f, +0x1.461964p-51f, +0x1.4ae476p-75f },
	{ +0x1.79db68p-2f, -0x1.7e5054p-26f, +0x1.673750p-51f, -0x1.a11f7ap-76f },
	{ +0x1.744f88p-2f, -0x1.cc0e18p-26f, -0x1.1e9d18p-50f, -0x1.6c06bcp-78f },
	{ +0x1.6e08ecp-2f, -0x1.5d45e0p-26f, -0x1.c73ec8p-50f, +0x1.318d72p-74f },
	{ +0x1.686c80p-2f, +0x1.e9b14cp-26f, -0x1.13bbd4p-50f, -0x1.efeb1cp-78f },
	{ +0x1.62c830p-2f, -0x1.a8c70cp-27f, -0x1.5a1214p-51f, -0x1.bab3fcp-79f },
	{ +0x1.5d1bdcp-2f, -0x1.4fec6cp-31f, +0x1.423638p-56f, +0x1.ee3feep-83f },
	{ +0x1.576770p-2f, +0x1.7455a8p-26f, -0x1.3ab654p-50f, -0x1.26be4cp-75f },
	{ +0x1.5262e0p-2f, -0x1.146778p-26f, -0x1.b9f708p-52f, -0x1.294018p-77f },
	{ +0x1.4c9f08p-2f, +0x1.e152c4p-26f, -0x1.dde710p-53f, +0x1.fd2208p-77f },
	{ +0x1.46d2d8p-2f, +0x1.c28058p-26f, -0x1.936284p-50f, +0x1.9fdd68p-74f },
	{ +0x1.41b940p-2f, +0x1.cce0c0p-26f, -0x1.1a4050p-50f, +0x1.bc0376p-76f },
	{ +0x1.3bdd24p-2f, +0x1.d6296cp-27f, +0x1.425b48p-51f, -0x1.cddb2cp-77f },
	{ +0x1.36b578p-2f, -0x1.287ddcp-27f, -0x1.2d0f4cp-51f, +0x1.38447ep-75f },
	{ +0x1.31871cp-2f, +0x1.2a8830p-27f, +0x1.3eae54p-52f, -0x1.898136p-77f },
	{ +0x1.2b9304p-2f, -0x1.51d8b8p-28f, +0x1.27694cp-52f, -0x1.fd852ap-76f },
	{ +0x1.265620p-2f, -0x1.d98f3cp-27f, +0x1.a44338p-51f, -0x1.56e85ep-78f },
	{ +0x1.211254p-2f, +0x1.986160p-26f, +0x1.73c5d0p-51f, +0x1.4a861ep-75f },
	{ +0x1.1bc794p-2f, +0x1.fa3918p-27f, +0x1.879c5cp-51f, +0x1.16107cp-78f },
	{ +0x1.1675ccp-2f, -0x1.4545a0p-26f, +0x1.c07398p-51f, +0x1.f55c42p-76f },
	{ +0x1.111ce4p-2f, +0x1.f72670p-37f, -0x1.b84b5cp-61f, +0x1.a4a4dcp-85f },
	{ +0x1.0c81d4p-2f, +0x1.0c150cp-27f, +0x1.218600p-51f, -0x1.d17312p-76f },
	{ +0x1.071b84p-2f, +0x1.fcd590p-26f, +0x1.a3a2e0p-51f, +0x1.fe5ef8p-76f },
	{ +0x1.01ade4p-2f, -0x1.bb1844p-28f, +0x1.db3cccp-52f, +0x1.1f56fcp-77f },
	{ +0x1.fa01c4p-3f, -0x1.12a0d0p-29f, -0x1.f71fb0p-54f, +0x1.e287a4p-78f },
	{ +0x1.ef0adcp-3f, +0x1.7b8b28p-28f, -0x1.35bce4p-52f, -0x1.abc8f8p-79f },
	{ +0x1.e598ecp-3f, +0x1.5a87e4p-27f, -0x1.134bd0p-51f, +0x1.c2cebep-76f },
	{ +0x1.da85d8p-3f, -0x1.df31b0p-27f, +0x1.94c16cp-57f, +0x1.8fd7eap-82f },
	{ +0x1.d0fb80p-3f, -0x1.bb5434p-28f, -0x1.ea5640p-52f, -0x1.8ceca4p-77f },
	{ +0x1.c765b8p-3f, +0x1.e4d68cp-27f, +0x1.5b59b4p-51f, +0x1.76f6c4p-76f },
	{ +0x1.bdc46cp-3f, -0x1.1cbb50p-27f, +0x1.2da010p-51f, +0x1.eb282cp-75f },
	{ +0x1.b27980p-3f, -0x1.1b9ce0p-27f, +0x1.7756f8p-52f, +0x1.2ff572p-76f },
	{ +0x1.a8bed0p-3f, -0x1.bbe874p-30f, +0x1.85cf20p-56f, +0x1.b9cf18p-80f },
	{ +0x1.9ef83cp-3f, +0x1.2769a4p-27f, -0x1.85bda0p-52f, +0x1.8c8018p-79f },
	{ +0x1.9525a8p-3f, +0x1.cf456cp-27f, -0x1.7137d8p-52f, -0x1.f158e8p-76f },
	{ +0x1.8b46f8p-3f, +0x1.11b12cp-30f, +0x1.9f2104p-54f, -0x1.22836ep-78f },
	{ +0x1.83040cp-3f, +0x1.2379e4p-28f, +0x1.b71c70p-52f, -0x1.990cdep-76f },
	{ +0x1.790ed4p-3f, +0x1.dc4c68p-28f, -0x1.910ac8p-52f, +0x1.dd1bd6p-76f },
	{ +0x1.6f0d28p-3f, +0x1.5cad68p-28f, +0x1.737c94p-52f, -0x1.9184bap-77f },
	{ +0x1.64fee8p-3f, +0x1.04bf88p-28f, +0x1.6fca28p-52f, +0x1.8884a8p-76f },
	{ +0x1.5c9400p-3f, +0x1.d65cb0p-29f, -0x1.b2919cp-53f, +0x1.b99bcep-77f },
	{ +0x1.526e60p-3f, -0x1.c5e4bcp-27f, -0x1.0ba380p-52f, +0x1.d6e3ccp-79f },
	{ +0x1.483bccp-3f, +0x1.9cdc7cp-28f, -0x1.5ad8dcp-54f, -0x1.392d3cp-83f },
	{ +0x1.3fb25cp-3f, -0x1.a6ad74p-27f, +0x1.5be6b4p-52f, -0x1.4e0114p-77f },
	{ +0x1.371fc4p-3f, -0x1.fe1708p-27f, -0x1.78864cp-52f, -0x1.27543ap-76f },
	{ +0x1.2cca10p-3f, -0x1.4141b4p-28f, -0x1.ef191cp-52f, +0x1.00ee08p-76f },
	{ +0x1.242310p-3f, +0x1.3ba510p-27f, -0x1.d003c8p-51f, +0x1.162640p-76f },
	{ +0x1.1b72acp-3f, +0x1.52f67cp-27f, -0x1.fd6fa0p-51f, +0x1.1a3966p-77f },
	{ +0x1.10f8e4p-3f, +0x1.129cd8p-30f, +0x1.31ef30p-55f, +0x1.a73e38p-79f },
	{ +0x1.08338cp-3f, -0x1.005d7cp-27f, -0x1.661a9cp-51f, +0x1.1f138ap-79f },
	{ +0x1.fec914p-4f, -0x1.c482a8p-29f, -0x1.55746cp-54f, +0x1.99f932p-80f },
	{ +0x1.ed1794p-4f, +0x1.d06f00p-29f, +0x1.75e45cp-53f, -0x1.d0483ep-78f },
	{ +0x1.db5270p-4f, +0x1.87d928p-32f, -0x1.0f52a4p-57f, +0x1.81f4a6p-84f },
	{ +0x1.c97978p-4f, +0x1.af1d24p-29f, -0x1.0977d0p-60f, -0x1.8839d0p-84f },
	{ +0x1.b78c84p-4f, -0x1.44f124p-28f, -0x1.ef7bc4p-52f, +0x1.9e0650p-78f },
	{ +0x1.a58b60p-4f, +0x1.856464p-29f, +0x1.c651d0p-55f, +0x1.b06b0cp-79f },
	{ +0x1.9375e4p-4f, +0x1.5595ecp-28f, +0x1.dc3738p-52f, +0x1.86c89ap-81f },
	{ +0x1.814be4p-4f, -0x1.c073fcp-28f, -0x1.371f88p-53f, -0x1.5f4080p-77f },
	{ +0x1.6f0d28p-4f, +0x1.5cad68p-29f, +0x1.737c94p-53f, -0x1.9184bap-78f },
	{ +0x1.60658cp-4f, -0x1.6c8af4p-28f, +0x1.d8ef74p-55f, +0x1.c4f792p-80f },
	{ +0x1.4e0110p-4f, +0x1.146b5cp-29f, +0x1.73f7ccp-54f, -0x1.d28db8p-79f },
	{ +0x1.3b8758p-4f, +0x1.8b1b70p-28f, -0x1.20aca4p-52f, -0x1.651894p-76f },
	{ +0x1.28f834p-4f, +0x1.43b6a4p-30f, -0x1.452af8p-55f, +0x1.976892p-80f },
	{ +0x1.1a0fbcp-4f, -0x1.e4075cp-28f, +0x1.1fe618p-52f, +0x1.9d6dc2p-77f },
	{ +0x1.075984p-4f, -0x1.4ce370p-29f, -0x1.d9fc98p-53f, +0x1.4ccf12p-77f },
	{ +0x1.f0a30cp-5f, +0x1.162a68p-37f, -0x1.e83368p-61f, -0x1.d222a6p-86f },
	{ +0x1.cae730p-5f, -0x1.1a8f7cp-31f, -0x1.5f9014p-55f, +0x1.2720c0p-79f },
	{ +0x1.ac9724p-5f, -0x1.e8ee08p-29f, +0x1.a7de04p-54f, -0x1.9bba74p-78f },
	{ +0x1.868a84p-5f, -0x1.ef8128p-30f, +0x1.dc5eccp-54f, -0x1.58d250p-79f },
	{ +0x1.67f950p-5f, -0x1.ed684cp-30f, -0x1.f060c0p-55f, -0x1.b1294cp-80f },
	{ +0x1.494accp-5f, +0x1.a6c890p-32f, -0x1.c3ad48p-56f, -0x1.6dc66cp-84f },
	{ +0x1.22c71cp-5f, -0x1.8abe2cp-32f, -0x1.7e7078p-56f, -0x1.ddc3dcp-86f },
	{ +0x1.03d5d8p-5f, +0x1.79cfbcp-31f, -0x1.da7c4cp-58f, +0x1.4e7582p-83f },
	{ +0x1.c98d18p-6f, +0x1.a01904p-31f, -0x1.854164p-55f, +0x1.883c36p-79f },
	{ +0x1.8b31fcp-6f, -0x1.356500p-30f, +0x1.c3ab48p-55f, +0x1.b69bdap-80f },
	{ +0x1.3cea44p-6f, +0x1.a352bcp-33f, -0x1.8865acp-57f, -0x1.48159cp-81f },
	{ +0x1.fc0a8cp-7f, -0x1.e07f84p-32f, +0x1.e7cf6cp-58f, +0x1.3a69c0p-82f },
	{ +0x1.7dc474p-7f, +0x1.f810a8p-31f, -0x1.245b5cp-56f, -0x1.a1f4f8p-80f },
	{ +0x1.fe02a8p-8f, -0x1.4ef988p-32f, +0x1.1f86ecp-57f, +0x1.20723cp-81f },
	{ +0x1.ff00acp-9f, -0x1.d4ef44p-33f, +0x1.2821acp-63f, +0x1.5a6d32p-87f },
	{ 0, 0, 0, 0 }
	};

	private Deviance() {
	}

	private static void addSplit(double[] accumulator, double value) {
		double integer = floor(value + 0.5);
		accumulator[0] += integer;
		accumulator[1] += value - integer;
	}

	static void extendedBd0(double x, double mean, double[] result) {
		if (result == null || result.length < 2)
			throw new IllegalArgumentException("result must contain at least two elements");

		double[] accumulator = {0.0, 0.0}; // high, low
		if (x == mean) {
			result[0] = result[1] = 0.0;
			return;
		}
		if (x == 0.0) {
			result[0] = mean;
			result[1] = 0.0;
			return;
		}
		if (mean == 0.0) {
			result[0] = Double.POSITIVE_INFINITY;
			result[1] = 0.0;
			return;
		}
		if (mean / x == Double.POSITIVE_INFINITY) {
			result[0] = mean;
			result[1] = 0.0;
			return;
		}

		int[] exponent = {0};
		double ratio = MathFunctions.frexp(mean / x, exponent);
		int e = exponent[0];
		if (M_LN2 * -(double) e > 1.0 + Double.MAX_VALUE / x) {
			result[0] = Double.POSITIVE_INFINITY;
			result[1] = 0.0;
			return;
		}

		int index = (int) floor((ratio - 0.5) * (2.0 * SCALE_SIZE) + 0.5);
		double scale = floor((1 << SCALE_BITS)
				/ (0.5 + index / (2.0 * SCALE_SIZE)) + 0.5);
		double factor = scalb(scale, -(e + SCALE_BITS));
		if (factor == Double.POSITIVE_INFINITY) {
			result[0] = factor;
			result[1] = 0.0;
			return;
		}

		addSplit(accumulator,
				-x * MathFunctions.log1pmx((mean * factor - x) / x));
		if (factor != 1.0) {
			for (int j = 0; j < 4; j++) {
				addSplit(accumulator, x * SCALE[index][j]);
				addSplit(accumulator, -x * SCALE[0][j] * e);
				if (!Double.isFinite(accumulator[0])) {
					result[0] = Double.POSITIVE_INFINITY;
					result[1] = 0.0;
					return;
				}
			}
			addSplit(accumulator, mean);
			addSplit(accumulator, -mean * factor);
		}

		result[0] = accumulator[0];
		result[1] = accumulator[1];
	}
}
