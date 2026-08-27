/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.Arrays;

import jdistlib.disttest.MultipleTesting;
import jdistlib.disttest.MultipleTesting.GroupedFdrResult;
import jdistlib.disttest.MultipleTesting.Method;
import jdistlib.disttest.online.LordPlusPlus;
import jdistlib.disttest.online.OnlineFdr;

/** Batch, weighted, grouped, q-value, and online FDR examples. */
public final class FdrIntegrationExamples {
	private FdrIntegrationExamples() {}
	public static void main(String[] arguments) {
		double[] p = {0.001, 0.02, 0.04, 0.3, 0.8};
		System.out.println("BH=" + Arrays.toString(MultipleTesting.adjust(
				p, Method.BENJAMINI_HOCHBERG)));
		System.out.println("weighted BH=" + Arrays.toString(
				MultipleTesting.adjustWeightedBenjaminiHochberg(p,
						new double[] {2, 1, 1, 0.5, 0.5})));
		System.out.println("q-values=" + Arrays.toString(MultipleTesting.qValues(p, 1)));
		GroupedFdrResult grouped = MultipleTesting.selectiveGroupedBenjaminiHochberg(
				p, new int[] {1, 1, 2, 2, 2}, 0.05, 0.05);
		System.out.println("selected groups=" + grouped.getSelectedGroupCount());
		LordPlusPlus online = new LordPlusPlus(0.05, 0.025,
				OnlineFdr.polynomialGamma(1000, 1.6));
		System.out.println("online decision=" + online.test(0.001).isRejected());
	}
}
