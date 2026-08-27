/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import java.util.Arrays;

import jdistlib.Binomial;
import jdistlib.ClaytonCopula;
import jdistlib.Copula;
import jdistlib.CopulaDistribution;
import jdistlib.CopulaMarginal;
import jdistlib.CopulaMeasureResult;
import jdistlib.Exponential;
import jdistlib.GaussianCopula;
import jdistlib.MixedCopulaDistribution;
import jdistlib.Normal;

/** Continuous and mixed-marginal copula workflows. */
public final class CopulaIntegrationExamples {
	private CopulaIntegrationExamples() {}
	public static CopulaDistribution continuousJointModel() {
		Copula dependence = GaussianCopula.fromKendallsTau(
				new double[][] {{1.0, 0.45}, {0.45, 1.0}});
		return new CopulaDistribution(dependence,
				new Normal(10, 2), new Exponential(3));
	}
	public static MixedCopulaDistribution mixedJointModel() {
		return new MixedCopulaDistribution(new ClaytonCopula(2, 1.2),
				CopulaMarginal.continuous(new Normal()),
				CopulaMarginal.discrete(new Binomial(1, 0.35)));
	}
	public static void main(String[] arguments) {
		CopulaDistribution continuous = continuousJointModel();
		System.out.println("joint CDF=" + continuous.cumulative(new double[] {11, 2}));
		System.out.println("draw=" + Arrays.toString(continuous.random(20260827L)));
		MixedCopulaDistribution mixed = mixedJointModel();
		CopulaMeasureResult measure = mixed.measure(new double[] {0.3, 1});
		System.out.println("mixed log measure=" + measure.logValue
				+ " status=" + measure.getStatus());
		System.out.println("250 mixed draws=" + mixed.random(250, 20260827L).length);
	}
}
