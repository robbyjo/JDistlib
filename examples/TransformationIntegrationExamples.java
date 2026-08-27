/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.CensoredDistribution;
import jdistlib.Distributions;
import jdistlib.MonotoneTransformDistribution;
import jdistlib.Normal;
import jdistlib.TruncatedContinuousDistribution;

/** Affine, nonlinear, truncation, and censoring compositions. */
public final class TransformationIntegrationExamples {
	private TransformationIntegrationExamples() {}
	public static MonotoneTransformDistribution logNormalFromNormal() {
		return Distributions.transform(new Normal(0, 0.5), Math::exp, Math::log,
				y -> -Math.log(y), true, 0, Double.POSITIVE_INFINITY);
	}
	public static MonotoneTransformDistribution calibratedMeasurement() {
		return Distributions.affine(new Normal(50, 8), 2, 1.1);
	}
	public static CensoredDistribution observedPositiveMeasurement() {
		TruncatedContinuousDistribution positive = Distributions.truncate(
				new Normal(50, 12), 0, Double.POSITIVE_INFINITY);
		return Distributions.censor(positive, 5, 100);
	}
	public static void main(String[] arguments) {
		MonotoneTransformDistribution positive = logNormalFromNormal();
		System.out.println("P(Y<=1)=" + positive.cumulative(1));
		System.out.println("Y p90=" + positive.quantile(0.9));
		CensoredDistribution observed = observedPositiveMeasurement();
		System.out.println("upper atom=" + observed.getUpperAtomProbability());
	}
}
