import java.util.Arrays;

import jdistlib.Copula;
import jdistlib.CopulaDistribution;
import jdistlib.Exponential;
import jdistlib.GaussianCopula;
import jdistlib.Normal;

/** Minimal continuous-marginal copula example compiled by the check task. */
public final class CopulaExamples {
    private CopulaExamples() {}

    public static void main(String[] args) {
        Copula copula = GaussianCopula.fromKendallsTau(
                new double[][] {{1.0, 0.45}, {0.45, 1.0}});
        CopulaDistribution joint = new CopulaDistribution(copula,
                new Normal(10.0, 2.0), new Exponential(3.0));
        System.out.println(joint.logDensity(new double[] {11.0, 2.0}));
        System.out.println(Arrays.toString(joint.random(20260826L)));
    }
}
