import jdistlib.Binomial;
import jdistlib.ClaytonCopula;
import jdistlib.CopulaFamily;
import jdistlib.CopulaFitOptions;
import jdistlib.CopulaMarginal;
import jdistlib.CopulaMeasureResult;
import jdistlib.CopulaSelectionCriterion;
import jdistlib.CopulaSelectionResult;
import jdistlib.CopulaSelector;
import jdistlib.MixedCopulaDistribution;
import jdistlib.Normal;

/** Mixed-marginal likelihood and automatic-selection example. */
public final class CopulaExpansionExamples {
    private CopulaExpansionExamples() {}

    public static void main(String[] args) {
        MixedCopulaDistribution joint = new MixedCopulaDistribution(
                new ClaytonCopula(2, 1.2),
                CopulaMarginal.continuous(new Normal()),
                CopulaMarginal.discrete(new Binomial(1, 0.35)));
        CopulaMeasureResult value = joint.measure(new double[] {0.3, 1.0});
        System.out.println(value.logValue + " " + value.getStatus());

        double[][] observations = joint.random(250, 20260826L);
        CopulaSelectionResult selected = CopulaSelector.selectMixed(observations,
                new CopulaMarginal[] {
                    CopulaMarginal.continuous(new Normal()),
                    CopulaMarginal.discrete(new Binomial(1, 0.35))
                }, null, new CopulaFitOptions(), CopulaSelectionCriterion.BIC,
                CopulaFamily.INDEPENDENCE, CopulaFamily.GAUSSIAN,
                CopulaFamily.CLAYTON, CopulaFamily.FRANK);
        System.out.println(selected.getSelected().getFamily());
    }
}
