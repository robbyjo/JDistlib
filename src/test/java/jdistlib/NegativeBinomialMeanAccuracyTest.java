package jdistlib;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class NegativeBinomialMeanAccuracyTest {
    @Test
    public void smallCountRetainsExactLogZeroMassAwayFromThePoissonLimit() {
        assertEquals(-693147180533.00743548186351856,
            NegBinomial.density_mu(1, 1e12, 1e12, true), 0.00013);
        assertEquals(-999972.5621283979624804354013345,
            NegBinomial.density_mu(2, 1e12, 1e6, true), 3e-10);
        assertEquals(-9982.27246643160957973727663282,
            NegBinomial.density_mu(2, 1e16, 10000, true), 4e-12);
        for (double size : new double[] {1e12, 1e16, 1e100}) {
            assertEquals(Math.log(size) - (size + 1) * Math.log(2),
                NegBinomial.density_mu(1, size, size, true), 2 * Math.ulp(size));
        }
    }

    @Test
    public void smallCountToSizeRatioDoesNotImplyTinyRisingFactorialCorrection() {
        // 100-digit log-gamma reference; x/size=1e-11 but x*x/(2*size)=0.05.
        assertEquals(-12.4318639981882344952035953432,
            NegBinomial.density_mu(1e10, 1e21, 1e10, true), 8e-15);
        double prob = 1.0 / (1.0 + 1e-11);
        // The exact double probability has a slightly different implied mean.
        assertEquals(NegBinomial.density_mu(1e10, 1e21, 1e21 * ((1 - prob) / prob), true),
            NegBinomial.density(1e10, 1e21, prob, true), 2e-11);
    }

    @Test
    public void zeroMassDoesNotOverflowTheParameterSumOrUnderflowTheirRatio() {
        assertEquals(-1e308 * Math.log(2), NegBinomial.density_mu(0, 1e308, 1e308, true),
            2 * Math.ulp(1e308));
        assertEquals(-1e-50, NegBinomial.density_mu(0, 1e308, 1e-50, true), 0);
        assertEquals(-1e-50 * Math.log(2), NegBinomial.density_mu(0, 1e-50, 1e-50, true), 1e-65);
    }
}
