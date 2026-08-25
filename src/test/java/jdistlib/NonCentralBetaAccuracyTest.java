package jdistlib;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import jdistlib.rng.MersenneTwister;

/** Reference values from 100-decimal Poisson mixtures evaluated with mpmath. */
public class NonCentralBetaAccuracyTest {

	@Test
	public void cdfComputesBothTailsDirectlyOnTheLogScale() {
		assertEquals(-.0766532687446536588568411685997801,
			NonCentralBeta.cumulative(.3, 2.5, 7, 11, false, true), 3e-15);
		assertEquals(-103.889584467792241873302147622278,
			NonCentralBeta.cumulative(.8, 5, .5, 1000, true, true), 2e-13);
		assertEquals(-90.1033103891557949930248339414584,
			NonCentralBeta.cumulative(.01, .25, 40, 200, true, true), 1e-13);
		assertEquals(-561.300515230231440070453451261239,
			NonCentralBeta.cumulative(.041571470865891254, 1.0149511645559415,
				126.03373615036026, 1370.9880095911913, true, true), 5e-13);
		assertEquals(-255.338879070244881404777315694966,
			NonCentralBeta.cumulative(.847507562537541, 3.009369292533997,
				187.95519621829496, 37.405600013184028, false, true), 5e-13);
		assertEquals(-1.28154483441059274729743972442e-111,
			NonCentralBeta.cumulative(.847507562537541, 3.009369292533997,
				187.95519621829496, 37.405600013184028, true, true), 3e-125);
	}

	@Test
	public void cdfRetainsProbabilitiesBelowDoubleUnderflow() {
		assertEquals(-774.071932851500784583717105502188,
			NonCentralBeta.cumulative(1e-40, 1.0149511645559415,
				126.03373615036026, 1370.9880095911913, true, true), 1e-12);
		assertEquals(0., NonCentralBeta.cumulative(1e-40, 1.0149511645559415,
			126.03373615036026, 1370.9880095911913, true, false), 0.);
	}

	@Test
	public void quantileInvertsHighPrecisionAndUnderflowingLogTails() {
		assertEquals(.714379926903104725858801470722063,
			NonCentralBeta.quantile(.91, 2.5, 7, 11, true, false), 2e-15);
		assertEquals(.01, NonCentralBeta.quantile(-90.103310389155794993,
			.25, 40, 200, true, true), 2e-17);
		assertEquals(1e-40, NonCentralBeta.quantile(-774.07193285150078458,
			1.0149511645559415, 126.03373615036026, 1370.9880095911913,
			true, true), 1e-52);
	}

	@Test
	public void randomUsesOnlyOneNoncentralChiSquareComponent() {
		MersenneTwister expectedRandom = new MersenneTwister(310);
		double numerator = NonCentralChiSquare.random(5., 11., expectedRandom);
		double denominator = Gamma.random(7., 2., expectedRandom);
		double ratio = numerator > denominator ? 1. / (1. + denominator / numerator)
			: (numerator / denominator) / (1. + numerator / denominator);

		assertEquals(ratio,
			NonCentralBeta.random(2.5, 7, 11, new MersenneTwister(310)), 0.);
	}
}
