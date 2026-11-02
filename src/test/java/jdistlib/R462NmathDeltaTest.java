package jdistlib;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import jdistlib.math.Bessel;

/** Regression vectors for the tagged R 4.6.1-to-4.6.2 src/nmath delta. */
public class R462NmathDeltaTest {
	@Test
	public void besselJLargeArgumentsMatchR462() {
		assertEquals(-0.0024828760514086267, Bessel.j(100_001., 0.), 5e-18);
		assertEquals(0.0003310430137398697, Bessel.j(1_000_000., 0.), 5e-19);
		assertEquals(2.8027763777383449e-05, Bessel.j(1_000_000., .25), 5e-20);
		assertEquals(-8.9301074693427334e-06,
				Bessel.j(2_147_483_648., 0.), 5e-21);
		assertEquals(4.0946478671133201e-06,
				Bessel.j(2_147_483_648., 3.5), 5e-21);
		assertEquals(-9.2768780984938035e-07, Bessel.j(1e10, .25), 5e-22);
		// Argument reduction at the documented upper cutoff is platform-sensitive.
		assertEquals(8.0982226090896178e-14, Bessel.j(1e24, 0.), 2e-25);
		assertEquals(0., Bessel.j(Math.nextUp(1e24), 0.), 0.);
	}
}
