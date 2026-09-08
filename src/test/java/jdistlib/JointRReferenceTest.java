/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import jdistlib.finance.*;
import jdistlib.generic.GenericDistribution;
import org.junit.Test;
import static org.junit.Assert.*;

/** Frozen R4.6.1/copula/VineCopula/mvtnorm/extraDistr references, with
 * independently adjudicated unreliable R regions excluded by the generator. */
public class JointRReferenceTest {
	@Test public void agreesWithTrustedRReferences() throws IOException {
		try(BufferedReader in=new BufferedReader(new InputStreamReader(
				getClass().getResourceAsStream("/jdistlib/audit2-joint.tsv"),StandardCharsets.UTF_8))) {
			String row;int count=0;
			while((row=in.readLine())!=null) {
				String[] a=row.split("\t");double[] actual,expected;double tolerance=2e-8;
				switch(a[0]) {
				case "copula": {
					double theta=p(a[2]);Copula c;
					switch(a[1]) {
					case "clayton":c=new ClaytonCopula(2,theta);break;
					case "gumbel":c=new GumbelCopula(2,theta);break;
					case "frank":c=new FrankCopula(2,theta);break;
					case "joe":c=new JoeCopula(theta);break;
					case "gaussian":c=new GaussianCopula(new double[][]{{1,theta},{theta,1}});break;
					default:c=new StudentTCopula(new double[][]{{1,theta},{theta,1}},5);
					}
					double[] uv={p(a[4]),p(a[5])};actual=new double[]{c.cumulative(uv),c.logDensity(uv),c.kendallsTau(0,1)};
					expected=new double[]{p(a[6]),p(a[7]),p(a[8])};tolerance=2e-7;break;
				}
				case "bb1": {
					BB1Copula c=new BB1Copula(p(a[1]),p(a[2]));double[] uv={p(a[3]),p(a[4])};
					actual=new double[]{c.cumulative(uv),c.logDensity(uv),new PairCopula(c).conditionalSecondGivenFirst(uv[0],uv[1])};
					expected=new double[]{p(a[5]),p(a[6]),p(a[7])};tolerance=2e-7;break;
				}
				case "multivariate": {
					int d=Integer.parseInt(a[1]);double[] x=new double[d],m=new double[d];Arrays.fill(x,p(a[3]));
					double[][] s=new double[d][d];for(int i=0;i<d;i++){Arrays.fill(s[i],.3);s[i][i]=1;}
					actual=new double[]{MultivariateNormal.density(x,m,s,true),MultivariateStudentT.density(x,m,s,p(a[2]),true)};
					expected=new double[]{p(a[4]),p(a[5])};break;
				}
				case "finance": {
					GenericDistribution c=a[1].equals("polya")?new PolyaAeppliDistribution(p(a[2]),p(a[4])):new DelaporteDistribution(p(a[2]),p(a[3]),p(a[4]));
					double x=p(a[5]);actual=new double[]{c.density(x,true),c.cumulative(x,true,true),c.cumulative(x,false,true)};
					expected=new double[]{p(a[6]),p(a[7]),p(a[8])};break;
				}
				default:
					actual=new double[]{Dirichlet.density(new double[]{p(a[4]),p(a[5]),p(a[6])},new double[]{p(a[1]),p(a[2]),p(a[3])},true)};
					expected=new double[]{p(a[7])};
				}
				for(int i=0;i<actual.length;i++){assertEquals(row+" component "+i,expected[i],actual[i],tolerance);count++;}
			}
			assertTrue("Reference set unexpectedly truncated",count>7000);
		}
	}
	private static double p(String s) {return s.equals("Inf")?Double.POSITIVE_INFINITY:s.equals("-Inf")?Double.NEGATIVE_INFINITY:Double.parseDouble(s);}
}
