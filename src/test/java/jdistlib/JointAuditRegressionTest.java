/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import org.junit.Test;
import static org.junit.Assert.*;
import jdistlib.finance.*;
import jdistlib.generic.GenericDistribution;

public class JointAuditRegressionTest {
	@Test public void stronglyDependentArchimedeanSamplesKeepUniformMargins() {
		for(Copula copula:new Copula[]{new ClaytonCopula(2,1000),new FrankCopula(2,1000),new FrankCopula(2,-1000)}) {
			double[][] sample=copula.random(10000,14123L);double first=0,second=0;
			for(double[] point:sample){first+=point[0];second+=point[1];}
			assertEquals(.5,first/sample.length,.015);assertEquals(.5,second/sample.length,.015);
		}
		Copula frank=new FrankCopula(2,1000);
		assertEquals(.9-Math.log(2)/1000,frank.cumulative(new double[]{.9,.9}),2e-15);
		assertEquals(Math.log(250),frank.logDensity(new double[]{.9,.9}),2e-14);
	}
	@Test public void bivariateEllipticalCdfsHaveExactOrthantsAndUniformMargins() {
		for(double rho:new double[]{-.99,-.9,0,.7,.99}) {
			double[][] r={{1,rho},{rho,1}};
			for(Copula c:new Copula[]{new GaussianCopula(r),new StudentTCopula(r,.3),new StudentTCopula(r,5)}) {
				assertEquals(.25+Math.asin(rho)/(2*Math.PI),c.cumulative(new double[]{.5,.5}),2e-8);
				assertEquals(.3,c.cumulative(new double[]{.3,1}),0);
				assertTrue(c.cumulative(new double[]{.01,1-1e-12})<=.01);
			}
		}
	}
	@Test public void discreteMultivariateDensitiesAvoidLargeShapeAndPopulationCancellation() {
		assertEquals(.5,MultivariateHypergeometric.density(new int[]{1,0},
				new int[]{1000000000,1000000000},1,false),2e-14);
		assertEquals(.25,DirichletMultinomial.density(new double[]{1,0},1,new double[]{1e16,3e16},false),2e-14);
		assertEquals(.25,DirichletMultinomial.probability(new int[]{1,0},new int[]{1,0},1,new double[]{1e16,3e16}),2e-14);
	}
	@Test public void strongArchimedeanCopulasRemainFiniteAndRespectMargins() {
		Copula clayton = new ClaytonCopula(2, 1000);
		assertEquals(.4 * Math.pow(2.0, -.001), clayton.cumulative(new double[]{.4,.4}), 2e-15);
		assertTrue(Double.isFinite(clayton.logDensity(new double[]{1e-100,1e-100})));
		Copula joe = new JoeCopula(1000);
		assertEquals(1.0-.3*Math.pow(2.0,.001),joe.cumulative(new double[]{.7,.7}),2e-15);
		assertTrue(Double.isFinite(joe.logDensity(new double[]{1-1e-12,1-1e-12})));
		assertEquals(0.0,new JoeCopula(1).logDensity(new double[]{1e-12,1e-12}),0.0);
		assertEquals(2.0-Math.PI*Math.PI/6.0,new JoeCopula(2).kendallsTau(0,1),2e-15);
		assertEquals(new JoeCopula(2).kendallsTau(0,1),new JoeCopula(2+1e-10).kendallsTau(0,1),3e-11);
		assertTrue(joe.kendallsTau(0,1)>.998);
	}
	@Test public void bb1DensityHasExactClaytonAndGumbelLimits() {
		for (double u : new double[]{1e-12,.001,.4,.99,1-1e-12}) {
			for (double v : new double[]{1e-12,.2,.9,1-1e-12}) {
				double[] point={u,v};
				assertEquals(new ClaytonCopula(2,4).logDensity(point),new BB1Copula(4,1).logDensity(point),0.0);
				assertEquals(new GumbelCopula(2,2).logDensity(point),new BB1Copula(0,2).logDensity(point),0.0);
			}
		}
	}
	@Test public void analyticJoeAndBb1DensitiesIntegrateToConditionalProbabilities() {
		// Integrate the joint density along the second coordinate; the CDF
		// derivative independently determines the conditional probability.
		for (Copula c : new Copula[]{new JoeCopula(2.7),new BB1Copula(1.3,2.1)}) {
			double first=.37,upper=.81,step=upper/20000,sum=0;
			for(int i=0;i<20000;i++) sum+=c.density(new double[]{first,(i+.5)*step});
			double h=1e-5;
			double derivative=(c.cumulative(new double[]{first+h,upper})-c.cumulative(new double[]{first-h,upper}))/(2*h);
			assertEquals(derivative,sum*step,2e-8);
			PairCopula pair=new PairCopula(c);
			assertEquals(derivative,pair.conditionalSecondGivenFirst(first,upper),2e-8);
			assertEquals(upper,pair.inverseSecondGivenFirst(first,pair.conditionalSecondGivenFirst(first,upper)),2e-14);
		}
	}
	@Test public void strongPairConditionalsAvoidOverflowAndCancellation() {
		for(double theta:new double[]{-1000,-100,-.00001,.00001,100,1000}) {
			PairCopula pair=new PairCopula(new FrankCopula(2,theta));
			for(double u:new double[]{.01,.4,.99}) for(double p:new double[]{.001,.2,.8,.999}) {
				double v=pair.inverseSecondGivenFirst(u,p);
				assertEquals(theta+" "+u+" "+p,p,pair.conditionalSecondGivenFirst(u,v),2e-11);
			}
		}
		assertTrue(Double.isFinite(new PairCopula(new ClaytonCopula(2,1000)).conditionalSecondGivenFirst(1e-20,1e-20)));
		assertTrue(Double.isFinite(new PairCopula(new GumbelCopula(2,1000)).conditionalSecondGivenFirst(1e-20,1e-20)));
	}
	@Test public void multivariateStudentHasAccurateLargeDfNormalLimit() {
		for(int d:new int[]{1,2,3,5,10}) {
			double[] x=new double[d],mean=new double[d];double[][] covariance=new double[d][d];
			for(int i=0;i<d;i++){x[i]=.5;for(int j=0;j<d;j++)covariance[i][j]=i==j?1:.3;}
			assertEquals(MultivariateNormal.density(x,mean,covariance,true),
					MultivariateStudentT.density(x,mean,covariance,1e16,true),2e-13);
		}
		double[][] correlation={{1,.3},{.3,1}};double[] u={.2,.7};
		assertEquals(new GaussianCopula(correlation).logDensity(u),new StudentTCopula(correlation,1e16).logDensity(u),2e-13);
	}
	@Test public void conditioningWorksWhenBothOrdinaryCdfsRoundToOneOrZero() {
		for(double low:new double[]{40,-41}) {
			ConditionalDistribution c=new ConditionalDistribution(new Normal(),low,low+1);
			for(double p:new double[]{.01,.5,.99}) {
				double q=c.quantile(p,true,false);
				assertTrue(q>low&&q<low+1);
				assertEquals(p,c.cumulative(q,true,false),4e-12);
				assertEquals(1-p,c.cumulative(q,false,false),4e-12);
				assertTrue(Double.isFinite(c.density(q,true)));
			}
			assertTrue(Double.isNaN(c.quantile(2,true,false)));
		}
	}
	@Test public void orderStatisticsPreserveLogTailsAndAtoms() {
		OrderStatisticDistribution max=OrderStatisticDistribution.maximum(new Normal(),100);
		OrderStatisticDistribution min=OrderStatisticDistribution.minimum(new Normal(),100);
		assertEquals(100*Normal.cumulative(-40,0,1,true,true),max.cumulative(-40,true,true),0);
		assertEquals(100*Normal.cumulative(40,0,1,false,true),min.cumulative(40,false,true),0);
		assertTrue(Double.isFinite(max.density(-40,true)));
		assertTrue(Double.isFinite(min.quantile(-1000,true,true)));
		EmpiricalDistribution empirical=new EmpiricalDistribution(new double[]{0,0,0,1});
		assertEquals(.75*.75,OrderStatisticDistribution.maximum(empirical,2).density(0,false),2e-15);
		assertEquals(1-.25*.25,OrderStatisticDistribution.minimum(empirical,2).density(0,false),2e-15);
		assertEquals(0,OrderStatisticDistribution.maximum(empirical,2).density(.5,false),0);
		OrderStatisticDistribution nested=OrderStatisticDistribution.maximum(max,2);
		assertTrue(nested.density(.1,false)>0);
	}
	@Test public void compoundCountTailsAndQuantilesAreConsistent() {
		for(GenericDistribution c:new GenericDistribution[]{new DelaporteDistribution(3,2,.4),new PolyaAeppliDistribution(3,.4)}) {
			assertTrue(Double.isNaN(c.density(Double.NaN,false)));
			assertTrue(Double.isNaN(c.cumulative(Double.NaN,true,false)));
			assertTrue(Double.isNaN(c.quantile(Double.NaN,true,false)));
			assertEquals(0,c.density(Double.POSITIVE_INFINITY,false),0);
			assertEquals(1,c.cumulative(Double.POSITIVE_INFINITY,true,false),0);
			assertEquals(Double.POSITIVE_INFINITY,c.quantile(1,true,false),0);
			double total=0;
			for(int n=0;n<100;n++) {
				total+=c.density(n,false);
				assertEquals(total,c.cumulative(n,true,false),3e-14);
				assertEquals(1,c.cumulative(n,true,false)+c.cumulative(n,false,false),3e-14);
			}
			assertTrue(Double.isFinite(c.cumulative(1000,false,true)));
			for(double p:new double[]{.01,.5,.99}) {
				double q=c.quantile(p,true,false);
				assertTrue(c.cumulative(q,true,false)>=p);
				assertTrue(q==0||c.cumulative(q-1,true,false)<p);
			}
		}
	}
	@Test public void stableSpecializationsRetainTheirExactTailContracts() {
		StableDistribution normal=new StableDistribution(2,.7,1,0);
		assertEquals(Normal.density(60,0,Math.sqrt(2),true),normal.density(60,true),0);
		assertEquals(Normal.cumulative(60,0,Math.sqrt(2),false,true),normal.cumulative(60,false,true),0);
		assertEquals(Normal.quantile(-1000,0,Math.sqrt(2),false,true),normal.quantile(-1000,false,true),0);
		assertTrue(normal.momentExists(10));
		assertEquals(3,new StableDistribution(.5,1,1,3).getLowerBound(),0);
		assertEquals(3,new StableDistribution(.5,-1,1,3).getUpperBound(),0);
	}
}
