/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import static org.junit.Assert.*;
import org.junit.Test;
import jdistlib.evd.*;
import jdistlib.generic.GenericDistribution;
import jdistlib.rng.MersenneTwister;

public class ScalarRemainingAuditTest {
    @Test public void betaBinomialUniformMixtureAndRandomAreCorrect() {
        BetaBinomial d=new BetaBinomial(.5,.5,20);
        for(int i=0;i<=20;i++) {
            assertEquals(1.0/21,d.density(i,false),2e-15);
            assertEquals((i+1.0)/21,d.cumulative(i),2e-14);
        }
        assertEquals(10,d.quantile(.5),0);
        assertEquals(15,d.quantile(Math.log(.25),false,true),0);
        assertEquals(0,d.density(-1,false),0);
        assertEquals(d.cumulative(2),d.cumulative(2.9),0);
        MersenneTwister rng=new MersenneTwister(412);double sum=0;
        for(int i=0;i<10000;i++){double x=BetaBinomial.random(.3,.2,20,rng);assertTrue(x>=0&&x<=20);sum+=x;}
        assertEquals(6,sum/10000,.15);
    }
    @Test(timeout=2000) public void zipfQuantileTerminatesAndLoggedSurvivalIsNormalized() {
        Zipf d=new Zipf(3,1);
        assertEquals(Math.log(5.0/11),d.cumulative(1,false,true),2e-15);
        assertEquals(1,d.quantile(.5),0);
        assertEquals(2,d.quantile(.7),0);
        assertEquals(3,d.quantile(.9),0);
        assertEquals(1,d.quantile(0),0);
        assertEquals(0,d.density(1.5,false),0);
        assertEquals(1,d.cumulative(-1,false,false),0);
    }
    @Test public void logarithmicHasPositiveIntegerSupportAndValidQuantiles() {
        Logarithmic d=new Logarithmic(.5);
        assertEquals(1,d.quantile(.5),0);
        assertEquals(2,d.quantile(.9),0);
        assertEquals(2,d.quantile(Math.log(.1),false,true),0);
        assertEquals(0,d.density(0,false),0);
        assertEquals(0,d.density(1.5,false),0);
        assertEquals(1,d.cumulative(0,false,false),0);
        double exact=0;for(int i=101;i<300;i++)exact+=Math.pow(.5,i)/(i*Math.log(2));
        assertEquals(Math.log(exact),d.cumulative(100,false,true),5e-13);
    }
    @Test public void halfDistributionsResolveQuantilesVeryCloseToZero() {
        for(GenericDistribution d:new GenericDistribution[]{new PositiveNormal(-1,.3),new HalfNormal(2),new HalfT(3,2)}) {
            for(double lp:new double[]{-20,-100,-600}) {
                double x=d.quantile(lp,true,true);assertTrue(x>0);
                assertEquals(lp,d.cumulative(x,true,true),2e-12);
            }
        }
        assertEquals(144,DiscreteWeibull.quantile(1e-12,.1,.5,false,false),0);
        double q=DiscreteWeibull.quantile(-1000,.7,2,false,true);
        assertTrue(DiscreteWeibull.cumulative(q,.7,2,false,true)<=-1000);
        assertTrue(DiscreteWeibull.cumulative(q-1,.7,2,false,true)>-1000);
    }
    @Test public void poissonInverseGaussianKeepsFiniteLogMassAndSurvival() {
        double expected=0;for(int i=21;i<100;i++)expected+=PoissonInverseGaussian.density(i,.3,.3,false);
        assertEquals(Math.log(expected),PoissonInverseGaussian.cumulative(20,.3,.3,false,true),2e-13);
        assertTrue(Double.isFinite(PoissonInverseGaussian.density(1000,.3,.3,true)));
        assertEquals(-1e-12,PoissonInverseGaussian.density(0,1e-12,1,true),1e-25);
    }
    @Test public void skewTAndTukeyLambdaContainTheirSymmetricLimits() {
        for(double x:new double[]{-100,-2,-.1,0,.1,2,100}) for(boolean lower:new boolean[]{true,false}) {
            assertEquals(T.cumulative(x,5,lower,true),SkewedT.cumulative(x,5,1,lower,true),2e-14);
            assertEquals(Logistic.cumulative(x,0,1,lower,true),TukeyLambda.cumulative(x,0,lower,true),2e-12);
        }
        for(double lp:new double[]{-.01,-2,-100}) for(boolean lower:new boolean[]{true,false}) {
            double x=SkewedT.quantile(lp,5,2,lower,true);assertEquals(lp,SkewedT.cumulative(x,5,2,lower,true),2e-12);
            assertEquals(Logistic.quantile(lp,0,1,lower,true),TukeyLambda.quantile(lp,0,lower,true),1e-13);
        }
        assertEquals(1,TukeyLambda.density(.5,2,false),0);
    }
    @Test public void slashKeepsExtremeLogTailsAndDensity() {
        Slash d=new Slash(0,1);
        for(boolean lower:new boolean[]{true,false}) for(double lp:new double[]{-.1,-2,-100,-600}) {
            double x=d.quantile(lp,lower,true);assertEquals(lp,d.cumulative(x,lower,true),2e-12);
        }
        assertTrue(Double.isFinite(d.density(1e200,true)));
        assertEquals(-Math.log(2),d.cumulative(0,true,true),0);
    }
    @Test public void arcsineHasExactContinuousInverseAndBetaDensity() {
        Arcsine d = new Arcsine(-2.0, 3.0);
        for (double p : new double[] {1e-8, .01, .25, .5, .9}) {
            double x = d.quantile(p);
            assertEquals(-2.0 + 5.0 * Math.pow(Math.sin(Math.PI * p / 2), 2), x, 2e-15);
            assertEquals(p, d.cumulative(x), 2e-9);
        }
        assertEquals(.5, d.cumulative(.5), 1e-15);
        assertEquals(0.0, d.density(-3, false), 0.0);
        assertEquals(1.0, d.cumulative(-3, false, false), 0.0);
    }
    @Test public void chiHasCorrectJacobianAndSupport() {
        for (double x : new double[] {.01, .3, 1, 2, 8}) {
            assertEquals(2 * Normal.density(x, 0, 1, false), Chi.density(x, 1, false), 2e-15);
            assertEquals(-Math.expm1(-x*x/2), Chi.cumulative(x, 2, true, false), 2e-15);
            assertEquals(x * Math.exp(-x*x/2), Chi.density(x, 2, false), 2e-15);
        }
        assertEquals(Math.sqrt(2/Math.PI), Chi.density(0,1,false), 0);
        assertEquals(0, Chi.cumulative(-2,2,true,false),0);
    }
    @Test public void reciprocalGammaPreservesLegacyScaleWithCorrectTails() {
        for (double x : new double[] {.05, .3, 1, 5, 100}) {
            assertEquals(Math.exp(-1/(2*x)), InvGamma.cumulative(x,1,2,true,false), 1e-15);
            assertEquals(Math.exp(-1/(2*x))/(2*x*x),InvGamma.density(x,1,2,false),2e-14);
        }
        assertEquals(-1/(2*Math.log(.3)), InvGamma.quantile(.3,1,2,true,false),1e-14);
        assertEquals(0,InvGamma.quantile(0,1,2,true,false),0);
    }
    @Test public void inverseGaussianMatchesStatmodAndExactDensity() {
        assertEquals(0.10377687435514868, InvNormal.density(2,1,.5,false), 2e-16);
        assertEquals(0.59441064130196886, InvNormal.cumulative(1,1,.5,true,false), 3e-15);
        for (double p : new double[]{-1e-12,-.1,-2,-20,-100}) for (boolean lower : new boolean[]{true,false}) {
            double q=InvNormal.quantile(p,2,.7,lower,true);
            assertEquals(p,InvNormal.cumulative(q,2,.7,lower,true),3e-10);
        }
        assertEquals(0,InvNormal.density(0,1,1,false),0);
        assertEquals(1,InvNormal.cumulative(0,1,1,false,false),0);
        assertEquals(Double.POSITIVE_INFINITY,InvNormal.quantile(1,1,1,true,false),0);
    }
    @Test public void inverseGaussianFastSamplerHasCorrectMoments() {
        MersenneTwister random=new MersenneTwister(842342L);double sum=0,ss=0;int n=100000;
        for(int i=0;i<n;i++){double v=InvNormal.random(2,.5,random);assertTrue(v>0);sum+=v;ss+=v*v;}
        double mean=sum/n;
        assertEquals(2,mean,.025);
        assertEquals(2,ss/n-mean*mean,.10);
    }
    @Test public void kumaraswamyObjectApiIsContinuousAndQuantileUsesRequestedTail() {
        Kumaraswamy d=new Kumaraswamy(2,3);double x=.4;
        assertEquals(6*x*Math.pow(1-x*x,2),d.density(x,false),2e-15);
        assertEquals(Math.pow(1-x*x,3),d.cumulative(x,false,false),2e-15);
        for(boolean lower:new boolean[]{true,false}) for(double p:new double[]{.01,.2,.8,1e-12}) {
            double q=d.quantile(Math.log(p),lower,true);
            assertEquals(Math.log(p),d.cumulative(q,lower,true),3e-8);
        }
        assertEquals(3,Kumaraswamy.density(0,1,3,false),0);
        assertEquals(2,Kumaraswamy.density(1,2,1,false),0);
    }
    @Test public void logLogisticObjectApiAndLogQuantilesMatchLogisticTransform() {
        LogLogistic d=new LogLogistic(2,3);
        for(double x:new double[]{.3,.8,3,100,1e100}) {
            assertEquals(Logistic.cumulative(Math.log(x),Math.log(3),.5,false,true),d.cumulative(x,false,true),3e-14);
            assertEquals(LogLogistic.density(x,2,3,false),d.density(x,false),0);
        }
        for(boolean lower:new boolean[]{true,false}) for(double lp:new double[]{-.001,-2,-100}) {
            double q=d.quantile(lp,lower,true);assertEquals(lp,d.cumulative(q,lower,true),2e-13);
        }
    }
    @Test public void levySurvivalIsAProbabilityAndEndpointsAreCorrect() {
        assertEquals(0.6826894921370859, Levy.cumulative(1,0,1,false,false),2e-15);
        for(boolean lower:new boolean[]{true,false}) for(double lp:new double[]{-.01,-2,-100}) {
            double q=Levy.quantile(lp,0,1,lower,true);assertEquals(lp,Levy.cumulative(q,0,1,lower,true),3e-13);
        }
        assertEquals(0,Levy.quantile(0,0,1,true,false),0);
        assertEquals(Double.POSITIVE_INFINITY,Levy.quantile(1,0,1,true,false),0);
    }
    @Test public void generalizedParetoExponentialLimitAndRandomSupport() {
        GeneralizedPareto d=new GeneralizedPareto(2,3,0);
        for(double p:new double[]{.01,.2,.8,1-1e-12}) assertEquals(2-3*Math.log1p(-p),d.quantile(p),2e-14);
        assertEquals(0,d.density(1,false),0);
        assertEquals(1,GeneralizedPareto.cumulative(1,2,3,0,false),0);
        assertEquals(0,GeneralizedPareto.density(10,2,3,-.5,false),0);
        MersenneTwister random=new MersenneTwister(42);double sum=0;
        for(int i=0;i<20000;i++){double v=GeneralizedPareto.random(2,3,0,random);assertTrue(v>=2);sum+=v;}
        assertEquals(5,sum/20000,.08);
    }
    @Test public void extremeValueFamiliesRetainSmallAndLoggedTails() {
        GenericDistribution[] laws={new GEV(0,1,0),new GEV(0,1,1e-10),new GeneralizedPareto(0,1,.2),new Fretchet(0,1,2),new ReverseWeibull(0,1,2),new Rayleigh(1),new Gumbel(0,1)};
        for(GenericDistribution d:laws) for(boolean lower:new boolean[]{true,false}) for(double lp:new double[]{-.1,-2,-20,-100}) {
            double q=d.quantile(lp,lower,true);assertEquals(d.getClass().getSimpleName(),lp,d.cumulative(q,lower,true),2e-9);
        }
        assertEquals(2/Math.E,Fretchet.density(1,0,1,2,false),1e-15);
        assertEquals(1/Math.E,Fretchet.cumulative(1,0,1,2,true),1e-15);
        assertEquals(0,ReverseWeibull.density(1,0,1,2,false),0);
        assertEquals(1,Rayleigh.cumulative(-1,1,false),0);
        assertEquals(-5000,new Rayleigh(1).cumulative(100,false,true),0);
    }
    @Test public void orderStatisticsAgreeWithUniformBetaLaw() {
        GenericDistribution uniform=new Uniform(0,1);
        for(int n:new int[]{1,5,100}) for(int rank:new int[]{1,n}) for(boolean largest:new boolean[]{false,true}) {
            int k=largest?n+1-rank:rank;Order order=new Order(uniform,n,rank,largest);
            for(double x:new double[]{.1,.3,.8}) for(boolean lower:new boolean[]{false,true}) {
                assertEquals(Beta.density(x,k,n+1-k,true),order.density(x,true),5e-13);
                assertEquals(Beta.cumulative(x,k,n+1-k,lower,true),order.cumulative(x,lower,true),5e-13);
                assertEquals(Beta.quantile(x,k,n+1-k,lower,false),order.quantile(x,lower,false),1e-13);
            }
        }
        Extreme maximum=new Extreme(uniform,5,true), minimum=new Extreme(uniform,5,false);
        assertEquals(Math.pow(.3,5),maximum.cumulative(.3),1e-16);
        assertEquals(1-Math.pow(.7,5),minimum.cumulative(.3),1e-15);
        assertEquals(Math.pow(.8,.2),maximum.quantile(.2,false,false),2e-15);
        assertEquals(0,maximum.density(-1,false),0);
    }
    @Test public void generalizedBetaTransformsRetainFiniteExtremeTails() {
        assertEquals(1e100-1,BetaPrime.quantile(1e-100,1,1,false,false),3e86);
        assertEquals(-Math.log1p(1e100),GeneralizedBetaSecondKind.cumulative(1e100,1,1,1,1,false,true),2e-13);
        assertEquals(-2*Math.log(1e100),GeneralizedBetaSecondKind.density(1e100,1,2,1,1,true)-Math.log(2)+Math.log(1e100),5e-13);
        assertEquals(-1000,GeneralizedBetaSecondKind.cumulative(Math.exp(500),1,2,1,1,false,true),3e-13);
        assertEquals(-Math.log1p(1e100),FellerPareto.cumulative(1e100,0,1,1,1,1,false,true),2e-13);
        assertEquals(1e100-1,FellerPareto.quantile(1e-100,0,1,1,1,1,false,false),3e86);
    }
    @Test public void truncatedNormalRetainsMassFarBeyondOrdinaryCdfPrecision() {
        Normal normal=new Normal(0,1);
        for(double lo:new double[]{10,40,-41}) {
            TruncatedContinuousDistribution d=new TruncatedContinuousDistribution(normal,lo,lo+1);
            assertTrue(Double.isFinite(d.getLogRetainedProbability()));
            double median=d.quantile(.5);
            assertTrue(median>lo && median<lo+1);
            assertEquals(.5,d.cumulative(median),1e-12);
            assertEquals(d.density(median,true),Math.log(d.density(median,false)),2e-13);
        }
    }
    @Test public void mixtureAndCensoringKeepLogTailsAndInfiniteDensities() {
        Normal normal=new Normal(0,1);
        MixtureDistribution d=new MixtureDistribution(new double[]{.3,.7},normal,normal);
        assertEquals(normal.cumulative(40,false,true),d.cumulative(40,false,true),2e-13);
        for(boolean lower:new boolean[]{true,false}) assertEquals(normal.quantile(-1000,lower,true),d.quantile(-1000,lower,true),2e-13);
        MixtureDistribution singular=new MixtureDistribution(new double[]{1,1},new Beta(.5,1),new Beta(.5,1));
        assertEquals(Double.POSITIVE_INFINITY,singular.density(0,true),0);
        CensoredDistribution c=new CensoredDistribution(normal,-50,50);
        assertEquals(normal.cumulative(40,false,true),c.cumulative(40,false,true),0);
        assertEquals(normal.density(40,true),c.density(40,true),0);
        assertTrue(Double.isFinite(c.density(50,true)));
    }
}
