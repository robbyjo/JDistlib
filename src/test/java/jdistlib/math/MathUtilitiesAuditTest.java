package jdistlib.math;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import static org.junit.Assert.*;
import jdistlib.math.density.*;
import jdistlib.math.approx.ApproximationFunction;
import jdistlib.math.opt.Optimization;
import jdistlib.math.spline.*;
import jdistlib.disttest.NormalityTest;
import jdistlib.disttest.MultipleTesting;

/** R fixtures plus identities independent of its implementation. */
public class MathUtilitiesAuditTest {
    private static Map<String, Double> reference() throws Exception {
        Map<String, Double> values = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                MathUtilitiesAuditTest.class.getResourceAsStream("/audit2/math-reference.tsv"), StandardCharsets.UTF_8))) {
            reader.readLine(); String line;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split("\t");
                values.put(fields[0] + ":" + fields[1], fields[2].equals("NA") ? Double.NaN : Double.valueOf(fields[2]));
            }
        }
        return values;
    }
    public static double[] data(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = Math.sin((i+1)*.37)+Math.cos((i+1)*.11)*.3+(i+1.)/n;
        return x;
    }
    @Test public void densityKernelsAndWeightsMatchR() throws Exception {
        Map<String,Double> r = reference();
        double[] finite = {-2,-.9,-.3,0,.15,.4,.7,1.1,1.8,3};
        for (Kernel kernel : Kernel.values()) for (String mode : new String[]{"plain","weighted","infinite","weighted_infinite"}) {
            double[] x = finite, weights = null;
            if (mode.equals("infinite")) { x = java.util.Arrays.copyOf(finite,12); x[10]=Double.POSITIVE_INFINITY; x[11]=Double.NEGATIVE_INFINITY; }
            if (mode.equals("weighted")) { weights = new double[10]; for(int i=0;i<10;i++) weights[i]=(i+1.)/55; }
            if (mode.equals("weighted_infinite")) {
                x=java.util.Arrays.copyOf(finite,11); x[10]=Double.POSITIVE_INFINITY;
                weights=new double[11]; for(int i=0;i<10;i++) weights[i]=.75*(i+1.)/55; weights[10]=.25;
            }
            Density actual = Density.density(x,null,1,kernel,weights,.4*kernel.getFactor(),65,-4,5,3);
            for(int i=0;i<65;i++) {
                String key="density:"+kernel.name().toLowerCase(java.util.Locale.ROOT)+":"+mode+":"+i;
                assertEquals(key,r.get(key),actual.y[i],2e-14);
            }
        }
    }
    @Test public void bandwidthsMatchRIncludingLargePairCounts() throws Exception {
        Map<String,Double> r=reference();
        Bandwidth[] methods={Bandwidth.NRD0,Bandwidth.NRD,Bandwidth.UCV,Bandwidth.BCV,Bandwidth.SJ_STE,Bandwidth.SJ_DPI};
        String[] names={"nrd0","nrd","ucv","bcv","SJ-ste","SJ-dpi"};
        for(int n:new int[]{20,1000,10000}) for(int i=0;i<methods.length;i++) {
            String key="bandwidth:"+n+":"+names[i]+":0";
            assertEquals(key,r.get(key),methods[i].calculate(data(n)),1e-10);
        }
        double[] repeated=new double[100000]; for(int i=0;i<repeated.length;i++) repeated[i]=i%2;
        assertEquals(r.get("bandwidth:counts:bcv:0"),Bandwidth.BCV.calculate(repeated),1e-10);
    }
    @Test public void stableMomentsRetainOffsetsAndExtremeScales() throws Exception {
        Map<String,Double> r=reference(); double[] x={1e12,1e12+1,1e12+2,1e12+3,1e12+4};
        assertEquals(r.get("summary:offset:0"),VectorMath.mean(x),0);
        assertEquals(r.get("summary:offset:1"),VectorMath.var(x),1e-15);
        assertEquals(r.get("summary:offset:2"),VectorMath.sd(x),1e-15);
        assertEquals(1./3,VectorMath.mean(new double[]{1e300,1,-1e300}),1e-15);
        assertEquals(1e-100/3,VectorMath.mean(new double[]{1e300,1e-100,-1e300}),1e-115);
        assertEquals(2e-101,VectorMath.mean(new double[]{1e308,1e308,-1e308,-1e308,1e-100}),1e-115);
        assertEquals(Double.MIN_VALUE,VectorMath.mean(new double[]{Double.MIN_VALUE,Double.MIN_VALUE}),0);
        assertEquals(1e200,VectorMath.distance(new double[]{1e200,-1e200}),1e185);
        assertEquals(Math.sqrt(2)*1e200,VectorMath.sd(new double[]{1e200,-1e200}),2e185);
        assertEquals(Math.sqrt(2)*1e-200,VectorMath.sd(new double[]{1e-200,-1e-200}),2e-215);
        assertTrue(Double.isNaN(VectorMath.mean(new double[0])));
        assertTrue(Double.isNaN(VectorMath.var(new double[]{1})));
    }
    @Test public void normalityMatchesR() throws Exception {
        Map<String,Double> r=reference();
        for(int n:new int[]{8,20,100,2000}) {
            double statistic=NormalityTest.shapiro_wilk_statistic(data(n),true);
            assertEquals(r.get("shapiro:"+n+":0"),statistic,2e-13);
            assertEquals(r.get("shapiro:"+n+":1"),NormalityTest.shapiro_wilk_pvalue(statistic,n),2e-12);
        }
    }
    @Test public void interpolationUsesRightContinuousConstantConvention() {
        double[] x={0,1},y={2,4};
        assertEquals(2,ApproximationFunction.constant(.4,x,y,-1,5,0),0);
        assertEquals(4,ApproximationFunction.constant(.4,x,y,-1,5,1),0);
        assertEquals(2.5,ApproximationFunction.constant(.4,x,y,-1,5,.25),0);
        assertTrue(Double.isNaN(ApproximationFunction.constant(Double.NaN,x,y,-1,5,0)));
    }
    @Test public void remainingNormalityTestsMatchRPackages() throws Exception {
        Map<String,Double> r=reference();
        for(String mode:new String[]{"ordinary","offset","outlier","skew"}) {
            double[] x=new double[mode.equals("outlier")?1000:100];
            for(int i=0;i<x.length;i++) x[i]=mode.equals("offset")?1e12+i+1:mode.equals("outlier")?(i==999?1:0):mode.equals("skew")?Math.exp(i*5./99):Math.sin((i+1)*.37)+(i+1.)/100;
            double ad=NormalityTest.anderson_darling_statistic(x), cvm=NormalityTest.cramer_vonmises_statistic(x), lillie=NormalityTest.kolmogorov_lilliefors_statistic(x);
            double jb=NormalityTest.jarque_bera_statistic(x), dp=NormalityTest.dagostino_pearson_statistic(x);
            double[] actual={ad,NormalityTest.anderson_darling_pvalue(ad,x.length),cvm,NormalityTest.cramer_vonmises_pvalue(cvm,x.length),lillie,NormalityTest.kolmogorov_lilliefors_pvalue(lillie,x.length),jb,NormalityTest.jarque_bera_pvalue(jb),dp,NormalityTest.dagostino_pearson_pvalue(dp)};
            for(int i=0;i<actual.length;i++) {
                double expected=r.get("normality:"+mode+":"+i);
                // R moments::jarque.test subtracts its CDF from one; check tiny
                // p-values against the exact chi-square(2) survival identity.
                if(i==7) expected=Math.exp(-jb/2);
                assertEquals(mode+":"+i,expected,actual[i],2e-10*Math.max(1,Math.abs(expected)));
            }
        }
        assertEquals(1,NormalityTest.jarque_bera_pvalue(0),0);
        assertTrue(Double.isNaN(NormalityTest.kolmogorov_lilliefors_statistic(new double[]{1,1,1})));
        assertEquals(Math.exp(-50),NormalityTest.dagostino_pearson_pvalue(100),1e-35);
        double[] symmetric={-4,-3,-2,-1,1,2,3,4};
        assertTrue(Double.isFinite(NormalityTest.dagostino_pearson_statistic(symmetric)));
        for(int negative:new int[]{1,5}) {
            double[] unit=new double[10], huge=new double[10];
            for(int i=0;i<10;i++){unit[i]=i<negative?-1:1;huge[i]=unit[i]*Double.MAX_VALUE;}
            assertEquals(NormalityTest.jarque_bera_statistic(unit),NormalityTest.jarque_bera_statistic(huge),1e-12);
            assertEquals(NormalityTest.dagostino_pearson_statistic(unit),NormalityTest.dagostino_pearson_statistic(huge),1e-12);
            assertEquals(NormalityTest.anderson_darling_statistic(unit),NormalityTest.anderson_darling_statistic(huge),1e-12);
            assertEquals(NormalityTest.cramer_vonmises_statistic(unit),NormalityTest.cramer_vonmises_statistic(huge),1e-12);
            assertEquals(NormalityTest.kolmogorov_lilliefors_statistic(unit),NormalityTest.kolmogorov_lilliefors_statistic(huge),1e-12);
        }
    }
    @Test public void multipleTestingMatchesRWithMissingValues() throws Exception {
        Map<String,Double> r=reference();
        double[] p={.001,.003,.01,.02,.05,.15,.7,.99,Double.NaN,.4};
        String[] names={"bonferroni","holm","hochberg","hommel","BH","BY","none"};
        MultipleTesting.Method[] methods={MultipleTesting.Method.BONFERRONI,MultipleTesting.Method.HOLM,MultipleTesting.Method.HOCHBERG,MultipleTesting.Method.HOMMEL,MultipleTesting.Method.BENJAMINI_HOCHBERG,MultipleTesting.Method.BENJAMINI_YEKUTIELI,MultipleTesting.Method.NONE};
        for(int m=0;m<methods.length;m++) {
            double[] actual=MultipleTesting.adjust(p,methods[m]);
            for(int i=0;i<p.length;i++) assertEquals(names[m]+":"+i,r.get("adjust:"+names[m]+":"+i),actual[i],2e-15);
        }
    }
    @Test public void weightedSmoothingPreservesInputsAndPermutation() {
        double[] x=new double[30],y=new double[30],w=new double[30];
        for(int i=0;i<30;i++){x[i]=(i/2)/14.;y[i]=Math.sin(x[i]*6)+i*.01;w[i]=i+1;}
        double[] original=w.clone(),rx=new double[30],ry=new double[30],rw=new double[30];
        for(int i=0;i<30;i++){rx[i]=x[29-i];ry[i]=y[29-i];rw[i]=w[29-i];}
        for(SmoothSplineCriterion criterion:new SmoothSplineCriterion[]{SmoothSplineCriterion.CV,SmoothSplineCriterion.GCV}) {
            SmoothSplineResult a=SmoothSpline.fit(x,y,w,criterion,1,0,.5);
            SmoothSplineResult b=SmoothSpline.fit(rx,ry,rw,criterion,1,0,.5);
            assertArrayEquals(original,w,0);
            assertEquals(a.mCVScore,b.mCVScore,1e-12);
            for(int i=0;i<10;i++) assertEquals(SmoothSpline.predict(a,i/9.,0),SmoothSpline.predict(b,i/9.,0),1e-12);
        }
    }
    @Test public void smoothingAndScalarOptimizationMatchReferences() throws Exception {
        Map<String,Double> r=reference(); double[] x=new double[30],y=new double[30];
        for(int i=0;i<30;i++) { x[i]=i/29.; y[i]=Math.sin(x[i]*6)+x[i]*x[i]; }
        for(double spar:new double[]{.3,.7}) {
            SmoothSplineResult fit=SmoothSpline.fit(x,y,null,SmoothSplineCriterion.NO_CRITERION,1,0,spar);
            for(int i=0;i<25;i++) assertEquals("spline "+spar+" "+i,r.get("spline:"+spar+":"+i),
                SmoothSpline.predict(fit,-.1+1.2*i/24,0),2e-9);
        }
        UnivariateFunction root=new UnivariateFunction() { public double eval(double z){return Math.cos(z)-z;} public void setParameters(double...p){} public void setObjects(Object...p){} };
        UnivariateFunction minimum=new UnivariateFunction() { public double eval(double z){return (z-.37)*(z-.37)+Math.exp(z)/10;} public void setParameters(double...p){} public void setObjects(Object...p){} };
        assertEquals(r.get("opt:root:0"),Optimization.zeroin(root,0,1,1e-12,1000),1e-12);
        assertEquals(r.get("opt:minimum:0"),Optimization.optimize(minimum,-1,2,1e-12,1000),1e-8);
    }
}
