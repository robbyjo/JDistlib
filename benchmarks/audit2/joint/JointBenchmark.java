import jdistlib.*;
import jdistlib.finance.*;
import java.util.*;
import java.util.function.IntToDoubleFunction;

/** Five-run median; same harness can run with baseline or candidate classes. */
public final class JointBenchmark {
  private static volatile double sink;
  public static void main(String[] args) {
    Locale.setDefault(Locale.ROOT);
    Copula joe=new JoeCopula(2.7),bb1=new BB1Copula(1.3,2.1);
    double[][] points=new double[256][2];for(int i=0;i<256;i++){points[i][0]=.1+.8*i/256;points[i][1]=.1+.8*((i*19)%256)/256;}
    run("joe_density",20000,i->joe.logDensity(points[i%256]));
    run("bb1_density",20000,i->bb1.logDensity(points[i%256]));
    run("joe_tau",1000,i->joe.kendallsTau(0,1));
    PolyaAeppliDistribution polya=new PolyaAeppliDistribution(30,.4);
    DelaporteDistribution delaporte=new DelaporteDistribution(30,10,.4);
    run("polya_cdf",100,i->polya.cumulative(200+i%50,true,false));
    run("delaporte_cdf",100,i->delaporte.cumulative(200+i%50,true,false));
    run("polya_density",1000,i->polya.density(200+i%50,true));
    GaussianCopula gaussian=new GaussianCopula(new double[][]{{1,.7},{.7,1}});
    StudentTCopula student=new StudentTCopula(new double[][]{{1,.7},{.7,1}},5);
    run("gaussian_cdf",100,i->gaussian.cumulative(points[i%256]));
    run("student_cdf",100,i->student.cumulative(points[i%256]));
    OrderStatisticDistribution max=OrderStatisticDistribution.maximum(new Normal(),1000);
    run("normal_maximum_random",1000,i->max.random());
  }
  private static void run(String name,int n,IntToDoubleFunction f) {
    double[] times=new double[5];for(int pass=-2;pass<5;pass++){
      long start=System.nanoTime();double value=0;for(int i=0;i<n;i++)value+=f.applyAsDouble(i);sink=value;
      if(pass>=0)times[pass]=(System.nanoTime()-start)/1e6;
    }Arrays.sort(times);System.out.printf("%s,%d,%.6f%n",name,n,times[2]);
  }
}
