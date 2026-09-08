import java.util.Arrays;
import jdistlib.math.density.Bandwidth;
import jdistlib.math.density.Density;
import jdistlib.math.density.Kernel;

/** Run separately with pristine and candidate classes first on the classpath. */
public final class MathBenchmark {
    private static volatile double sink;
    private static double[] data(int n) {
        double[] x = new double[n];
        for (int i=0;i<n;i++) x[i]=Math.sin((i+1)*.37)+.3*Math.cos((i+1)*.11)+(i+1.)/n;
        return x;
    }
    private static void measure(String name, int calls, Runnable work) {
        for(int i=0;i<3;i++) work.run();
        double[] times=new double[5];
        for(int r=0;r<5;r++) {
            long start=System.nanoTime();
            for(int j=0;j<calls;j++)work.run();
            times[r]=(System.nanoTime()-start)*1e-6/calls;
        }
        Arrays.sort(times);
        System.out.println(name+"\t"+times[2]+"\t"+sink);
    }
    public static void main(String[] args) {
        final double[] x=data(20000);
        measure("SJ_DPI_20000",5,()->sink=Bandwidth.SJ_DPI.calculate(x));
        measure("UCV_20000",5,()->sink=Bandwidth.UCV.calculate(x));
        measure("Density_Gaussian_20000_512",200,()->sink=Density.density(x,null,1,Kernel.GAUSSIAN,null,.4*Kernel.GAUSSIAN.getFactor(),512,-4,5,3).y[256]);
    }
}
