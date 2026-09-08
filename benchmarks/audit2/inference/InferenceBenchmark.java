import java.util.*;
import jdistlib.inference.*;
/** Deterministic diagnostic timing; includes summaries and four ESS calculations. */
public class InferenceBenchmark {
 private static volatile double sink;
 public static void main(String[] args) {
  int n=args.length==0?16384:Integer.parseInt(args[0]);ChainResult[] chains=new ChainResult[4];
  for(int c=0;c<4;c++) {
   double[][] x=new double[n][1];IterationStats[] stats=new IterationStats[n];
   for(int i=0;i<n;i++){x[i][0]=Math.sin((i+1)*(.003*4096/n)+c)+.05*Math.cos((i+1)*.13+c);stats[i]=new IterationStats(true,1,1,Double.NaN,0,false,0,0);}
   chains[c]=new ChainResult(x,new double[n],stats,null,null,ChainResult.Status.SUCCESS,Collections.<String>emptyList());
  }
  for(int i=0;i<2;i++)sink=McmcDiagnostics.analyze(chains).parameter("state[0]").bulkEffectiveSampleSize();
  double[] ms=new double[5];
  for(int i=0;i<ms.length;i++){long start=System.nanoTime();sink=McmcDiagnostics.analyze(chains).parameter("state[0]").bulkEffectiveSampleSize();ms[i]=(System.nanoTime()-start)/1e6;}
  Arrays.sort(ms);System.out.printf(Locale.ROOT,"diagnostics_4x%d\t%.6f\t%.17g%n",n,ms[2],sink);
 }
}
