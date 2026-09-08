import jdistlib.*;
import jdistlib.rng.MersenneTwister;
/** Run in separate JVMs with baseline/candidate classpaths. Times exclude setup. */
public class ScalarSpeed {
    static volatile double sink;
    public static void main(String[] args) {
        int n=Integer.parseInt(args[0]);
        for(int kind=0;kind<3;kind++) {
            double best=Double.POSITIVE_INFINITY;
            for(int trial=0;trial<7;trial++) {
                MersenneTwister rng=new MersenneTwister(473);double sum=0;
                long start=System.nanoTime();
                for(int i=0;i<n;i++) {
                    double p=(i%997+.5)/997;
                    if(kind==0)sum+=InvNormal.random(2,.5,rng);
                    else if(kind==1)sum+=PositiveNormal.quantile(p,-1,2,true,false);
                    else sum+=BetaBinomial.cumulative(i%80,.3,.2,100,true,false);
                }
                double elapsed=(System.nanoTime()-start)/(double)n;sink=sum;
                if(trial>=2)best=Math.min(best,elapsed);
            }
            System.out.println(new String[]{"InvNormal.random","PositiveNormal.quantile","BetaBinomial.cumulative"}[kind]+"\t"+best+" ns/call");
        }
    }
}
