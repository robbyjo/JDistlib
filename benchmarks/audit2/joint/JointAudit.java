import java.nio.file.*;
import java.util.*;
import jdistlib.*;

/** Reproducible R comparisons; reports mismatches rather than trusting R blindly. */
public final class JointAudit {
  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.ROOT);
    System.out.println("family\ttheta\tu\tv\tmetric\tjava\tR\terror");
    List<String> rows=Files.readAllLines(Paths.get("build/audit2/joint/reference.tsv"));
    int total=0, mismatches=0;
    for(String row:rows.subList(1,rows.size())) {
      String[] a=row.split("\t"); double theta=Double.parseDouble(a[1]);
      Copula c;
      switch(a[0]) {
        case "clayton":c=new ClaytonCopula(2,theta);break;
        case "gumbel":c=new GumbelCopula(2,theta);break;
        case "frank":c=new FrankCopula(2,theta);break;
        case "joe":c=new JoeCopula(theta);break;
        case "gaussian":c=new GaussianCopula(new double[][]{{1,theta},{theta,1}});break;
        default:c=new StudentTCopula(new double[][]{{1,theta},{theta,1}},5);
      }
      double[] uv={Double.parseDouble(a[3]),Double.parseDouble(a[4])};
      double[] actual={c.cumulative(uv),c.logDensity(uv),c.kendallsTau(0,1)};
      for(int k=0;k<3;k++) {
        double ref=parse(a[5+k]),error=Math.abs(actual[k]-ref);total++;
        double tolerance=k==0?2e-7:k==1?1e-7:2e-10;
        if(!(error<=tolerance||actual[k]==ref)) {
          mismatches++;
          System.out.printf("%s\t%.15g\t%.15g\t%.15g\t%d\t%.17g\t%.17g\t%.6g%n",a[0],theta,uv[0],uv[1],k,actual[k],ref,error);
        }
      }
    }
    System.err.println("copula values="+total+" mismatches="+mismatches);
    rows=Files.readAllLines(Paths.get("build/audit2/joint/multivariate.tsv"));
    total=0;mismatches=0;
    for(String row:rows.subList(1,rows.size())) {
      String[] a=row.split("\t");int d=Integer.parseInt(a[0]);double df=Double.parseDouble(a[1]),x=Double.parseDouble(a[2]);
      double[][] s=new double[d][d];double[] xx=new double[d],m=new double[d];Arrays.fill(xx,x);
      for(int i=0;i<d;i++){Arrays.fill(s[i],.3);s[i][i]=1;}
      double[] actual={MultivariateNormal.density(xx,m,s,true),MultivariateStudentT.density(xx,m,s,df,true)};
      for(int k=0;k<2;k++) {total++;double ref=parse(a[3+k]);if(Math.abs(actual[k]-ref)>1e-7){mismatches++;System.out.printf("mv%d\t%.15g\t%.15g\t0\t%d\t%.17g\t%.17g\t%.6g%n",d,df,x,k,actual[k],ref,Math.abs(actual[k]-ref));}}
    }
    System.err.println("multivariate values="+total+" mismatches="+mismatches);
  }
  private static double parse(String s) {return s.equals("Inf")?Double.POSITIVE_INFINITY:s.equals("-Inf")?Double.NEGATIVE_INFINITY:s.equals("NA")?Double.NaN:Double.parseDouble(s);}
}
