import java.nio.file.*;
import java.util.*;
import jdistlib.*;
import jdistlib.finance.*;
import jdistlib.generic.GenericDistribution;

public final class FinanceAudit {
  public static void main(String[] args)throws Exception {
    Locale.setDefault(Locale.ROOT);int total=0,bad=0;
    List<String> rows=Files.readAllLines(Paths.get("build/audit2/joint/finance.tsv"));
    for(String row:rows.subList(1,rows.size())) {
      String[] a=row.split("\t");double lambda=parse(a[1]),shape=parse(a[2]),p=parse(a[3]),x=parse(a[4]);
      GenericDistribution d=a[0].equals("polya")?new PolyaAeppliDistribution(lambda,p):new DelaporteDistribution(lambda,shape,p);
      double[] values={d.density(x,true),d.cumulative(x,true,true),d.cumulative(x,false,true)};
      for(int k=0;k<3;k++){total++;double ref=parse(a[5+k]);if(!(values[k]==ref||Math.abs(values[k]-ref)<2e-8)){bad++;System.out.println(row+"\tmetric="+k+" Java="+values[k]);}}
    }
    System.out.println("finance values="+total+" mismatches="+bad);
    total=0;bad=0;rows=Files.readAllLines(Paths.get("build/audit2/joint/bb1.tsv"));
    for(String row:rows.subList(1,rows.size())) {
      String[] a=row.split("\t");BB1Copula d=new BB1Copula(parse(a[0]),parse(a[1]));double[] uv={parse(a[2]),parse(a[3])};
      double[] values={d.cumulative(uv),d.logDensity(uv),new PairCopula(d).conditionalSecondGivenFirst(uv[0],uv[1])};
      for(int k=0;k<3;k++){total++;double ref=parse(a[4+k]);if(!(values[k]==ref||Math.abs(values[k]-ref)<2e-7)){bad++;System.out.println(row+"\tmetric="+k+" Java="+values[k]);}}
    }
    System.out.println("BB1 values="+total+" mismatches="+bad);
    total=0;bad=0;rows=Files.readAllLines(Paths.get("build/audit2/joint/dirichlet.tsv"));
    for(String row:rows.subList(1,rows.size())) {
      String[] a=row.split("\t");double value=Dirichlet.density(new double[]{parse(a[3]),parse(a[4]),parse(a[5])},new double[]{parse(a[0]),parse(a[1]),parse(a[2])},true);total++;
      if(Math.abs(value-parse(a[6]))>1e-10){bad++;System.out.println(row+" Java="+value);}
    }
    System.out.println("Dirichlet values="+total+" mismatches="+bad);
  }
  private static double parse(String s) {return s.equals("Inf")?Double.POSITIVE_INFINITY:s.equals("-Inf")?Double.NEGATIVE_INFINITY:s.equals("NA")?Double.NaN:Double.parseDouble(s);}
}
