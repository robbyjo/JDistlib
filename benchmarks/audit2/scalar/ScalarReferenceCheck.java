import java.nio.file.*;
import java.util.*;
import java.lang.reflect.*;

/** Reads the tab-separated fixtures produced by reference.R. */
public class ScalarReferenceCheck {
    static double number(String s) { return s.equals("Inf") ? Double.POSITIVE_INFINITY : s.equals("-Inf") ? Double.NEGATIVE_INFINITY : s.equals("NA") ? Double.NaN : Double.parseDouble(s); }
    public static void main(String[] args) throws Exception {
        List<String> lines=Files.readAllLines(Paths.get(args[0]));
        Map<String,int[]> counts=new TreeMap<>(); List<String> failures=new ArrayList<>();
        for(String line:lines.subList(1,lines.size())) {
            String[] s=line.split("\t"); String name=s[0], op=s[1];
            String[] params=s[3].split(";"); boolean lower=Boolean.parseBoolean(s[4]), log=Boolean.parseBoolean(s[5]);
            Class<?>[] types=new Class<?>[params.length+(op.equals("density")?2:3)]; Object[] values=new Object[types.length];
            types[0]=double.class;values[0]=number(s[2]);
            for(int i=0;i<params.length;i++){types[i+1]=double.class;values[i+1]=number(params[i]);}
            int index=params.length+1;
            if(!op.equals("density")){types[index]=boolean.class;values[index++]=lower;}
            types[index]=boolean.class;values[index]=log;
            double actual=(Double)Class.forName("jdistlib."+name).getMethod(op,types).invoke(null,values);
            double expected=number(s[6]);
            double tolerance=Double.parseDouble(s[7]);
            boolean ok=actual==expected || (Double.isNaN(actual)&&Double.isNaN(expected)) || (Double.isFinite(actual)&&Double.isFinite(expected)&&Math.abs(actual-expected)<=tolerance*Math.max(1e-280,Math.abs(expected)));
            int[] c=counts.computeIfAbsent(name,k->new int[2]);c[0]++;if(!ok){c[1]++;failures.add(line+"\t"+actual);}
        }
        for(Map.Entry<String,int[]> e:counts.entrySet())System.out.println(e.getKey()+"\t"+e.getValue()[0]+"\t"+e.getValue()[1]);
        Files.write(Paths.get(args[1]), failures);
        System.out.println("TOTAL\t"+(lines.size()-1)+"\t"+failures.size());
    }
}
