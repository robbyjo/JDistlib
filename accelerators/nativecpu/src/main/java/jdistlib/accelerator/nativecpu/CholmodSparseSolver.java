/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator.nativecpu;

import com.sun.jna.Function;
import com.sun.jna.IntegerType;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import jdistlib.accelerator.MatrixTriangle;
import jdistlib.accelerator.PreparedFloatSparseCholesky;
import jdistlib.accelerator.PreparedSparseCholesky;
import jdistlib.matrix.CsrMatrix;
import jdistlib.matrix.FloatCsrMatrix;

/** SuiteSparse CHOLMOD adapter used alongside the OpenBLAS dense provider. */
final class CholmodSparseSolver {
	private static final int COMMON_BYTES = 16384;
	private static final int CHOLMOD_A = 0, CHOLMOD_INT = 0, CHOLMOD_REAL = 1;
	private static final int CHOLMOD_DOUBLE = 0, CHOLMOD_SINGLE = 4;
	private CholmodSparseSolver() {}
	static PreparedSparseCholesky prepare(OpenBlasComputeBackend backend,CsrMatrix matrix,MatrixTriangle triangle){
		return new DoubleHandle(backend,CanonicalDouble.create(matrix,triangle));
	}
	static PreparedFloatSparseCholesky prepare(OpenBlasComputeBackend backend,FloatCsrMatrix matrix,MatrixTriangle triangle){
		return new FloatHandle(backend,CanonicalFloat.create(matrix,triangle));
	}

	private abstract static class Handle {
		final OpenBlasComputeBackend backend; final Memory common;
		final int dimension,structuralNonzeros,dtype; Pointer factor; boolean closed;
		int factorNonzeros;
		Handle(OpenBlasComputeBackend backend,int dimension,int structuralNonzeros,int dtype){
			this.backend=backend;this.dimension=dimension;this.structuralNonzeros=structuralNonzeros;this.dtype=dtype;
			common=new Memory(COMMON_BYTES);common.clear();int started=callInt("cholmod_start",common);
			if(started==0)throw new IllegalStateException("SuiteSparse CHOLMOD initialization failed");
		}
		final void analyze(NativeSparse matrix){
			factor=callPointer("cholmod_analyze",matrix.structure.getPointer(),common);
			if(factor==null)throw new IllegalStateException("SuiteSparse CHOLMOD symbolic analysis failed");
		}
		final void numeric(NativeSparse matrix){
			if(callInt("cholmod_factorize",matrix.structure.getPointer(),factor,common)==0)
				throw new ArithmeticException("SuiteSparse CHOLMOD sparse Cholesky factorization failed");
			if(callInt("cholmod_change_factor",CHOLMOD_REAL,1,0,1,1,factor,common)==0)
				throw new IllegalStateException("SuiteSparse CHOLMOD could not expose an LL' factor");
			updateMetadata();
		}
		final void updateMetadata(){
			Pointer copy=callPointer("cholmod_copy_factor",factor,common);
			if(copy==null)throw new IllegalStateException("SuiteSparse CHOLMOD could not copy its factor");
			Pointer sparse=null;
			try{
				sparse=callPointer("cholmod_factor_to_sparse",copy,common);
				if(sparse==null)throw new IllegalStateException("SuiteSparse CHOLMOD could not export its factor");
				CholmodSparse view=new CholmodSparse(sparse);int[] starts=view.p.getIntArray(0,dimension+1);
				int[] rows=view.i.getIntArray(0,starts[dimension]);factorNonzeros=starts[dimension];
				updateLogDeterminant(view,starts,rows);
			}finally{
				if(sparse!=null){PointerByReference ref=new PointerByReference(sparse);callInt("cholmod_free_sparse",ref,common);}
				PointerByReference ref=new PointerByReference(copy);callInt("cholmod_free_factor",ref,common);
			}
		}
		abstract void updateLogDeterminant(CholmodSparse sparse,int[] starts,int[] rows);
		final Pointer solve(NativeDense right){
			Pointer result=callPointer("cholmod_solve",CHOLMOD_A,factor,right.structure.getPointer(),common);
			if(result==null)throw new IllegalStateException("SuiteSparse CHOLMOD sparse solve failed");return result;
		}
		final void freeDense(Pointer dense){PointerByReference ref=new PointerByReference(dense);callInt("cholmod_free_dense",ref,common);}
		final void freeFactor(Pointer value){if(value!=null){PointerByReference ref=new PointerByReference(value);callInt("cholmod_free_factor",ref,common);}}
		final int[] permutationValue(){checkOpen();return new int[0];}
		final void checkOpen(){if(closed)throw new IllegalStateException("SuiteSparse CHOLMOD sparse factor is closed");}
		final void closeNative(){if(closed)return;try{if(factor!=null){PointerByReference ref=new PointerByReference(factor);callInt("cholmod_free_factor",ref,common);factor=null;}}
			finally{callInt("cholmod_finish",common);closed=true;}}
		final int callInt(String name,Object...args){return backend.cholmod(name).invokeInt(args);}
		final Pointer callPointer(String name,Object...args){return (Pointer)backend.cholmod(name).invoke(Pointer.class,args);}
	}

	private static final class DoubleHandle extends Handle implements PreparedSparseCholesky {
		private CanonicalDouble matrix;private NativeSparse nativeMatrix;private double logDeterminant;
		DoubleHandle(OpenBlasComputeBackend backend,CanonicalDouble matrix){super(backend,matrix.dimension,matrix.values.length,CHOLMOD_DOUBLE);this.matrix=matrix;
			try{nativeMatrix=NativeSparse.create(matrix);analyze(nativeMatrix);numeric(nativeMatrix);}catch(RuntimeException error){closeNative();throw error;}}
		public int dimension(){checkOpen();return dimension;}public int structuralNonzeroCount(){checkOpen();return structuralNonzeros;}
		public int factorNonzeroCount(){checkOpen();return factorNonzeros;}public int[] permutation(){return permutationValue();}
		public double logDeterminant(){checkOpen();return logDeterminant;}
		public synchronized void refactor(CsrMatrix input){checkOpen();CanonicalDouble next=CanonicalDouble.create(input,matrix.triangle);matrix.requireStructure(next);
			Pointer previousFactor=callPointer("cholmod_copy_factor",factor,common);if(previousFactor==null)throw new IllegalStateException("SuiteSparse CHOLMOD could not preserve the prior factor");
			CanonicalDouble previous=matrix;double previousLog=logDeterminant;int previousNonzeros=factorNonzeros;
			try{nativeMatrix.write(next.values);numeric(nativeMatrix);matrix=next;freeFactor(previousFactor);}
			catch(RuntimeException failure){freeFactor(factor);factor=previousFactor;nativeMatrix.write(previous.values);matrix=previous;logDeterminant=previousLog;factorNonzeros=previousNonzeros;throw failure;}}
		public synchronized void solveInPlace(double[]right,int columns){checkOpen();checkRight(right,columns,dimension);NativeDense dense=NativeDense.create(right,dimension,columns);
			Pointer result=solve(dense);try{CholmodDense view=new CholmodDense(result);double[]columnMajor=view.x.getDoubleArray(0,right.length);fromColumnMajor(columnMajor,right,dimension,columns);}finally{freeDense(result);}}
		public synchronized void close(){closeNative();nativeMatrix=null;matrix=null;}
		void updateLogDeterminant(CholmodSparse sparse,int[]starts,int[]rows){double[]values=sparse.x.getDoubleArray(0,factorNonzeros);double sum=0;
			for(int column=0;column<dimension;column++){double diagonal=Double.NaN;for(int at=starts[column];at<starts[column+1];at++)if(rows[at]==column){diagonal=values[at];break;}
				if(!(diagonal>0)||!Double.isFinite(diagonal))throw new ArithmeticException("SuiteSparse CHOLMOD returned an invalid Cholesky diagonal");sum+=2*Math.log(diagonal);}logDeterminant=sum;}
	}
	private static final class FloatHandle extends Handle implements PreparedFloatSparseCholesky {
		private CanonicalFloat matrix;private NativeSparse nativeMatrix;private float logDeterminant;
		FloatHandle(OpenBlasComputeBackend backend,CanonicalFloat matrix){super(backend,matrix.dimension,matrix.values.length,CHOLMOD_SINGLE);this.matrix=matrix;
			try{nativeMatrix=NativeSparse.create(matrix);analyze(nativeMatrix);numeric(nativeMatrix);}catch(RuntimeException error){closeNative();throw error;}}
		public int dimension(){checkOpen();return dimension;}public int structuralNonzeroCount(){checkOpen();return structuralNonzeros;}
		public int factorNonzeroCount(){checkOpen();return factorNonzeros;}public int[] permutation(){return permutationValue();}
		public float logDeterminant(){checkOpen();return logDeterminant;}
		public synchronized void refactor(FloatCsrMatrix input){checkOpen();CanonicalFloat next=CanonicalFloat.create(input,matrix.triangle);matrix.requireStructure(next);
			Pointer previousFactor=callPointer("cholmod_copy_factor",factor,common);if(previousFactor==null)throw new IllegalStateException("SuiteSparse CHOLMOD could not preserve the prior FP32 factor");
			CanonicalFloat previous=matrix;float previousLog=logDeterminant;int previousNonzeros=factorNonzeros;
			try{nativeMatrix.write(next.values);numeric(nativeMatrix);matrix=next;freeFactor(previousFactor);}
			catch(RuntimeException failure){freeFactor(factor);factor=previousFactor;nativeMatrix.write(previous.values);matrix=previous;logDeterminant=previousLog;factorNonzeros=previousNonzeros;throw failure;}}
		public synchronized void solveInPlace(float[]right,int columns){checkOpen();checkRight(right,columns,dimension);NativeDense dense=NativeDense.create(right,dimension,columns);
			Pointer result=solve(dense);try{CholmodDense view=new CholmodDense(result);float[]columnMajor=view.x.getFloatArray(0,right.length);fromColumnMajor(columnMajor,right,dimension,columns);}finally{freeDense(result);}}
		public synchronized void close(){closeNative();nativeMatrix=null;matrix=null;}
		void updateLogDeterminant(CholmodSparse sparse,int[]starts,int[]rows){float[]values=sparse.x.getFloatArray(0,factorNonzeros);float sum=0;
			for(int column=0;column<dimension;column++){float diagonal=Float.NaN;for(int at=starts[column];at<starts[column+1];at++)if(rows[at]==column){diagonal=values[at];break;}
				if(!(diagonal>0)||!Float.isFinite(diagonal))throw new ArithmeticException("SuiteSparse CHOLMOD returned an invalid FP32 Cholesky diagonal");sum+=2*(float)Math.log(diagonal);}logDeterminant=sum;}
	}

	private static final class NativeSparse {
		final CholmodSparse structure;final Memory starts,indices,values;final int dtype,count;
		NativeSparse(CholmodSparse structure,Memory starts,Memory indices,Memory values,int dtype,int count){this.structure=structure;this.starts=starts;this.indices=indices;this.values=values;this.dtype=dtype;this.count=count;}
		static NativeSparse create(CanonicalDouble matrix){return create(matrix.dimension,matrix.rowStarts,matrix.columns,matrix.values,null,CHOLMOD_DOUBLE);}
		static NativeSparse create(CanonicalFloat matrix){return create(matrix.dimension,matrix.rowStarts,matrix.columns,null,matrix.values,CHOLMOD_SINGLE);}
		static NativeSparse create(int n,int[]rowStarts,int[]columns,double[]doubles,float[]floats,int dtype){int count=columns.length;
			Memory p=new Memory((long)(n+1)*4),i=new Memory(Math.max(1L,(long)count*4));p.write(0,rowStarts,0,rowStarts.length);if(count>0)i.write(0,columns,0,count);
			int bytes=dtype==CHOLMOD_DOUBLE?8:4;Memory x=new Memory(Math.max(1L,(long)count*bytes));if(doubles!=null&&count>0)x.write(0,doubles,0,count);if(floats!=null&&count>0)x.write(0,floats,0,count);
			CholmodSparse s=new CholmodSparse();s.nrow=new SizeT(n);s.ncol=new SizeT(n);s.nzmax=new SizeT(count);s.p=p;s.i=i;s.nz=null;s.x=x;s.z=null;
			s.stype=1;s.itype=CHOLMOD_INT;s.xtype=CHOLMOD_REAL;s.dtype=dtype;s.sorted=1;s.packed=1;s.write();return new NativeSparse(s,p,i,x,dtype,count);}
		void write(double[]source){if(dtype!=CHOLMOD_DOUBLE||source.length!=count)throw new IllegalArgumentException("invalid CHOLMOD values");if(count>0)values.write(0,source,0,count);}
		void write(float[]source){if(dtype!=CHOLMOD_SINGLE||source.length!=count)throw new IllegalArgumentException("invalid CHOLMOD FP32 values");if(count>0)values.write(0,source,0,count);}
	}
	private static final class NativeDense {
		final CholmodDense structure;final Memory values;
		NativeDense(CholmodDense structure,Memory values){this.structure=structure;this.values=values;}
		static NativeDense create(double[]right,int rows,int columns){double[]nativeValues=toColumnMajor(right,rows,columns);Memory values=new Memory((long)nativeValues.length*8);values.write(0,nativeValues,0,nativeValues.length);return create(rows,columns,values,CHOLMOD_DOUBLE);}
		static NativeDense create(float[]right,int rows,int columns){float[]nativeValues=toColumnMajor(right,rows,columns);Memory values=new Memory((long)nativeValues.length*4);values.write(0,nativeValues,0,nativeValues.length);return create(rows,columns,values,CHOLMOD_SINGLE);}
		static NativeDense create(int rows,int columns,Memory values,int dtype){CholmodDense d=new CholmodDense();d.nrow=new SizeT(rows);d.ncol=new SizeT(columns);d.nzmax=new SizeT((long)rows*columns);d.d=new SizeT(rows);d.x=values;d.z=null;d.xtype=CHOLMOD_REAL;d.dtype=dtype;d.write();return new NativeDense(d,values);}
	}

	public static final class SizeT extends IntegerType {private static final long serialVersionUID=1L;public SizeT(){this(0);}public SizeT(long value){super(Native.SIZE_T_SIZE,value,true);}}
	public static final class CholmodSparse extends Structure {
		public SizeT nrow,ncol,nzmax;public Pointer p,i,nz,x,z;public int stype,itype,xtype,dtype,sorted,packed;
		public CholmodSparse(){}CholmodSparse(Pointer pointer){super(pointer);read();}
		protected List<String> getFieldOrder(){return Arrays.asList("nrow","ncol","nzmax","p","i","nz","x","z","stype","itype","xtype","dtype","sorted","packed");}
	}
	public static final class CholmodDense extends Structure {
		public SizeT nrow,ncol,nzmax,d;public Pointer x,z;public int xtype,dtype;
		public CholmodDense(){}CholmodDense(Pointer pointer){super(pointer);read();}
		protected List<String> getFieldOrder(){return Arrays.asList("nrow","ncol","nzmax","d","x","z","xtype","dtype");}
	}

	private static final class CanonicalDouble {
		final int dimension;final MatrixTriangle triangle;final double[]values;final int[]columns,rowStarts;
		CanonicalDouble(int dimension,MatrixTriangle triangle,double[]values,int[]columns,int[]rowStarts){this.dimension=dimension;this.triangle=triangle;this.values=values;this.columns=columns;this.rowStarts=rowStarts;}
		static CanonicalDouble create(CsrMatrix matrix,MatrixTriangle triangle){check(matrix,triangle);int n=matrix.rows();TreeMap<Integer,Double>[]rows=doubleRows(n);double[]input=matrix.values();int[]ci=matrix.columnIndices(),rs=matrix.rowStarts();
			for(int row=0;row<n;row++)for(int at=rs[row]-1;at<rs[row+1]-1;at++){int column=ci[at]-1;double value=input[at];if(!Double.isFinite(value))throw new IllegalArgumentException("sparse matrix must be finite");if((triangle==MatrixTriangle.LOWER&&column<=row)||(triangle==MatrixTriangle.UPPER&&column>=row)){int r=Math.max(row,column),c=Math.min(row,column);Double old=rows[r].get(c);rows[r].put(c,old==null?value:old+value);}}
			int count=count(rows),offset=0;double[]values=new double[count];int[]columns=new int[count],starts=new int[n+1];for(int row=0;row<n;row++){starts[row]=offset;for(Map.Entry<Integer,Double>entry:rows[row].entrySet()){columns[offset]=entry.getKey();values[offset++]=entry.getValue();}}starts[n]=offset;return new CanonicalDouble(n,triangle,values,columns,starts);}
		void requireStructure(CanonicalDouble other){if(dimension!=other.dimension||!Arrays.equals(columns,other.columns)||!Arrays.equals(rowStarts,other.rowStarts))throw new IllegalArgumentException("sparse refactorization structure differs from SuiteSparse analysis");}
	}
	private static final class CanonicalFloat {
		final int dimension;final MatrixTriangle triangle;final float[]values;final int[]columns,rowStarts;
		CanonicalFloat(int dimension,MatrixTriangle triangle,float[]values,int[]columns,int[]rowStarts){this.dimension=dimension;this.triangle=triangle;this.values=values;this.columns=columns;this.rowStarts=rowStarts;}
		static CanonicalFloat create(FloatCsrMatrix matrix,MatrixTriangle triangle){check(matrix,triangle);int n=matrix.rows();TreeMap<Integer,Float>[]rows=floatRows(n);float[]input=matrix.values();int[]ci=matrix.columnIndices(),rs=matrix.rowStarts();
			for(int row=0;row<n;row++)for(int at=rs[row]-1;at<rs[row+1]-1;at++){int column=ci[at]-1;float value=input[at];if(!Float.isFinite(value))throw new IllegalArgumentException("FP32 sparse matrix must be finite");if((triangle==MatrixTriangle.LOWER&&column<=row)||(triangle==MatrixTriangle.UPPER&&column>=row)){int r=Math.max(row,column),c=Math.min(row,column);Float old=rows[r].get(c);rows[r].put(c,old==null?value:old+value);}}
			int count=count(rows),offset=0;float[]values=new float[count];int[]columns=new int[count],starts=new int[n+1];for(int row=0;row<n;row++){starts[row]=offset;for(Map.Entry<Integer,Float>entry:rows[row].entrySet()){columns[offset]=entry.getKey();values[offset++]=entry.getValue();}}starts[n]=offset;return new CanonicalFloat(n,triangle,values,columns,starts);}
		void requireStructure(CanonicalFloat other){if(dimension!=other.dimension||!Arrays.equals(columns,other.columns)||!Arrays.equals(rowStarts,other.rowStarts))throw new IllegalArgumentException("FP32 sparse refactorization structure differs from SuiteSparse analysis");}
	}
	private static void check(CsrMatrix matrix,MatrixTriangle triangle){if(matrix==null||triangle==null||matrix.rows()<1||matrix.rows()!=matrix.columns())throw new IllegalArgumentException("SuiteSparse sparse Cholesky requires a square matrix and triangle");}
	private static void check(FloatCsrMatrix matrix,MatrixTriangle triangle){if(matrix==null||triangle==null||matrix.rows()<1||matrix.rows()!=matrix.columns())throw new IllegalArgumentException("SuiteSparse FP32 sparse Cholesky requires a square matrix and triangle");}
	@SuppressWarnings("unchecked")private static TreeMap<Integer,Double>[]doubleRows(int n){TreeMap<Integer,Double>[]result=(TreeMap<Integer,Double>[])new TreeMap<?,?>[n];for(int i=0;i<n;i++)result[i]=new TreeMap<Integer,Double>();return result;}
	@SuppressWarnings("unchecked")private static TreeMap<Integer,Float>[]floatRows(int n){TreeMap<Integer,Float>[]result=(TreeMap<Integer,Float>[])new TreeMap<?,?>[n];for(int i=0;i<n;i++)result[i]=new TreeMap<Integer,Float>();return result;}
	private static int count(Map<?,?>[]rows){int result=0;for(Map<?,?>row:rows)result+=row.size();return result;}
	private static void checkRight(double[]right,int columns,int dimension){if(columns<1||right==null||right.length!=dimension*columns)throw new IllegalArgumentException("invalid SuiteSparse sparse right side");}
	private static void checkRight(float[]right,int columns,int dimension){if(columns<1||right==null||right.length!=dimension*columns)throw new IllegalArgumentException("invalid SuiteSparse FP32 sparse right side");}
	private static double[]toColumnMajor(double[]input,int rows,int columns){double[]result=new double[input.length];for(int row=0;row<rows;row++)for(int column=0;column<columns;column++)result[column*rows+row]=input[row*columns+column];return result;}
	private static float[]toColumnMajor(float[]input,int rows,int columns){float[]result=new float[input.length];for(int row=0;row<rows;row++)for(int column=0;column<columns;column++)result[column*rows+row]=input[row*columns+column];return result;}
	private static void fromColumnMajor(double[]input,double[]result,int rows,int columns){for(int row=0;row<rows;row++)for(int column=0;column<columns;column++)result[row*columns+column]=input[column*rows+row];}
	private static void fromColumnMajor(float[]input,float[]result,int rows,int columns){for(int row=0;row<rows;row++)for(int column=0;column<columns;column++)result[row*columns+column]=input[column*rows+row];}
}
