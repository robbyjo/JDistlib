/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.math;

/** Immutable double-precision complex value used by Java integrations. */
public final class Complex {
	public static final Complex ZERO = new Complex(0,0);
	public static final Complex ONE = new Complex(1,0);
	public static final Complex I = new Complex(0,1);
	private final double real, imaginary;
	public Complex(double real,double imaginary){this.real=real;this.imaginary=imaginary;}
	public double real(){return real;}
	public double imaginary(){return imaginary;}
	public double abs(){return Math.hypot(real,imaginary);}
	public double norm(){return real*real+imaginary*imaginary;}
	public double arg(){return Math.atan2(imaginary,real);}
	public Complex add(Complex other){return new Complex(real+other.real,imaginary+other.imaginary);}
	public Complex subtract(Complex other){return new Complex(real-other.real,imaginary-other.imaginary);}
	public Complex multiply(Complex other){return new Complex(real*other.real-imaginary*other.imaginary,real*other.imaginary+imaginary*other.real);}
	public Complex divide(Complex other){double d=other.norm();return new Complex((real*other.real+imaginary*other.imaginary)/d,(imaginary*other.real-real*other.imaginary)/d);}
	public Complex conjugate(){return new Complex(real,-imaginary);}
	public Complex exp(){double scale=Math.exp(real);return new Complex(scale*Math.cos(imaginary),scale*Math.sin(imaginary));}
	public Complex log(){return new Complex(Math.log(abs()),arg());}
	public Complex sqrt(){double magnitude=abs();return new Complex(Math.sqrt((magnitude+real)/2),Math.copySign(Math.sqrt(Math.max(0,(magnitude-real)/2)),imaginary));}
	public Complex sin(){return new Complex(Math.sin(real)*Math.cosh(imaginary),Math.cos(real)*Math.sinh(imaginary));}
	public Complex cos(){return new Complex(Math.cos(real)*Math.cosh(imaginary),-Math.sin(real)*Math.sinh(imaginary));}
	public Complex tan(){return sin().divide(cos());}
	public Complex sinh(){return new Complex(Math.sinh(real)*Math.cos(imaginary),Math.cosh(real)*Math.sin(imaginary));}
	public Complex cosh(){return new Complex(Math.cosh(real)*Math.cos(imaginary),Math.sinh(real)*Math.sin(imaginary));}
	public Complex tanh(){return sinh().divide(cosh());}
	public Complex pow(Complex exponent){return log().multiply(exponent).exp();}
	@Override public boolean equals(Object value){if(!(value instanceof Complex))return false;Complex other=(Complex)value;return Double.doubleToLongBits(real)==Double.doubleToLongBits(other.real)&&Double.doubleToLongBits(imaginary)==Double.doubleToLongBits(other.imaginary);}
	@Override public int hashCode(){long a=Double.doubleToLongBits(real),b=Double.doubleToLongBits(imaginary);return 31*(int)(a^(a>>>32))+(int)(b^(b>>>32));}
	@Override public String toString(){return real+(imaginary<0?"":"+")+imaginary+"i";}
}
