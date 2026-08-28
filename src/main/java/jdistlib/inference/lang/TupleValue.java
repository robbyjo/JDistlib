/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

/** Immutable heterogeneous tuple for Java data adapters and external functions. */
public final class TupleValue implements Iterable<Object> {
	private final Object[] values;
	public TupleValue(Object... values) {
		if(values==null||values.length==0)throw new IllegalArgumentException("tuple must contain at least one value");
		this.values=values.clone();for(Object value:this.values)if(value==null)throw new IllegalArgumentException("tuple members must not be null");
	}
	public int size(){return values.length;}
	/** Returns the one-based member used by Stan tuple syntax. */
	public Object member(int oneBasedIndex){if(oneBasedIndex<1||oneBasedIndex>values.length)throw new IndexOutOfBoundsException();return values[oneBasedIndex-1];}
	public TupleValue withMember(int oneBasedIndex,Object value){if(value==null)throw new IllegalArgumentException("tuple member must not be null");Object[] copy=values.clone();if(oneBasedIndex<1||oneBasedIndex>copy.length)throw new IndexOutOfBoundsException();copy[oneBasedIndex-1]=value;return new TupleValue(copy);}
	public Object[] toArray(){return values.clone();}
	@Override public Iterator<Object> iterator(){return new Iterator<Object>(){int index;@Override public boolean hasNext(){return index<values.length;}@Override public Object next(){if(!hasNext())throw new NoSuchElementException();return values[index++];}@Override public void remove(){throw new UnsupportedOperationException();}};}
	@Override public boolean equals(Object value){return value instanceof TupleValue&&Arrays.deepEquals(values,((TupleValue)value).values);}
	@Override public int hashCode(){return Arrays.deepHashCode(values);}
	@Override public String toString(){return Arrays.deepToString(values);}
}
