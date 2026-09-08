package jdistlib;

import static org.junit.Assert.*;
import java.io.*;
import java.lang.reflect.Field;
import java.math.BigInteger;
import org.junit.Test;
import jdistlib.rng.*;

public class RandomEngineAudit2Test {
	private static RandomEngine[] engines(long seed) {
		return new RandomEngine[]{new RandomWELL44497b(seed),new RandomCMWC(seed),new MersenneTwister(seed),new MersenneTwisterSafe(seed)};
	}
	@Test public void wellMatchesIndependentCommonsMathAcrossStateWraps() {
		int[] state=new int[1391];for(int i=0;i<state.length;i++)state[i]=0x9e3779b9*(i+1)^Integer.rotateLeft(i,13);
		RandomWELL44497b actual=new RandomWELL44497b(state);
		org.apache.commons.math3.random.Well44497b reference=new org.apache.commons.math3.random.Well44497b(state);
		for(int i=0;i<20000;i++)assertEquals("WELL word "+i,reference.nextInt(),actual.nextInt());
	}
	@Test public void cmwcMatchesUnboundedIntegerRecurrenceAcrossStateWraps() throws Exception {
		RandomCMWC actual=new RandomCMWC(8675309);
		Field field=RandomCMWC.class.getDeclaredField("mBuffer");field.setAccessible(true);
		long[] state=((long[])field.get(actual)).clone();
		BigInteger modulus=BigInteger.ONE.shiftLeft(32), carry=BigInteger.valueOf(362436), multiplier=BigInteger.valueOf(18782);
		for(int i=0;i<16000;i++) {
			int index=i%4096;
			BigInteger product=BigInteger.valueOf(state[index]).multiply(multiplier).add(carry);
			BigInteger[] divided=product.divideAndRemainder(modulus);carry=divided[0];
			BigInteger sum=divided[1].add(carry);
			if(sum.compareTo(modulus)>=0){sum=sum.add(BigInteger.ONE).mod(modulus);carry=carry.add(BigInteger.ONE);}
			state[index]=modulus.subtract(BigInteger.valueOf(2)).subtract(sum).mod(modulus).longValue();
			assertEquals("CMWC word "+i,(int)state[index],actual.nextInt());
		}
	}
	@Test public void cloningReseedingAndSerializationPreserveMixedDrawStreams() throws Exception {
		for(RandomEngine original:engines(12345)) {
			for(int i=0;i<173;i++)original.nextInt();original.nextGaussian();
			RandomEngine copy=original.clone();assertSameStream(original,copy);
			original.setSeed(76543);original.nextGaussian();
			ByteArrayOutputStream bytes=new ByteArrayOutputStream();
			try(ObjectOutputStream out=new ObjectOutputStream(bytes)){out.writeObject(original);}
			try(ObjectInputStream in=new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))){copy=(RandomEngine)in.readObject();}
			assertSameStream(original,copy);
			original.nextGaussian();original.setSeed(12345);
			for(RandomEngine fresh:engines(12345))if(fresh.getClass()==original.getClass())assertSameStream(original,fresh);
		}
	}
	private static void assertSameStream(RandomEngine first,RandomEngine second) {
		for(int i=0;i<300;i++) {
			assertEquals(first.nextGaussian(),second.nextGaussian(),0);
			assertEquals(first.nextInt(),second.nextInt());assertEquals(first.nextLong(),second.nextLong());
			assertEquals(first.nextDouble(),second.nextDouble(),0);assertEquals(first.nextFloat(),second.nextFloat(),0);
			assertEquals(first.nextBoolean(),second.nextBoolean());
		}
	}
	@Test public void inheritedRandomMethodsUseTheActualEngineState() {
		for(RandomEngine engine:engines(9182)) {
			RandomEngine reference=engine.clone();assertEquals(reference.nextInt()<0,engine.nextBoolean());
			assertEquals(reference.nextInt(),engine.nextInt());
		}
	}
	@Test public void boundedDrawsHaveValidRangesAndRejectInvalidBounds() {
		for(RandomEngine engine:engines(4711)) {
			for(int bound:new int[]{1,2,7,(1<<30)+1,Integer.MAX_VALUE})for(int i=0;i<10000;i++){int v=engine.nextInt(bound);assertTrue(v>=0&&v<bound);}
			for(long bound:new long[]{1,2,7,(1L<<62)+1,Long.MAX_VALUE})for(int i=0;i<10000;i++){long v=engine.nextLong(bound);assertTrue(v>=0&&v<bound);}
			for(int bound:new int[]{0,-1,Integer.MIN_VALUE})try{engine.nextInt(bound);fail("invalid int bound");}catch(IllegalArgumentException expected){}
			for(long bound:new long[]{0,-1,Long.MIN_VALUE})try{engine.nextLong(bound);fail("invalid long bound");}catch(IllegalArgumentException expected){}
		}
	}
	@Test public void uniformsAndGaussiansHaveNondegenerateMoments() {
		for(RandomEngine engine:engines(20260908)) {
			int count=200000,negative=0;double sum=0,squares=0,normalSum=0,normalSquares=0;
			for(int i=0;i<count;i++) {
				double x=engine.nextDouble(),g=engine.nextGaussian();float f=engine.nextFloat();
				assertTrue(x>=0&&x<1);assertTrue(f>=0&&f<1);assertTrue(Double.isFinite(g));
				sum+=x;squares+=x*x;normalSum+=g;normalSquares+=g*g;if(g<0)negative++;
			}
			assertEquals(engine.getClass().getName(),.5,sum/count,.004);
			assertEquals(1.0/3,squares/count,.004);assertEquals(0,normalSum/count,.02);assertEquals(1,normalSquares/count,.03);
			assertTrue(negative>count*.48&&negative<count*.52);
		}
	}
}
