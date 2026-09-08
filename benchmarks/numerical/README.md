# Gamma coefficient allocation benchmark

On 2026-09-08, the baseline at `067d9c5` allocated fresh immutable coefficient
arrays for every call to several special functions. Those arrays now live in
private static fields. Constants and arithmetic are unchanged.

The standalone `CoefficientBenchmark.java` tests changing inputs across 4,096
values, consumes each result, warms each workload for four million calls, then
reports the median of seven one-million-call samples. Three independent JVM
forks per version were run sequentially. Timing includes the scalar Java loop;
this is a focused microbenchmark, not a claim about every distribution or end
to end application. JVM: Oracle HotSpot 25+37, Windows amd64, `-Xms256m -Xmx256m`.

| Workload | Before ns/call | After ns/call | Speedup | Before bytes/call | After bytes/call |
|---|---:|---:|---:|---:|---:|
| gamma | 28.29 | 17.14 | 1.65x | 352 | 0 |
| log gamma | 35.59 | 29.39 | 1.21x | 352 | 0 |
| Stirling correction | 30.22 | 5.58 | 5.41x | 432.73 | 0 |
| Gamma density | 107.14 | 83.53 | 1.28x | 796.23 | 28.23 |
| Poisson density | 48.83 | 45.04 | 1.08x | 441.90 | 30.07 |
| Beta density | 239.69 | 147.82 | 1.62x | 2304 | 0 |

Values above are medians of the three fork medians. Some fork medians varied
substantially (for example, log gamma ranged from 35.32 to 58.20 ns before and
29.25 to 47.42 ns after); all individual measurements are in
`coefficient-results-2026-09-08.csv`. The tiny change to `lgammacor` is not claimed
as a speed improvement: HotSpot already eliminated its table allocation.
Allocations were measured with HotSpot's `ThreadMXBean` allocation counter.

The benchmark fingerprints 65,536 additional inputs for each of its seven
workloads. Their fingerprints, calculated from 458,752 result bit patterns,
agree between the two versions, as do timed checksums. The regression test `MathCoefficientRegressionTest` also
checks half-integer values, recurrence, reflection and correction identities
independently of R. Broader R comparisons live in `benchmarks/audit`.

To repeat on two compiled checkouts, compile the same harness once, then run
it with each version's classes (the benchmark itself is not on the timed path
of class loading):

```powershell
javac -cp PATH_TO_BASELINE_CLASSES -d build/coefficient-benchmark benchmarks/numerical/CoefficientBenchmark.java
java -Xms256m -Xmx256m -cp 'build/coefficient-benchmark;PATH_TO_BASELINE_CLASSES' CoefficientBenchmark 1000000
java -Xms256m -Xmx256m -cp 'build/coefficient-benchmark;PATH_TO_CANDIDATE_CLASSES' CoefficientBenchmark 1000000
```

For the recorded results, all baseline class files were copied into the
candidate directory and only `MathFunctions.java` was recompiled there. This
isolates the coefficient change from the audit's other numerical fixes.
Repeat each command in three fresh JVMs. For POSIX shells, use `:` instead of
`;` as the classpath separator.
