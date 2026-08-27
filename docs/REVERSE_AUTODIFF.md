# Reverse-mode automatic differentiation

`ReverseTape` is JDistlib's allocation-conscious reverse-mode backend for
Java-authored HMC and NUTS targets. Nodes are integer handles stored in reusable
primitive arrays. `reset()` clears a completed evaluation without releasing its
arena; `mark()` and `rewind(mark)` discard temporary subgraphs. Create one tape
per sampler chain because a tape is intentionally not thread-safe.

The sampler-facing wrapper is `ReverseModeLogDensity`:

```java
ReverseModeLogDensity target = new ReverseModeLogDensity(2, (tape, x) -> {
    int quadratic = tape.add(tape.multiply(x[0], x[0]),
                             tape.multiply(x[1], x[1]));
    return tape.multiply(quadratic, -0.5);
});

double[] gradient = new double[2];
double logDensity = target.logDensityAndGradient(
    new double[] {0.2, -0.4}, gradient);
```

`ReverseModeLogDensity` implements `DifferentiableLogDensity` and
`GradientProvider`, so it can be passed directly to `HamiltonianMonteCarlo` or
`NoUTurnSampler`. The wrapper reuses both its parameter-handle buffer and tape
arena. It grows the arena geometrically when required and retains that capacity
for later evaluations.

Compiled Stan-inspired scripts still use their existing forward-mode evaluator.
This keeps the interpreter simple and is often efficient for small parameter
vectors. Reverse mode is preferable when a scalar target depends on many
parameters. Automatic lowering of all script expressions to the reverse tape is
a remaining integration task; the reverse backend itself is production API and
is already usable by hand-written or generated Java factors.

Run the checked comparison benchmark with:

```text
./gradlew compileDocumentationExamples
java -cp build/libs/jdistlib-0.8.3-SNAPSHOT.jar;build/documentation-examples examples.ReverseAutodiffBenchmark
```

Use `:` instead of `;` as the classpath separator on Unix-like systems. The
benchmark warms up both engines, evaluates the same eight-parameter target
100,000 times through the script forward engine and primitive reverse tape, and
reports nanoseconds per value-and-gradient evaluation plus retained tape
capacity. It is a smoke benchmark, not a substitute for JMH when making release
performance claims.
