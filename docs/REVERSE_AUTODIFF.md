# Reverse-mode automatic differentiation

`ReverseTape` is JDistlib's allocation-conscious reverse-mode backend for both
compiled scripts and Java-authored HMC/NUTS targets. Nodes are integer handles
stored in reusable primitive arrays. `reset()` clears a completed evaluation
without releasing its arena; `mark()` and `rewind(mark)` discard temporary
subgraphs. N-ary atomic nodes retain their parent handles and partials in a
second reusable edge arena. Create one tape per sampler chain because a tape is
intentionally not thread-safe.

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

Compiled Stan-inspired scripts lower expression evaluation directly onto a
thread-local tape. Every constrained coordinate is registered once, the tape is
reset between density calls, and a single reverse sweep produces the model
gradient. Normal, Student-t, dot product, squared distance, matrix-normal,
external-function, and numerical-solver results use atomic kernels so their
intermediate scalar algebra does not inflate the tape. Solver callbacks suspend
the outer tape while they calculate values and sensitivities, then attach the
result as an atomic node.

The older forward `Diff` path is retained internally for callback calculations
and finite-difference validation. This dual backend is an implementation detail;
`BayesianModel.logDensityAndGradient` is the stable script-facing API.

Run the checked comparison benchmark with:

```text
./gradlew compileDocumentationExamples
java -cp build/libs/*;build/documentation-examples examples.ReverseAutodiffBenchmark
```

Use `:` instead of `;` as the classpath separator on Unix-like systems. The
benchmark warms up both paths, evaluates the same eight-parameter target
100,000 times through compiled-script lowering and a hand-authored primitive tape, and
reports nanoseconds per value-and-gradient evaluation plus retained tape
capacity. It is a smoke benchmark, not a substitute for JMH when making release
performance claims.
