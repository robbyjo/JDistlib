# Matched R / Java distribution audit

This harness compares 15 scalar distribution families, with density, CDF and
quantile operations, both CDF/quantile tails, and ordinary/log representations.
Each of 150 workloads has 1,024 permuted, evenly spaced inputs with fixed
parameters. The R generator rejects invalid or nonfinite references and emits
17 significant digits. This is an ordinary-parameter sweep; dedicated JUnit
identity tests and the high-precision generators cover exceptional parameters.

Timing uses identical inputs. R calls the vectorized `stats` functions; Java
loops over static scalar methods and includes workload dispatch and summation.
Parsing, I/O and process startup are excluded. R calibrates repeated batches
to at least 200 ms (up to 65,536 repetitions) to reduce clock quantization;
Java uses at least 40 ms (up to 8,192 repetitions) with its nanosecond clock.
Both report five-sample medians. Java also warms each workload for at least
100 ms. The broad comparison uses one JVM
per version; the isolated coefficient benchmark in `../numerical` uses three
fresh JVMs per version and measures allocation as well. These are host/workload
measurements, not universal performance guarantees.

Run R and each Java process sequentially on an otherwise quiet machine:

```powershell
& 'C:/Program Files/R/R-4.6.1/bin/Rscript.exe' benchmarks/audit/compare-r.R build/r-audit 1024
./gradlew.bat classes
javac --release 8 -Xlint:-options -cp build/classes/java/main -d build/audit-runner benchmarks/audit/DistributionAudit.java
java -Xms256m -Xmx512m -cp 'build/audit-runner;BASELINE_CLASSES' DistributionAudit build/r-audit/reference.tsv build/r-audit/baseline
java -Xms256m -Xmx512m -cp 'build/audit-runner;build/classes/java/main' DistributionAudit build/r-audit/reference.tsv build/r-audit/candidate
python benchmarks/audit/validate-ncbeta.py build/r-audit/candidate/differences.tsv build/r-audit/ncbeta-validation
python benchmarks/audit/summarize.py build/r-audit benchmarks/audit/results-2026-09-08.csv
```

`BASELINE_CLASSES` means the compiled classes of a separate checkout at
`067d9c5`. On POSIX use `:` as the classpath separator. The validator requires
mpmath (recorded run: 1.4.1); the summary script uses only the Python standard
library. An optional third Java argument `accuracy-only` skips timings.

The Java discrepancy screen intentionally exits **1** when R differs by more
than 5e-10 relative error (central) or 2e-7 (noncentral), even for known R bugs.
Do not treat that screen as a reason to copy R. In the recorded run, all 584
flagged values were noncentral-beta CDFs, already improved in JDistlib before
this audit. Run the high-precision validator to resolve them. Its exit status
and JSON summary establish whether Java agrees with the independent reference.
Do not silently discard unexpected discrepancies or loosen tolerances.

The committed CSV joins all 150 timing/accuracy summaries. Full generated
reference values, individual discrepancies and high-precision evidence are
written under the ignored `build/r-audit` directory. See
[`docs/NUMERICAL_AUDIT_2026-09-08.md`](../../docs/NUMERICAL_AUDIT_2026-09-08.md)
for findings, environment and limitations.
