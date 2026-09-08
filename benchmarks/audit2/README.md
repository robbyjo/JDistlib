# Reproducing the follow-up numerical audit

Run from the repository root. JDK, R and optional Python/mpmath are external
tools; normal `gradlew check` uses frozen fixtures and requires neither R nor
Python. The R package installer uses an isolated build-directory library.

```powershell
.\gradlew.bat check javadoc
Rscript benchmarks/audit2/install-reference-packages.R
Rscript benchmarks/audit2/math/generate-reference.R
Rscript benchmarks/audit2/inference/generate-reference.R
```

Additional scalar and joint commands, checked package versions, discrepancy
adjudication and workload definitions are in the component `AUDIT.md` files.
The installer queries current CRAN; use the recorded package versions when
reproducing the historical numbers exactly.

## Pristine Java timing baseline

Build baseline classes from the named commit rather than copying a mutable
Gradle class directory. First build the current all-in-one distribution JAR
with `gradlew check`, then:

```powershell
git archive --format=tar --output=build/audit2/pristine-source.tar 4bdbb8a src/main/java
New-Item -ItemType Directory -Force build/audit2/pristine-source,build/audit2/pristine-main | Out-Null
tar -xf build/audit2/pristine-source.tar -C build/audit2/pristine-source
$auditSources = Get-ChildItem build/audit2/pristine-source/src/main/java -Recurse -Filter *.java |
    ForEach-Object { $_.FullName.Replace('\','/') }
[IO.File]::WriteAllLines((Join-Path (Get-Location) 'build/audit2/pristine-sources.txt'), $auditSources)
javac --release 8 -Xlint:-options -cp distribution/all/build/libs/jdistlib-all-0.10.1-SNAPSHOT.jar -d build/audit2/pristine-main '@build/audit2/pristine-sources.txt'
javac --release 8 -Xlint:-options -cp build/classes/java/main -d build/audit2 benchmarks/audit2/math/MathBenchmark.java
java -cp 'build/audit2;build/audit2/pristine-main;distribution/all/build/libs/jdistlib-all-0.10.1-SNAPSHOT.jar' MathBenchmark
java -cp 'build/audit2;build/classes/java/main;distribution/all/build/libs/jdistlib-all-0.10.1-SNAPSHOT.jar' MathBenchmark
Rscript benchmarks/audit2/math/benchmark.R
```

Run timing workloads separately to avoid competing audit processes. Results
depend on the host, JVM warmup, R versions, input shape and requested accuracy.
