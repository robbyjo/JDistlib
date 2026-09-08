param(
 [string]$BaselineClasses = 'build/audit2/pristine-main',
 [string]$Rscript = 'C:/Program Files/R/R-4.6.1/bin/Rscript.exe',
 [int]$Draws = 16384
)
$ErrorActionPreference = 'Stop'
# Run Gradle classes first. Build BaselineClasses from git archive 4bdbb8a as described in ../README.md.
$output = Join-Path (Get-Location) 'build/audit2/inference'
$library = Join-Path $output 'lib'
New-Item -ItemType Directory -Force -Path $library | Out-Null
$cache = Join-Path $env:USERPROFILE '.gradle/caches/modules-2/files-2.1'
foreach($module in @('com.github.wendykierp/JTransforms','pl.edu.icm/JLargeArrays','org.apache.commons/commons-math3')) {
 $jar = Get-ChildItem -LiteralPath (Join-Path $cache $module) -Recurse -Filter '*.jar' | Select-Object -First 1
 if(!$jar) { throw "Run gradlew classes first: missing $module" }
 Copy-Item -LiteralPath $jar.FullName -Destination $library -Force
}
& javac --release 8 -Xlint:-options -cp build/classes/java/main -d $output benchmarks/audit2/inference/InferenceBenchmark.java
if($LASTEXITCODE) { throw 'Benchmark compilation failed' }
if(Test-Path -LiteralPath $BaselineClasses) {
 & java -cp "$output;$BaselineClasses;$library/*" InferenceBenchmark $Draws
 if($LASTEXITCODE) { throw 'Baseline benchmark failed' }
}
& java -cp "$output;build/classes/java/main;$library/*" InferenceBenchmark $Draws
if($LASTEXITCODE) { throw 'Candidate benchmark failed' }
& $Rscript benchmarks/audit2/inference/benchmark.R $Draws
if($LASTEXITCODE) { throw 'R benchmark failed' }
